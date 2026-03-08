// MLX-backed PJRT executable: StableHLO → mlx::core op-by-op lowering

#include "pjrt_plugin/mlx_executable.h"

#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <iostream>
#include <unordered_set>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlx/fft.h"
#include "mlx/mlx.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "pjrt_plugin/issue_url.h"
#include "pjrt_plugin/mlx_type_utils.h"
#include "pjrt_plugin/stablehlo_parser.h"

namespace mlxc = mlx::core;

namespace jax_mlx {

// ValueMap: MLIR Value → MLX array (lazy, not yet evaluated)
// Uses void* (v.getImpl()) as key since mlxc::array is not default-constructible.
using ValueMap = std::unordered_map<void*, mlxc::array>;

static void* valKey(mlir::Value v) {
    return v.getImpl();
}

// ============================================================================
// Helpers
// ============================================================================

static mlxc::Shape toMlxShape(mlir::RankedTensorType type) {
    mlxc::Shape s;
    for (int64_t d : type.getShape()) s.push_back(static_cast<int>(d));
    return s;
}

static std::vector<int64_t> normalizeStridesToElements(const std::vector<int64_t>& dims,
                                                       const std::vector<int64_t>& rawStrides,
                                                       int64_t dataSizeElems,
                                                       size_t elemSize) {
    if (rawStrides.empty()) return rawStrides;
    auto maxOffset = [&](const std::vector<int64_t>& strides) {
        int64_t off = 0;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] > 1) off += (dims[i] - 1) * std::abs(strides[i]);
        }
        return off;
    };
    if (maxOffset(rawStrides) < dataSizeElems || elemSize <= 1) return rawStrides;

    std::vector<int64_t> normalized = rawStrides;
    bool divisible = true;
    for (auto& s : normalized) {
        if (s % static_cast<int64_t>(elemSize) != 0) {
            divisible = false;
            break;
        }
        s /= static_cast<int64_t>(elemSize);
    }
    if (!divisible) return rawStrides;
    return maxOffset(normalized) < dataSizeElems ? normalized : rawStrides;
}

static void copyStridedToLinearBytes(const uint8_t* src,
                                     uint8_t* dst,
                                     const std::vector<int64_t>& dims,
                                     const std::vector<int64_t>& stridesElems,
                                     size_t elemSize,
                                     size_t dim,
                                     int64_t srcIndex,
                                     size_t& dstOffset) {
    if (dim == dims.size()) {
        auto* srcBytes = reinterpret_cast<const std::byte*>(src);
        std::ptrdiff_t byteOffset =
            static_cast<std::ptrdiff_t>(srcIndex) * static_cast<std::ptrdiff_t>(elemSize);
        std::memcpy(dst + dstOffset,
                    srcBytes + byteOffset,
                    elemSize);
        dstOffset += elemSize;
        return;
    }
    for (int64_t i = 0; i < dims[dim]; ++i) {
        copyStridedToLinearBytes(src,
                                 dst,
                                 dims,
                                 stridesElems,
                                 elemSize,
                                 dim + 1,
                                 srcIndex + i * stridesElems[dim],
                                 dstOffset);
    }
}

// Apply an arbitrary axis permutation.
static mlxc::array permuteAxes(mlxc::array input, const std::vector<int>& perm) {
    if (perm.empty()) return input;
    return mlxc::transpose(input, perm);
}

// Build an MLX array from a DenseElementsAttr constant
static mlxc::array makeConstant(mlir::stablehlo::ConstantOp op) {
    auto denseAttr = mlir::cast<mlir::DenseElementsAttr>(op.getValue());
    auto tensorType = mlir::cast<mlir::RankedTensorType>(denseAttr.getType());

    mlxc::Shape shape = toMlxShape(tensorType);
    auto elemType = tensorType.getElementType();
    auto dtype = MlirTypeToMlx(elemType);

    // Handle bool (i1): MLIR stores packed bits, convert to uint8
    if (elemType.isInteger(1)) {
        std::vector<uint8_t> data;
        for (bool v : denseAttr.getValues<bool>()) data.push_back(v ? 1 : 0);
        return mlxc::array(data.data(), shape, mlxc::bool_);
    }

    // All other types: copy raw bytes (or expand splats) and hand ownership to MLX.
    auto rawData = denseAttr.getRawData();
    int64_t numElems = tensorType.getNumElements();
    size_t elemBytes = static_cast<size_t>(dtype.size());
    size_t nbytes = rawData.size();
    if (denseAttr.isSplat() && numElems > 0) {
        nbytes = static_cast<size_t>(numElems) * elemBytes;
    }
    void* buf = std::malloc(nbytes > 0 ? nbytes : 1);
    if (nbytes > 0) {
        if (denseAttr.isSplat() && numElems > 0) {
            const char* one = rawData.data();
            for (int64_t i = 0; i < numElems; ++i) {
                std::memcpy(static_cast<char*>(buf) + static_cast<size_t>(i) * elemBytes,
                            one,
                            elemBytes);
            }
        } else {
            std::memcpy(buf, rawData.data(), nbytes);
        }
    }
    return mlxc::array(buf, shape, dtype, [](void* p) { std::free(p); });
}

// stablehlo.broadcast_in_dim: insert size-1 dims then broadcast_to
static mlxc::array broadcastInDim(mlxc::array input,
                                 llvm::ArrayRef<int64_t> bdcDims,
                                 mlxc::Shape outShape) {
    if (input.ndim() == 0 && bdcDims.empty()) {
        // MLX broadcast_to on reshaped scalars can produce sparse-looking outputs.
        // Force scalar broadcasting via arithmetic expansion.
        return mlxc::add(mlxc::zeros(outShape, input.dtype()), input);
    }
    int outRank = static_cast<int>(outShape.size());
    mlxc::Shape interShape(outRank, 1);
    int inDimIdx = 0;
    for (int64_t d : bdcDims)
        interShape[static_cast<size_t>(d)] = input.shape()[inDimIdx++];
    auto reshaped = mlxc::reshape(input, interShape);
    return mlxc::add(mlxc::zeros(outShape, input.dtype()), reshaped);
}

// stablehlo.dot_general lowered through MLX einsum.
static mlxc::array dotGeneral(mlxc::array lhs, mlxc::array rhs,
                              mlir::stablehlo::DotDimensionNumbersAttr dimNums) {
    auto lhsBatchArr = dimNums.getLhsBatchingDimensions();
    auto rhsBatchArr = dimNums.getRhsBatchingDimensions();
    auto lhsContractArr = dimNums.getLhsContractingDimensions();
    auto rhsContractArr = dimNums.getRhsContractingDimensions();

    static constexpr const char* kSymbols =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static constexpr int kSymbolCount = 52;

    if (lhs.ndim() + rhs.ndim() > kSymbolCount) {
        // Conservative fallback for unusually high-rank tensors.
        return mlxc::matmul(lhs, rhs);
    }

    std::vector<char> lhsLabels(lhs.ndim(), '\0');
    std::vector<char> rhsLabels(rhs.ndim(), '\0');
    int nextSymbol = 0;
    auto allocSymbol = [&]() -> char { return kSymbols[nextSymbol++]; };

    for (size_t i = 0; i < lhsBatchArr.size(); ++i) {
        char sym = allocSymbol();
        lhsLabels[static_cast<size_t>(lhsBatchArr[i])] = sym;
        rhsLabels[static_cast<size_t>(rhsBatchArr[i])] = sym;
    }
    for (size_t i = 0; i < lhsContractArr.size(); ++i) {
        char sym = allocSymbol();
        lhsLabels[static_cast<size_t>(lhsContractArr[i])] = sym;
        rhsLabels[static_cast<size_t>(rhsContractArr[i])] = sym;
    }
    for (int i = 0; i < lhs.ndim(); ++i)
        if (lhsLabels[static_cast<size_t>(i)] == '\0')
            lhsLabels[static_cast<size_t>(i)] = allocSymbol();
    for (int i = 0; i < rhs.ndim(); ++i)
        if (rhsLabels[static_cast<size_t>(i)] == '\0')
            rhsLabels[static_cast<size_t>(i)] = allocSymbol();

    std::string lhsSub(lhsLabels.begin(), lhsLabels.end());
    std::string rhsSub(rhsLabels.begin(), rhsLabels.end());

    std::string outSub;
    outSub.reserve(static_cast<size_t>(lhs.ndim() + rhs.ndim()));
    std::set<int> lhsBatchSet(lhsBatchArr.begin(), lhsBatchArr.end());
    std::set<int> lhsContractSet(lhsContractArr.begin(), lhsContractArr.end());
    std::set<int> rhsBatchSet(rhsBatchArr.begin(), rhsBatchArr.end());
    std::set<int> rhsContractSet(rhsContractArr.begin(), rhsContractArr.end());

    for (auto d : lhsBatchArr)
        outSub.push_back(lhsLabels[static_cast<size_t>(d)]);
    for (int i = 0; i < lhs.ndim(); ++i)
        if (!lhsBatchSet.count(i) && !lhsContractSet.count(i))
            outSub.push_back(lhsLabels[static_cast<size_t>(i)]);
    for (int i = 0; i < rhs.ndim(); ++i)
        if (!rhsBatchSet.count(i) && !rhsContractSet.count(i))
            outSub.push_back(rhsLabels[static_cast<size_t>(i)]);

    return mlxc::einsum(lhsSub + "," + rhsSub + "->" + outSub, {lhs, rhs});
}

static std::vector<int> toIntVector(std::optional<llvm::ArrayRef<int64_t>> arr,
                                    int defaultValue,
                                    int sizeHint) {
    if (!arr.has_value()) return std::vector<int>(sizeHint, defaultValue);
    std::vector<int> out;
    out.reserve(arr->size());
    for (int64_t v : *arr) out.push_back(static_cast<int>(v));
    return out;
}

static std::pair<std::vector<int>, std::vector<int>> parsePadding(
    std::optional<mlir::DenseIntElementsAttr> attr, int spatialDims) {
    std::vector<int> lo(spatialDims, 0), hi(spatialDims, 0);
    if (!attr.has_value()) return {lo, hi};

    std::vector<int64_t> vals;
    for (int64_t v : attr->getValues<int64_t>()) vals.push_back(v);
    if (vals.size() != static_cast<size_t>(2 * spatialDims)) return {lo, hi};
    for (int i = 0; i < spatialDims; ++i) {
        lo[i] = static_cast<int>(vals[static_cast<size_t>(2 * i)]);
        hi[i] = static_cast<int>(vals[static_cast<size_t>(2 * i + 1)]);
    }
    return {lo, hi};
}

static int findSpatialPos(llvm::ArrayRef<int64_t> spatialDims, int64_t axis) {
    for (int i = 0; i < static_cast<int>(spatialDims.size()); ++i) {
        if (spatialDims[static_cast<size_t>(i)] == axis) return i;
    }
    return -1;
}

static mlxc::array HandleConvolution(mlxc::array lhs, mlxc::array rhs,
                                   mlir::stablehlo::ConvolutionOp convOp) {
    auto dimNums = convOp.getDimensionNumbers();
    int spatialDims = static_cast<int>(dimNums.getInputSpatialDimensions().size());

    // MLX expects input [N, spatial..., C_in], weights [C_out, spatial..., C_in].
    std::vector<int> inputPerm;
    inputPerm.reserve(static_cast<size_t>(lhs.ndim()));
    inputPerm.push_back(static_cast<int>(dimNums.getInputBatchDimension()));
    for (int64_t d : dimNums.getInputSpatialDimensions()) inputPerm.push_back(static_cast<int>(d));
    inputPerm.push_back(static_cast<int>(dimNums.getInputFeatureDimension()));
    auto lhsCanonical = permuteAxes(lhs, inputPerm);

    std::vector<int> kernelPerm;
    kernelPerm.reserve(static_cast<size_t>(rhs.ndim()));
    kernelPerm.push_back(static_cast<int>(dimNums.getKernelOutputFeatureDimension()));
    for (int64_t d : dimNums.getKernelSpatialDimensions()) kernelPerm.push_back(static_cast<int>(d));
    kernelPerm.push_back(static_cast<int>(dimNums.getKernelInputFeatureDimension()));
    auto rhsCanonical = permuteAxes(rhs, kernelPerm);

    auto strides = toIntVector(convOp.getWindowStrides(), 1, spatialDims);
    auto lhsDil = toIntVector(convOp.getLhsDilation(), 1, spatialDims);
    auto rhsDil = toIntVector(convOp.getRhsDilation(), 1, spatialDims);
    auto [padLo, padHi] = parsePadding(convOp.getPadding(), spatialDims);

    bool flip = false;
    if (auto reversal = convOp.getWindowReversal(); reversal.has_value()) {
        for (bool v : *reversal)
            if (v) flip = true;
    }

    uint64_t batchGroups = convOp.getBatchGroupCount();
    int featureGroups = static_cast<int>(convOp.getFeatureGroupCount());

    mlxc::array outCanonical = mlxc::array({});
    if (batchGroups == 1) {
        outCanonical = mlxc::conv_general(lhsCanonical, rhsCanonical,
                                         strides, padLo, padHi,
                                         rhsDil, lhsDil, featureGroups, flip);
    } else {
        // batch_group_count > 1: used by JAX for weight gradients of grouped/depthwise
        // convolutions. Split LHS along batch (axis 0) and RHS along output-feature
        // (axis 0) into batchGroups groups, convolve each pair independently, then
        // concatenate the results along the output-feature axis (last canonical axis).
        int64_t G = static_cast<int64_t>(batchGroups);
        int64_t N = lhsCanonical.shape()[0];
        int64_t C_out = rhsCanonical.shape()[0];
        if (N % G != 0 || C_out % G != 0) {
            throw std::invalid_argument(
                "stablehlo.convolution batch_group_count does not evenly divide "
                "batch or output-feature dimension");
        }
        int64_t bPerGroup = N / G;
        int64_t coutPerGroup = C_out / G;

        // Build full-range start/stop helpers for slicing all non-split dims.
        int rank = lhsCanonical.ndim();
        auto fullSlice = [&](mlxc::array arr, int axis, int64_t lo, int64_t hi) {
            mlxc::Shape starts(static_cast<size_t>(arr.ndim()), 0);
            mlxc::Shape stops;
            stops.reserve(static_cast<size_t>(arr.ndim()));
            for (int d = 0; d < arr.ndim(); ++d)
                stops.push_back(static_cast<int>(arr.shape()[d]));
            starts[static_cast<size_t>(axis)] = static_cast<int>(lo);
            stops[static_cast<size_t>(axis)] = static_cast<int>(hi);
            return mlxc::slice(arr, starts, stops);
        };

        std::vector<mlxc::array> pieces;
        pieces.reserve(static_cast<size_t>(G));
        for (int64_t g = 0; g < G; ++g) {
            auto lhsG = fullSlice(lhsCanonical, 0, g * bPerGroup,   (g + 1) * bPerGroup);
            auto rhsG = fullSlice(rhsCanonical, 0, g * coutPerGroup, (g + 1) * coutPerGroup);
            pieces.push_back(mlxc::conv_general(lhsG, rhsG,
                                               strides, padLo, padHi,
                                               rhsDil, lhsDil, featureGroups, flip));
        }
        // Each piece: [bPerGroup, spatial..., coutPerGroup]
        // Concatenate along the output-feature axis (last = rank - 1 in canonical form).
        outCanonical = mlxc::concatenate(pieces, rank - 1);
    }

    int outRank = outCanonical.ndim();
    std::vector<int> outPerm(static_cast<size_t>(outRank), 0);
    for (int axis = 0; axis < outRank; ++axis) {
        if (axis == dimNums.getOutputBatchDimension()) {
            outPerm[static_cast<size_t>(axis)] = 0;
        } else if (axis == dimNums.getOutputFeatureDimension()) {
            outPerm[static_cast<size_t>(axis)] = outRank - 1;
        } else {
            int p = findSpatialPos(dimNums.getOutputSpatialDimensions(), axis);
            if (p < 0) throw std::invalid_argument("invalid output spatial dimensions");
            outPerm[static_cast<size_t>(axis)] = 1 + p;
        }
    }
    return permuteAxes(outCanonical, outPerm);
}

static mlxc::array HandleGather(mlxc::array operand, mlxc::array startIndices,
                              mlir::stablehlo::GatherOp gatherOp) {
    auto dimNums = gatherOp.getDimensionNumbers();
    auto startIndexMap = dimNums.getStartIndexMap();
    auto collapsedSliceDims = dimNums.getCollapsedSliceDims();
    auto operandBatchingDims = dimNums.getOperandBatchingDims();
    auto startIndicesBatchingDims = dimNums.getStartIndicesBatchingDims();
    auto sliceSizes = gatherOp.getSliceSizes();
    int indexVectorDim = static_cast<int>(dimNums.getIndexVectorDim());

    auto normalizedStartIndices = [&]() -> mlxc::array {
        auto idx = startIndices;
        if (indexVectorDim == idx.ndim()) {
            idx = mlxc::expand_dims(idx, {static_cast<int>(idx.ndim())});
        } else if (indexVectorDim < idx.ndim() - 1) {
            std::vector<int> perm;
            perm.reserve(static_cast<size_t>(idx.ndim()));
            for (int i = 0; i < idx.ndim(); ++i)
                if (i != indexVectorDim) perm.push_back(i);
            perm.push_back(indexVectorDim);
            idx = mlxc::transpose(idx, perm);
        }
        return idx;
    };

    // Handle the common "take along one axis" gather pattern used by embeddings.
    if (startIndexMap.size() == 1 && collapsedSliceDims.size() == 1) {
        int axis = static_cast<int>(startIndexMap[0]);
        if (collapsedSliceDims[0] != startIndexMap[0]) {
            throw std::invalid_argument("collapsed axis must match gathered axis");
        }
        if (sliceSizes[static_cast<size_t>(axis)] != 1) {
            throw std::invalid_argument("slice size for gathered axis must be 1");
        }

        auto indices = normalizedStartIndices();
        if (indices.shape().back() != 1) {
            throw std::invalid_argument("index_vector_dim width must be 1");
        }
        indices = mlxc::squeeze(indices, {static_cast<int>(indices.ndim() - 1)});

        if (!operandBatchingDims.empty() || !startIndicesBatchingDims.empty()) {
            // Batched take_along_axis: used by categorical/log_prob style gathers.
            if (operandBatchingDims.size() != startIndicesBatchingDims.size()) {
                throw std::invalid_argument("operand/start_indices batching dims must match");
            }
            if (operandBatchingDims.size() != 1) {
                throw std::invalid_argument("only one batching dim is currently supported");
            }
            int batchAxis = static_cast<int>(operandBatchingDims[0]);
            if (startIndicesBatchingDims[0] != operandBatchingDims[0]) {
                throw std::invalid_argument("batching dims must align");
            }

            for (int i = 0; i < operand.ndim(); ++i) {
                if (i == axis || i == batchAxis) {
                    if (sliceSizes[static_cast<size_t>(i)] != 1) {
                        throw std::invalid_argument(
                            "gather and batching axes must have slice size 1");
                    }
                } else if (sliceSizes[static_cast<size_t>(i)] != operand.shape()[i]) {
                    throw std::invalid_argument(
                        "non-gather/non-batching axes must use full slice");
                }
            }

            if (indices.ndim() != operand.ndim()) {
                throw std::invalid_argument("batched gather indices rank mismatch");
            }
            return mlxc::take_along_axis(operand, indices, axis);
        }

        for (int i = 0; i < operand.ndim(); ++i) {
            if (i != axis && sliceSizes[static_cast<size_t>(i)] != operand.shape()[i]) {
                throw std::invalid_argument("non-gather axes must use full slice");
            }
        }
        return mlxc::take(operand, indices, axis);
    }

    // Two-axis point gather lowering used by MultivariateNormal-like paths.
    if (startIndexMap.size() == 2 && collapsedSliceDims.size() == 2 &&
        operandBatchingDims.empty() && startIndicesBatchingDims.empty()) {
        if (collapsedSliceDims[0] != startIndexMap[0] ||
            collapsedSliceDims[1] != startIndexMap[1]) {
            throw std::invalid_argument("collapsed dims must match gathered dims");
        }
        auto idx = normalizedStartIndices();
        if (idx.shape().back() != 2) {
            throw std::invalid_argument("two-axis gather expects index width 2");
        }
        auto idx0 = mlxc::take(idx, 0, idx.ndim() - 1);
        auto idx1 = mlxc::take(idx, 1, idx.ndim() - 1);
        if (idx0.ndim() != 1 || idx1.ndim() != 1) {
            throw std::invalid_argument("two-axis gather currently expects 1D point indices");
        }
        int n = idx0.shape()[0];

        if (operand.ndim() == 2 && startIndexMap[0] == 0 && startIndexMap[1] == 1) {
            if (sliceSizes.size() != 2 || sliceSizes[0] != 1 || sliceSizes[1] != 1) {
                throw std::invalid_argument("rank-2 point gather requires slice sizes [1, 1]");
            }
            auto rowIdx = mlxc::broadcast_to(
                mlxc::reshape(idx0, {n, 1}), {n, operand.shape()[1]});
            auto rows = mlxc::take_along_axis(operand, rowIdx, 0);
            auto colIdx = mlxc::reshape(idx1, {n, 1});
            return mlxc::squeeze(mlxc::take_along_axis(rows, colIdx, 1), {1});
        }

        if (operand.ndim() == 3 && startIndexMap[0] == 1 && startIndexMap[1] == 2) {
            if (sliceSizes.size() != 3 || sliceSizes[1] != 1 || sliceSizes[2] != 1) {
                throw std::invalid_argument("rank-3 point gather requires unit gathered slices");
            }
            auto rowIdx = mlxc::broadcast_to(
                mlxc::reshape(idx0, {1, n, 1}),
                {operand.shape()[0], n, operand.shape()[2]});
            auto rows = mlxc::take_along_axis(operand, rowIdx, 1);
            auto colIdx = mlxc::broadcast_to(
                mlxc::reshape(idx1, {1, n, 1}), {operand.shape()[0], n, 1});
            return mlxc::squeeze(mlxc::take_along_axis(rows, colIdx, 2), {2});
        }
    }

    throw std::invalid_argument("unsupported gather pattern");
}

static mlxc::array HandleScatter(mlxc::array operand, mlxc::array scatterIndices, mlxc::array updates,
                               mlir::stablehlo::ScatterOp scatterOp) {
    auto dimNums = scatterOp.getScatterDimensionNumbers();
    auto scatterDimsToOperandDims = dimNums.getScatterDimsToOperandDims();
    int indexVectorDim = static_cast<int>(dimNums.getIndexVectorDim());
    auto insertedWindowDims = dimNums.getInsertedWindowDims();
    auto updateWindowDims = dimNums.getUpdateWindowDims();

    std::string scatterFn;
    for (auto& op : scatterOp.getUpdateComputation().front()) {
        auto nm = op.getName().getStringRef();
        if (nm != "stablehlo.return" && nm != "func.return") {
            scatterFn = nm.str();
            break;
        }
    }

    auto normalizedIndices = [&]() -> mlxc::array {
        auto idx = scatterIndices;
        if (indexVectorDim == idx.ndim()) {
            idx = mlxc::expand_dims(idx, {static_cast<int>(idx.ndim())});
        } else if (indexVectorDim < idx.ndim() - 1) {
            std::vector<int> perm;
            perm.reserve(static_cast<size_t>(idx.ndim()));
            for (int i = 0; i < idx.ndim(); ++i)
                if (i != indexVectorDim) perm.push_back(i);
            perm.push_back(indexVectorDim);
            idx = mlxc::transpose(idx, perm);
        }
        return idx;
    };

    auto idx = normalizedIndices();

    // Full-index scalar scatter into a single point of a rank-N tensor.
    if (scatterDimsToOperandDims.size() == static_cast<size_t>(operand.ndim()) &&
        insertedWindowDims.size() == static_cast<size_t>(operand.ndim()) &&
        idx.ndim() == 1 && idx.shape()[0] == operand.ndim()) {
        for (int d = 0; d < operand.ndim(); ++d) {
            if (scatterDimsToOperandDims[static_cast<size_t>(d)] != d ||
                insertedWindowDims[static_cast<size_t>(d)] != d) {
                throw std::invalid_argument("full-index scatter requires identity dims");
            }
        }

        std::vector<int> strides(static_cast<size_t>(operand.ndim()), 1);
        for (int d = operand.ndim() - 2; d >= 0; --d) {
            strides[static_cast<size_t>(d)] =
                strides[static_cast<size_t>(d + 1)] * operand.shape()[d + 1];
        }

        auto linear = mlxc::take(idx, 0, 0) * strides[0];
        for (int d = 1; d < operand.ndim(); ++d) {
            linear = linear + mlxc::take(idx, d, 0) * strides[static_cast<size_t>(d)];
        }

        auto flatOperand = mlxc::reshape(operand, {static_cast<int>(operand.size())});
        auto flatIndex = mlxc::reshape(linear, {1});
        auto flatUpdate = mlxc::reshape(updates, {1});
        auto outFlat =
            scatterFn == "stablehlo.add"
                ? mlxc::scatter_add_axis(flatOperand, flatIndex, flatUpdate, 0)
                : mlxc::put_along_axis(flatOperand, flatIndex, flatUpdate, 0);
        return mlxc::reshape(outFlat, operand.shape());
    }

    // Axis-style scatter/accumulate along a single operand axis.
    if (scatterDimsToOperandDims.size() == 1 && insertedWindowDims.size() == 1 &&
        insertedWindowDims[0] == scatterDimsToOperandDims[0] && idx.shape().back() == 1) {
        int axis = static_cast<int>(scatterDimsToOperandDims[0]);
        auto idxAxis = mlxc::squeeze(idx, {static_cast<int>(idx.ndim() - 1)});
        if (updates.ndim() == idxAxis.ndim()) {
            if (scatterFn == "stablehlo.add") {
                return mlxc::scatter_add_axis(operand, idxAxis, updates, axis);
            }
            return mlxc::put_along_axis(operand, idxAxis, updates, axis);
        }
    }

    // Two-axis point scatter into rank-2 tensors.
    if (scatterDimsToOperandDims.size() == 2 && insertedWindowDims.size() == 2 &&
        scatterDimsToOperandDims[0] == 0 && scatterDimsToOperandDims[1] == 1 &&
        insertedWindowDims[0] == 0 && insertedWindowDims[1] == 1 &&
        operand.ndim() == 2 && idx.shape().back() == 2) {
        auto idx0 = mlxc::take(idx, 0, idx.ndim() - 1);
        auto idx1 = mlxc::take(idx, 1, idx.ndim() - 1);
        if (idx0.ndim() != 1 || idx1.ndim() != 1) {
            throw std::invalid_argument("rank-2 point scatter expects 1D indices");
        }
        int64_t points = idx0.shape()[0];
        auto linear = idx0 * operand.shape()[1] + idx1;
        auto flatOperand = mlxc::reshape(operand, {operand.shape()[0] * operand.shape()[1]});
        auto flatUpdates = mlxc::reshape(updates, {static_cast<int>(points)});
        auto outFlat =
            scatterFn == "stablehlo.add"
                ? mlxc::scatter_add_axis(flatOperand, linear, flatUpdates, 0)
                : mlxc::put_along_axis(flatOperand, linear, flatUpdates, 0);
        return mlxc::reshape(outFlat, {operand.shape()[0], operand.shape()[1]});
    }

    // Batched two-axis point scatter into rank-3 tensors with update_window_dims=[0].
    if (scatterDimsToOperandDims.size() == 2 && insertedWindowDims.size() == 2 &&
        updateWindowDims.size() == 1 && updateWindowDims[0] == 0 &&
        scatterDimsToOperandDims[0] == 1 && scatterDimsToOperandDims[1] == 2 &&
        insertedWindowDims[0] == 1 && insertedWindowDims[1] == 2 &&
        operand.ndim() == 3 && idx.shape().back() == 2) {
        auto idx0 = mlxc::take(idx, 0, idx.ndim() - 1);
        auto idx1 = mlxc::take(idx, 1, idx.ndim() - 1);
        if (idx0.ndim() != 1 || idx1.ndim() != 1) {
            throw std::invalid_argument("rank-3 point scatter expects 1D indices");
        }
        int b = operand.shape()[0];
        int64_t points = idx0.shape()[0];
        auto linear = idx0 * operand.shape()[2] + idx1;

        auto upd = updates;
        if (upd.ndim() == 1 && b == 1) {
            upd = mlxc::reshape(upd, {1, static_cast<int>(points)});
        } else if (upd.ndim() == 2 && upd.shape()[0] == static_cast<int>(points) &&
                   upd.shape()[1] == b) {
            upd = mlxc::transpose(upd, {1, 0});
        }
        if (upd.ndim() != 2 || upd.shape()[0] != b ||
            upd.shape()[1] != static_cast<int>(points)) {
            throw std::invalid_argument("rank-3 point scatter updates must be [B, P]");
        }

        std::vector<mlxc::array> slices;
        slices.reserve(static_cast<size_t>(b));
        for (int bi = 0; bi < b; ++bi) {
            auto opSlice = mlxc::take(operand, bi, 0);
            auto flatOp = mlxc::reshape(opSlice, {operand.shape()[1] * operand.shape()[2]});
            auto updSlice = mlxc::take(upd, bi, 0);
            auto outFlat =
                scatterFn == "stablehlo.add"
                    ? mlxc::scatter_add_axis(flatOp, linear, updSlice, 0)
                    : mlxc::put_along_axis(flatOp, linear, updSlice, 0);
            slices.push_back(mlxc::reshape(outFlat, {operand.shape()[1], operand.shape()[2]}));
        }
        return mlxc::stack(slices, 0);
    }

    // Existing single-axis fallback.
    if (scatterDimsToOperandDims.size() != 1 || insertedWindowDims.size() != 1) {
        throw std::invalid_argument("only single-axis scatter is implemented");
    }
    int axis = static_cast<int>(scatterDimsToOperandDims[0]);
    if (insertedWindowDims[0] != scatterDimsToOperandDims[0]) {
        throw std::invalid_argument("inserted_window_dims must match scatter axis");
    }
    auto idxAxis = idx;
    if (idxAxis.shape().back() != 1) {
        throw std::invalid_argument("index_vector_dim width must be 1");
    }
    idxAxis = mlxc::squeeze(idxAxis, {static_cast<int>(idxAxis.ndim() - 1)});

    std::vector<mlxc::array> indices = {idxAxis};
    std::vector<int> axes = {axis};
    auto updatesForMlx = updates;
    int indexRank = idxAxis.ndim();
    if (updatesForMlx.ndim() == indexRank + operand.ndim() - 1) {
        updatesForMlx = mlxc::expand_dims(updatesForMlx, indexRank + axis);
    }
    if (scatterFn == "stablehlo.add") {
        return mlxc::scatter_add(operand, indices, updatesForMlx, axes);
    }
    if (scatterFn == "stablehlo.subtract" || scatterFn == "stablehlo.multiply" ||
        scatterFn == "stablehlo.divide" || scatterFn == "stablehlo.maximum" ||
        scatterFn == "stablehlo.minimum" || scatterFn == "stablehlo.power") {
        auto current = mlxc::take(operand, idxAxis, axis);
        mlxc::array reduced = current;
        if (scatterFn == "stablehlo.subtract") {
            reduced = mlxc::subtract(current, updates);
        } else if (scatterFn == "stablehlo.multiply") {
            reduced = mlxc::multiply(current, updates);
        } else if (scatterFn == "stablehlo.divide") {
            reduced = mlxc::divide(current, updates);
        } else if (scatterFn == "stablehlo.maximum") {
            reduced = mlxc::maximum(current, updates);
        } else if (scatterFn == "stablehlo.minimum") {
            reduced = mlxc::minimum(current, updates);
        } else if (scatterFn == "stablehlo.power") {
            reduced = mlxc::power(current, updates);
        }

        auto reducedForMlx = reduced;
        if (reducedForMlx.ndim() == indexRank + operand.ndim() - 1) {
            reducedForMlx = mlxc::expand_dims(reducedForMlx, indexRank + axis);
        }
        return mlxc::scatter(operand, indices, reducedForMlx, axes);
    }
    return mlxc::scatter(operand, indices, updatesForMlx, axes);
}

static mlxc::array HandleCholesky(mlxc::array a, mlir::stablehlo::CholeskyOp choleskyOp) {
    bool lower = true;
    if (choleskyOp.getLowerAttr()) lower = choleskyOp.getLower();

    if (a.ndim() < 2) {
        throw std::invalid_argument("cholesky expects rank >= 2");
    }
    int n = a.shape()[a.ndim() - 1];
    int m = a.shape()[a.ndim() - 2];
    if (n != m) throw std::invalid_argument("cholesky expects square matrices");
    if (a.dtype() != mlxc::float32 && a.dtype() != mlxc::float64) {
        throw std::invalid_argument("cholesky only supports f32/f64");
    }

    a.eval();
    std::vector<int64_t> dims(a.shape().begin(), a.shape().end());
    std::vector<int64_t> rawStrides(a.strides().begin(), a.strides().end());
    auto strides = normalizeStridesToElements(
        dims, rawStrides, static_cast<int64_t>(a.data_size()),
        static_cast<size_t>(a.dtype().size()));

    int64_t batch = 1;
    for (int i = 0; i < a.ndim() - 2; ++i) batch *= a.shape()[i];
    int64_t matrixElems = static_cast<int64_t>(n) * static_cast<int64_t>(n);
    int64_t totalElems = batch * matrixElems;

    auto makeOutput = [&](const auto* srcPtr, auto nanValue) -> mlxc::array {
        using T = std::decay_t<decltype(*srcPtr)>;
        std::vector<T> in(static_cast<size_t>(totalElems));
        std::vector<T> out(static_cast<size_t>(totalElems), static_cast<T>(0));

        size_t dstOffset = 0;
        copyStridedToLinearBytes(reinterpret_cast<const uint8_t*>(srcPtr),
                                 reinterpret_cast<uint8_t*>(in.data()),
                                 dims,
                                 strides,
                                 sizeof(T),
                                 0,
                                 0,
                                 dstOffset);

        for (int64_t b = 0; b < batch; ++b) {
            const int64_t base = b * matrixElems;
            bool ok = true;
            for (int i = 0; i < n && ok; ++i) {
                for (int j = 0; j <= i; ++j) {
                    T sum = in[static_cast<size_t>(base + i * n + j)];
                    for (int k = 0; k < j; ++k) {
                        sum -= out[static_cast<size_t>(base + i * n + k)] *
                               out[static_cast<size_t>(base + j * n + k)];
                    }
                    if (i == j) {
                        if (!(sum > static_cast<T>(0)) || std::isnan(sum)) {
                            ok = false;
                            break;
                        }
                        out[static_cast<size_t>(base + i * n + j)] = std::sqrt(sum);
                    } else {
                        T d = out[static_cast<size_t>(base + j * n + j)];
                        out[static_cast<size_t>(base + i * n + j)] = sum / d;
                    }
                }
            }

            if (!ok) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        out[static_cast<size_t>(base + i * n + j)] = nanValue;
                    }
                }
            }

            if (!lower) {
                std::vector<T> upper(static_cast<size_t>(matrixElems), static_cast<T>(0));
                for (int i = 0; i < n; ++i) {
                    for (int j = i; j < n; ++j) {
                        upper[static_cast<size_t>(i * n + j)] =
                            out[static_cast<size_t>(base + j * n + i)];
                    }
                }
                std::copy(upper.begin(),
                          upper.end(),
                          out.begin() + static_cast<std::ptrdiff_t>(base));
            }
        }

        size_t nbytes = out.size() * sizeof(T);
        void* buf = std::malloc(nbytes > 0 ? nbytes : 1);
        if (nbytes > 0) std::memcpy(buf, out.data(), nbytes);
        return mlxc::array(buf, a.shape(), a.dtype(), [](void* p) { std::free(p); });
    };

    if (a.dtype() == mlxc::float32) {
        return makeOutput(a.data<float>(), std::numeric_limits<float>::quiet_NaN());
    }
    return makeOutput(a.data<double>(), std::numeric_limits<double>::quiet_NaN());
}

static mlxc::array HandleTriangularSolve(mlxc::array a, mlxc::array b,
                                       mlir::stablehlo::TriangularSolveOp triSolveOp) {
    bool leftSide = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unitDiagonal = triSolveOp.getUnitDiagonal();
    auto transposeA = triSolveOp.getTransposeA();
    bool isTranspose = (transposeA == mlir::stablehlo::Transpose::TRANSPOSE);
    bool isAdjoint = (transposeA == mlir::stablehlo::Transpose::ADJOINT);

    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("triangular_solve expects rank >= 2 operands");
    }
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("triangular_solve expects matching dtypes");
    }
    if (a.dtype() != mlxc::float32 && a.dtype() != mlxc::float64) {
        throw std::invalid_argument("triangular_solve only supports f32/f64");
    }

    int n = a.shape()[a.ndim() - 1];
    if (a.shape()[a.ndim() - 2] != n) {
        throw std::invalid_argument("triangular_solve expects square A");
    }
    if (a.ndim() != b.ndim()) {
        throw std::invalid_argument("triangular_solve expects equal operand ranks");
    }
    for (int i = 0; i < a.ndim() - 2; ++i) {
        if (a.shape()[i] != b.shape()[i]) {
            throw std::invalid_argument("triangular_solve batch dims must match");
        }
    }

    int bRows = b.shape()[b.ndim() - 2];
    int bCols = b.shape()[b.ndim() - 1];
    if (leftSide) {
        if (bRows != n) throw std::invalid_argument("left triangular_solve requires b.shape[-2] == n");
    } else {
        if (bCols != n) throw std::invalid_argument("right triangular_solve requires b.shape[-1] == n");
    }

    a.eval();
    b.eval();
    std::vector<int64_t> aDims(a.shape().begin(), a.shape().end());
    std::vector<int64_t> bDims(b.shape().begin(), b.shape().end());
    std::vector<int64_t> aRawStrides(a.strides().begin(), a.strides().end());
    std::vector<int64_t> bRawStrides(b.strides().begin(), b.strides().end());
    auto aStrides = normalizeStridesToElements(
        aDims, aRawStrides, static_cast<int64_t>(a.data_size()),
        static_cast<size_t>(a.dtype().size()));
    auto bStrides = normalizeStridesToElements(
        bDims, bRawStrides, static_cast<int64_t>(b.data_size()),
        static_cast<size_t>(b.dtype().size()));

    int64_t batch = 1;
    for (int i = 0; i < a.ndim() - 2; ++i) batch *= a.shape()[i];
    int64_t aMatrixElems = static_cast<int64_t>(n) * static_cast<int64_t>(n);
    int64_t bMatrixElems = static_cast<int64_t>(bRows) * static_cast<int64_t>(bCols);

    auto solveImpl = [&](const auto* aPtr, const auto* bPtr) -> mlxc::array {
        using T = std::decay_t<decltype(*aPtr)>;
        std::vector<T> aData(static_cast<size_t>(batch * aMatrixElems));
        std::vector<T> bData(static_cast<size_t>(batch * bMatrixElems));
        std::vector<T> out(static_cast<size_t>(batch * bMatrixElems), static_cast<T>(0));

        size_t aOff = 0;
        copyStridedToLinearBytes(reinterpret_cast<const uint8_t*>(aPtr),
                                 reinterpret_cast<uint8_t*>(aData.data()),
                                 aDims,
                                 aStrides,
                                 sizeof(T),
                                 0,
                                 0,
                                 aOff);
        size_t bOff = 0;
        copyStridedToLinearBytes(reinterpret_cast<const uint8_t*>(bPtr),
                                 reinterpret_cast<uint8_t*>(bData.data()),
                                 bDims,
                                 bStrides,
                                 sizeof(T),
                                 0,
                                 0,
                                 bOff);

        bool upper = false;
        if (leftSide) upper = (isTranspose || isAdjoint) ? lower : !lower;
        else upper = (isTranspose || isAdjoint) ? !lower : lower;

        const int rhsCols = leftSide ? bCols : bRows;

        for (int64_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
            const int64_t aBase = batchIdx * aMatrixElems;
            const int64_t bBase = batchIdx * bMatrixElems;
            std::vector<T> c(static_cast<size_t>(aMatrixElems));

            // C is the effective left-side matrix in C * Y = B'.
            for (int r = 0; r < n; ++r) {
                for (int cIdx = 0; cIdx < n; ++cIdx) {
                    if (leftSide) {
                        if (isTranspose || isAdjoint) {
                            c[static_cast<size_t>(r * n + cIdx)] =
                                aData[static_cast<size_t>(aBase + cIdx * n + r)];
                        } else {
                            c[static_cast<size_t>(r * n + cIdx)] =
                                aData[static_cast<size_t>(aBase + r * n + cIdx)];
                        }
                    } else {
                        if (isTranspose || isAdjoint) {
                            c[static_cast<size_t>(r * n + cIdx)] =
                                aData[static_cast<size_t>(aBase + r * n + cIdx)];
                        } else {
                            c[static_cast<size_t>(r * n + cIdx)] =
                                aData[static_cast<size_t>(aBase + cIdx * n + r)];
                        }
                    }
                }
            }

            std::vector<T> y(static_cast<size_t>(n * rhsCols), static_cast<T>(0));
            auto bPrime = [&](int row, int col) -> T {
                if (leftSide) {
                    return bData[static_cast<size_t>(bBase + row * bCols + col)];
                }
                // B' = B^T for right-side solves.
                return bData[static_cast<size_t>(bBase + col * bCols + row)];
            };

            for (int col = 0; col < rhsCols; ++col) {
                if (upper) {
                    for (int i = n - 1; i >= 0; --i) {
                        T sum = bPrime(i, col);
                        for (int j = i + 1; j < n; ++j) {
                            sum -= c[static_cast<size_t>(i * n + j)] *
                                   y[static_cast<size_t>(j * rhsCols + col)];
                        }
                        if (!unitDiagonal) sum /= c[static_cast<size_t>(i * n + i)];
                        y[static_cast<size_t>(i * rhsCols + col)] = sum;
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        T sum = bPrime(i, col);
                        for (int j = 0; j < i; ++j) {
                            sum -= c[static_cast<size_t>(i * n + j)] *
                                   y[static_cast<size_t>(j * rhsCols + col)];
                        }
                        if (!unitDiagonal) sum /= c[static_cast<size_t>(i * n + i)];
                        y[static_cast<size_t>(i * rhsCols + col)] = sum;
                    }
                }
            }

            if (leftSide) {
                for (int i = 0; i < n; ++i) {
                    for (int col = 0; col < bCols; ++col) {
                        out[static_cast<size_t>(bBase + i * bCols + col)] =
                            y[static_cast<size_t>(i * rhsCols + col)];
                    }
                }
            } else {
                for (int row = 0; row < bRows; ++row) {
                    for (int col = 0; col < n; ++col) {
                        out[static_cast<size_t>(bBase + row * bCols + col)] =
                            y[static_cast<size_t>(col * rhsCols + row)];
                    }
                }
            }
        }

        size_t nbytes = out.size() * sizeof(T);
        void* buf = std::malloc(nbytes > 0 ? nbytes : 1);
        if (nbytes > 0) std::memcpy(buf, out.data(), nbytes);
        return mlxc::array(buf, b.shape(), b.dtype(), [](void* p) { std::free(p); });
    };

    if (a.dtype() == mlxc::float32) {
        return solveImpl(a.data<float>(), b.data<float>());
    }
    return solveImpl(a.data<double>(), b.data<double>());
}

static mlxc::array HandleFft(mlxc::array input, mlir::stablehlo::FftOp fftOp) {
    auto fftLength = fftOp.getFftLength();
    int nAxes = static_cast<int>(fftLength.size());
    if (nAxes <= 0 || nAxes > input.ndim()) {
        throw std::invalid_argument("fft: invalid fft_length rank");
    }
    std::vector<int> axes;
    axes.reserve(static_cast<size_t>(nAxes));
    int startAxis = input.ndim() - nAxes;
    for (int i = 0; i < nAxes; ++i) axes.push_back(startAxis + i);

    mlxc::Shape lengths;
    lengths.reserve(static_cast<size_t>(nAxes));
    for (int64_t n : fftLength) lengths.push_back(static_cast<int>(n));

    switch (fftOp.getFftType()) {
        case mlir::stablehlo::FftType::FFT:
            return mlxc::fft::fftn(input, lengths, axes);
        case mlir::stablehlo::FftType::IFFT:
            return mlxc::fft::ifftn(input, lengths, axes);
        case mlir::stablehlo::FftType::RFFT:
            return mlxc::fft::rfftn(input, lengths, axes);
        case mlir::stablehlo::FftType::IRFFT:
            return mlxc::fft::irfftn(input, lengths, axes);
        default:
            throw std::invalid_argument("fft: unsupported fft type");
    }
}

static mlxc::array HandleLgamma(mlxc::array x) {
    if (x.dtype() != mlxc::float32 && x.dtype() != mlxc::float64) {
        throw std::invalid_argument("chlo.lgamma only supports f32/f64");
    }
    x.eval();
    std::vector<int64_t> dims(x.shape().begin(), x.shape().end());
    std::vector<int64_t> rawStrides(x.strides().begin(), x.strides().end());
    auto strides = normalizeStridesToElements(
        dims, rawStrides, static_cast<int64_t>(x.data_size()),
        static_cast<size_t>(x.dtype().size()));

    auto apply = [&](const auto* inPtr) -> mlxc::array {
        using T = std::decay_t<decltype(*inPtr)>;
        std::vector<T> in(static_cast<size_t>(x.size()));
        std::vector<T> out(static_cast<size_t>(x.size()));
        size_t off = 0;
        copyStridedToLinearBytes(reinterpret_cast<const uint8_t*>(inPtr),
                                 reinterpret_cast<uint8_t*>(in.data()),
                                 dims,
                                 strides,
                                 sizeof(T),
                                 0,
                                 0,
                                 off);
        for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<T>(std::lgamma(in[i]));
        size_t nbytes = out.size() * sizeof(T);
        void* buf = std::malloc(nbytes > 0 ? nbytes : 1);
        if (nbytes > 0) std::memcpy(buf, out.data(), nbytes);
        return mlxc::array(buf, x.shape(), x.dtype(), [](void* p) { std::free(p); });
    };

    if (x.dtype() == mlxc::float32) return apply(x.data<float>());
    return apply(x.data<double>());
}

static bool SortComparatorDescending(mlir::Region& comparator) {
    for (auto& op : comparator.front()) {
        auto nm = op.getName().getStringRef();
        if (nm == "stablehlo.compare") {
            auto cmp = mlir::cast<mlir::stablehlo::CompareOp>(op);
            auto dir = cmp.getComparisonDirection();
            if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                dir == mlir::stablehlo::ComparisonDirection::GE) {
                return true;
            }
            return false;
        }
    }
    return false;
}

static mlxc::array ReverseAlongAxis(mlxc::array x, int axis) {
    int n = x.shape()[axis];
    auto idx = mlxc::arange(static_cast<double>(n - 1), -1.0, -1.0, mlxc::int32);
    return mlxc::take(x, idx, axis);
}

static std::vector<mlxc::array> HandleSort(std::vector<mlxc::array> inputs,
                                         mlir::stablehlo::SortOp sortOp) {
    if (inputs.empty()) return {};
    int axis = static_cast<int>(sortOp.getDimension());
    if (axis < 0) axis += inputs[0].ndim();
    if (axis < 0 || axis >= inputs[0].ndim()) {
        throw std::invalid_argument("stablehlo.sort: invalid dimension");
    }

    bool descending = SortComparatorDescending(sortOp.getComparator());
    auto indices = mlxc::argsort(inputs[0], axis);
    if (descending) {
        indices = ReverseAlongAxis(indices, axis);
    }

    std::vector<mlxc::array> out;
    out.reserve(inputs.size());
    for (auto& in : inputs) {
        out.push_back(mlxc::take_along_axis(in, indices, axis));
    }
    return out;
}

// Inspect reduction body to find the operation name
static std::string reductionOpName(mlir::Region& body) {
    for (auto& op : body.front()) {
        auto nm = op.getName().getStringRef();
        if (nm != "stablehlo.return" && nm != "func.return") return nm.str();
    }
    return "";
}

static std::optional<mlir::stablehlo::ComparisonDirection>
reductionCompareDirection(mlir::Region& body) {
    for (auto& op : body.front()) {
        if (op.getName().getStringRef() == "stablehlo.compare") {
            auto cmp = mlir::cast<mlir::stablehlo::CompareOp>(op);
            auto dir = cmp.getComparisonDirection();
            if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                dir == mlir::stablehlo::ComparisonDirection::LT) {
                return dir;
            }
        }
    }
    return std::nullopt;
}

// Bitwise NOT: implemented as XOR with all-ones for integer types, logical_not for bool
static mlxc::array bitwiseNot(mlxc::array a) {
    if (a.dtype() == mlxc::bool_) return mlxc::logical_not(a);
    // Use -1 (all-ones in two's complement) as the mask
    auto mask = mlxc::full(a.shape(), -1, a.dtype());
    return mlxc::bitwise_xor(a, mask);
}

// cbrt(x) = sign(x) * |x|^(1/3)
static mlxc::array cbrt(mlxc::array a) {
    auto absA = mlxc::abs(a);
    auto cbrtAbs = mlxc::power(absA, mlxc::array(1.0f / 3.0f, mlxc::float32));
    auto sgn = mlxc::sign(a);
    return mlxc::multiply(sgn, cbrtAbs);
}

// Population count via successive right-shifts
static mlxc::array popcnt(mlxc::array a) {
    int nbits = a.dtype().size() * 8;
    auto count = mlxc::zeros(a.shape(), mlxc::uint32);
    auto one = mlxc::array(1u, mlxc::uint32);
    for (int i = 0; i < nbits; i++) {
        auto shifted = mlxc::right_shift(mlxc::astype(a, mlxc::uint32), mlxc::array((unsigned)i, mlxc::uint32));
        count = mlxc::add(count, mlxc::bitwise_and(shifted, one));
    }
    return mlxc::astype(count, a.dtype());
}

// stablehlo.remainder follows truncating remainder semantics.
static mlxc::array truncRemainder(mlxc::array x, mlxc::array y) {
    auto r = mlxc::remainder(x, y);
    auto zero = mlxc::zeros(r.shape(), r.dtype());
    auto xNeg = mlxc::less(x, zero);
    auto yNeg = mlxc::less(y, zero);
    auto signDiff = mlxc::not_equal(xNeg, yNeg);
    auto nonZero = mlxc::not_equal(r, zero);
    auto adjust = mlxc::logical_and(signDiff, nonZero);
    return mlxc::where(adjust, mlxc::subtract(r, y), r);
}

static mlxc::Dtype unsignedDtypeFor(mlxc::Dtype dt) {
    if (dt == mlxc::int32 || dt == mlxc::uint32) return mlxc::uint32;
    if (dt == mlxc::int64 || dt == mlxc::uint64) return mlxc::uint64;
    return dt;
}

static mlxc::array invalidShiftMask(mlxc::array shift, mlxc::array value) {
    int bits = value.dtype().size() * 8;
    auto zero = mlxc::zeros(shift.shape(), shift.dtype());
    auto bitsArr = mlxc::full(shift.shape(), bits, shift.dtype());
    return mlxc::logical_or(mlxc::less(shift, zero), mlxc::greater_equal(shift, bitsArr));
}

static mlxc::array shiftLeftLikeCpu(mlxc::array value, mlxc::array shift) {
    auto invalid = invalidShiftMask(shift, value);
    auto zeroShift = mlxc::zeros(shift.shape(), shift.dtype());
    auto safeShift = mlxc::where(invalid, zeroShift, shift);
    auto shifted = mlxc::left_shift(value, safeShift);
    auto zeroOut = mlxc::zeros(value.shape(), value.dtype());
    return mlxc::where(invalid, zeroOut, shifted);
}

static mlxc::array shiftRightLogicalLikeCpu(mlxc::array value, mlxc::array shift) {
    auto invalid = invalidShiftMask(shift, value);
    auto zeroShift = mlxc::zeros(shift.shape(), shift.dtype());
    auto safeShift = mlxc::where(invalid, zeroShift, shift);
    auto u = mlxc::astype(value, unsignedDtypeFor(value.dtype()));
    auto shifted = mlxc::right_shift(u, safeShift);
    auto shiftedCast = mlxc::astype(shifted, value.dtype());
    auto zeroOut = mlxc::zeros(value.shape(), value.dtype());
    return mlxc::where(invalid, zeroOut, shiftedCast);
}

static mlxc::array shiftRightArithmeticLikeCpu(mlxc::array value, mlxc::array shift) {
    auto invalid = invalidShiftMask(shift, value);
    auto zeroShift = mlxc::zeros(shift.shape(), shift.dtype());
    auto safeShift = mlxc::where(invalid, zeroShift, shift);
    auto shifted = mlxc::right_shift(value, safeShift);
    auto zero = mlxc::zeros(value.shape(), value.dtype());
    auto signFill = mlxc::where(mlxc::less(value, zero), mlxc::full(value.shape(), -1, value.dtype()), zero);
    return mlxc::where(invalid, signFill, shifted);
}

// ============================================================================
// Main interpreter
// ============================================================================

// Result of the inner interpreter: either an error string or a list of arrays
struct InterpResult {
    std::vector<mlxc::array> outputs;
    std::string error;
    bool ok() const { return error.empty(); }
    static InterpResult Error(const std::string& msg) {
        InterpResult r;
        r.error = msg;
        return r;
    }
};

// Forward declaration so func.call can recurse
static InterpResult interpretBlock(mlir::Block& block,
                                    mlir::ModuleOp module,
                                    ValueMap vm);

static InterpResult interpretFunction(mlir::func::FuncOp func,
                                       mlir::ModuleOp module,
                                       const std::vector<mlxc::array>& inputs);

static void BindBlockArguments(mlir::Block& block,
                                const std::vector<mlxc::array>& inputs,
                                ValueMap& vm) {
    if (inputs.size() != block.getNumArguments()) return;
    for (size_t i = 0; i < inputs.size(); i++)
        vm.emplace(valKey(block.getArgument(i)), inputs[i]);
}

static InterpResult interpretRegion(mlir::Region& region,
                                     mlir::ModuleOp module,
                                     const std::vector<mlxc::array>& inputs,
                                     const ValueMap& parentVm) {
    if (region.empty()) {
        return InterpResult::Error("Region is empty");
    }
    auto& block = region.front();
    if (inputs.size() != block.getNumArguments()) {
        return InterpResult::Error(
            "Region input arity mismatch: expected " +
            std::to_string(block.getNumArguments()) + ", got " +
            std::to_string(inputs.size()));
    }
    ValueMap vm = parentVm;
    BindBlockArguments(block, inputs, vm);
    return interpretBlock(block, module, std::move(vm));
}

static int64_t ScalarToInt64(mlxc::array a) {
    a.eval();
    if (a.dtype() == mlxc::int32) return static_cast<int64_t>(a.item<int32_t>());
    if (a.dtype() == mlxc::uint32) return static_cast<int64_t>(a.item<uint32_t>());
    if (a.dtype() == mlxc::uint64) return static_cast<int64_t>(a.item<uint64_t>());
    if (a.dtype() == mlxc::bool_) return a.item<bool>() ? 1 : 0;
    return a.item<int64_t>();
}

static ExecutionResult runFunction(mlir::func::FuncOp func,
                                    mlir::ModuleOp module,
                                    const std::vector<MlxBuffer*>& inputs,
                                    MlxDevice* device) {
    std::vector<mlxc::array> arrays;
    arrays.reserve(inputs.size());
    for (auto* buf : inputs) arrays.push_back(buf->array());

    auto interp = interpretFunction(func, module, arrays);
    if (!interp.ok()) return ExecutionResult::Error(interp.error);

    mlxc::eval(interp.outputs);
    ExecutionResult result;
    for (auto& arr : interp.outputs) {
        int pjrt_dtype = MlxDtypeToPjrt(arr.dtype());
        std::vector<int64_t> dims;
        for (int d : arr.shape()) dims.push_back(d);
        result.buffers.push_back(
            std::make_unique<MlxBuffer>(device, std::move(arr), pjrt_dtype, dims));
    }
    return result;
}

static InterpResult interpretFunction(mlir::func::FuncOp func,
                                       mlir::ModuleOp module,
                                       const std::vector<mlxc::array>& inputs) {
    ValueMap vm;

    // Map function arguments → input arrays
    auto& entry = func.getBody().front();
    auto blockArgs = entry.getArguments();
    if (inputs.size() != blockArgs.size()) {
        return InterpResult::Error(
            "Input count mismatch: expected " + std::to_string(blockArgs.size()) +
            ", got " + std::to_string(inputs.size()));
    }
    for (size_t i = 0; i < blockArgs.size(); i++)
        vm.emplace(valKey(blockArgs[i]), inputs[i]);

    return interpretBlock(entry, module, std::move(vm));
}

static InterpResult interpretBlock(mlir::Block& entry,
                                    mlir::ModuleOp module,
                                    ValueMap vm) {
    std::vector<mlxc::array> outputs;
    bool traceOps = std::getenv("JAX_MLX_TRACE_OPS") != nullptr;

    for (auto& op : entry) {
        auto opName = op.getName().getStringRef();
        if (traceOps) std::cerr << "[mlx-op] " << opName.str() << "\n";

        // Helper: get the i-th operand array
        auto operand = [&](unsigned i) -> mlxc::array& {
            return vm.at(valKey(op.getOperand(i)));
        };

        // Helper: set the i-th result
        auto set = [&](unsigned i, mlxc::array arr) {
            vm.emplace(valKey(op.getResult(i)), std::move(arr));
        };

        // --- func.return: collect outputs ---
        if (opName == "func.return" || opName == "stablehlo.return") {
            for (auto v : op.getOperands()) {
                auto it = vm.find(valKey(v));
                if (it == vm.end())
                    return InterpResult::Error("Return value missing from value map");
                outputs.push_back(it->second);
            }
            break;
        }

        // --- stablehlo.constant ---
        if (opName == "stablehlo.constant") {
            set(0, makeConstant(mlir::cast<mlir::stablehlo::ConstantOp>(op)));
        }

        // --- Shape / type ops ---
        else if (opName == "stablehlo.reshape") {
            auto reshOp = mlir::cast<mlir::stablehlo::ReshapeOp>(op);
            set(0, mlxc::reshape(operand(0),
                               toMlxShape(mlir::cast<mlir::RankedTensorType>(reshOp.getType()))));
        } else if (opName == "stablehlo.broadcast_in_dim") {
            auto bdcOp = mlir::cast<mlir::stablehlo::BroadcastInDimOp>(op);
            set(0, broadcastInDim(operand(0), bdcOp.getBroadcastDimensions(),
                                   toMlxShape(mlir::cast<mlir::RankedTensorType>(bdcOp.getType()))));
        } else if (opName == "stablehlo.convert") {
            auto cvtOp = mlir::cast<mlir::stablehlo::ConvertOp>(op);
            auto dtype = MlirTypeToMlx(
                mlir::cast<mlir::RankedTensorType>(cvtOp.getType()).getElementType());
            set(0, mlxc::astype(operand(0), dtype));
        } else if (opName == "stablehlo.bitcast_convert" || opName.contains("bitcast_convert")) {
            auto outType = mlir::cast<mlir::RankedTensorType>(op.getResult(0).getType());
            auto outDtype = MlirTypeToMlx(outType.getElementType());
            auto outShape = toMlxShape(outType);

            auto in = operand(0);
            int64_t inBytes = static_cast<int64_t>(in.size()) * in.dtype().size();
            int64_t outElems = 1;
            for (int d : outShape) outElems *= d;
            int64_t outBytes = outElems * outDtype.size();
            if (inBytes != outBytes) {
                return InterpResult::Error(
                    "stablehlo.bitcast_convert: byte-size mismatch");
            }
            // Reshape to 1D first to ensure logical row-major byte order (handles
            // non-contiguous strides lazily), then view, then reshape to output shape.
            auto flat = mlxc::reshape(in, {static_cast<int>(in.size())});
            set(0, mlxc::reshape(mlxc::view(flat, outDtype), outShape));
        } else if (opName == "stablehlo.transpose") {
            auto transpOp = mlir::cast<mlir::stablehlo::TransposeOp>(op);
            std::vector<int> perm;
            for (int64_t d : transpOp.getPermutation())
                perm.push_back(static_cast<int>(d));
            set(0, permuteAxes(operand(0), perm));
        } else if (opName == "stablehlo.iota") {
            auto iotaOp = mlir::cast<mlir::stablehlo::IotaOp>(op);
            auto resultType = mlir::cast<mlir::RankedTensorType>(iotaOp.getType());
            int iotaDim = static_cast<int>(iotaOp.getIotaDimension());
            int64_t n = resultType.getShape()[iotaDim];
            auto dtype = MlirTypeToMlx(resultType.getElementType());
            auto range = mlxc::arange(0.0, static_cast<double>(n), 1.0, dtype);
            mlxc::Shape outShape = toMlxShape(resultType);
            mlxc::Shape rangeShape(outShape.size(), 1);
            rangeShape[iotaDim] = static_cast<int>(n);
            set(0, mlxc::broadcast_to(mlxc::reshape(range, rangeShape), outShape));
        } else if (opName == "stablehlo.copy") {
            set(0, operand(0));  // identity
        } else if (opName == "stablehlo.reverse") {
            auto revOp = mlir::cast<mlir::stablehlo::ReverseOp>(op);
            auto out = operand(0);
            for (int64_t d : revOp.getDimensions()) {
                int axis = static_cast<int>(d);
                int n = out.shape()[axis];
                auto idx = mlxc::arange(static_cast<double>(n - 1), -1.0, -1.0, mlxc::int32);
                out = mlxc::take(out, idx, axis);
            }
            set(0, out);
        }

        // --- Unary math ---
        else if (opName == "stablehlo.abs")               set(0, mlxc::abs(operand(0)));
        else if (opName == "stablehlo.negate")            set(0, mlxc::negative(operand(0)));
        else if (opName == "stablehlo.sign")              set(0, mlxc::sign(operand(0)));
        else if (opName == "stablehlo.not")               set(0, bitwiseNot(operand(0)));
        else if (opName == "stablehlo.exponential")       set(0, mlxc::exp(operand(0)));
        else if (opName == "stablehlo.exponential_minus_one") set(0, mlxc::expm1(operand(0)));
        else if (opName == "stablehlo.sqrt")              set(0, mlxc::sqrt(operand(0)));
        else if (opName == "stablehlo.rsqrt")             set(0, mlxc::rsqrt(operand(0)));
        else if (opName == "stablehlo.cbrt")              set(0, cbrt(operand(0)));
        else if (opName == "stablehlo.log")               set(0, mlxc::log(operand(0)));
        else if (opName == "stablehlo.log_plus_one")      set(0, mlxc::log1p(operand(0)));
        else if (opName == "stablehlo.logistic")          set(0, mlxc::sigmoid(operand(0)));
        else if (opName == "stablehlo.sine")              set(0, mlxc::sin(operand(0)));
        else if (opName == "stablehlo.cosine")            set(0, mlxc::cos(operand(0)));
        else if (opName == "stablehlo.tan")               set(0, mlxc::tan(operand(0)));
        else if (opName == "stablehlo.tanh")              set(0, mlxc::tanh(operand(0)));
        else if (opName == "stablehlo.floor")             set(0, mlxc::floor(operand(0)));
        else if (opName == "stablehlo.ceil")              set(0, mlxc::ceil(operand(0)));
        else if (opName == "stablehlo.round_nearest_afz") set(0, mlxc::round(operand(0)));
        else if (opName == "stablehlo.round_nearest_even") set(0, mlxc::round(operand(0)));
        else if (opName == "stablehlo.is_finite")         set(0, mlxc::isfinite(operand(0)));
        else if (opName == "stablehlo.real")              set(0, mlxc::real(operand(0)));
        else if (opName == "stablehlo.imag")              set(0, mlxc::imag(operand(0)));
        else if (opName == "stablehlo.popcnt")            set(0, popcnt(operand(0)));

        // --- CHLO unary ops ---
        else if (opName == "chlo.asin")    set(0, mlxc::arcsin(operand(0)));
        else if (opName == "chlo.acos")    set(0, mlxc::arccos(operand(0)));
        else if (opName == "chlo.atan")    set(0, mlxc::arctan(operand(0)));
        else if (opName == "chlo.asinh")   set(0, mlxc::arcsinh(operand(0)));
        else if (opName == "chlo.acosh")   set(0, mlxc::arccosh(operand(0)));
        else if (opName == "chlo.atanh")   set(0, mlxc::arctanh(operand(0)));
        else if (opName == "chlo.sinh")    set(0, mlxc::sinh(operand(0)));
        else if (opName == "chlo.cosh")    set(0, mlxc::cosh(operand(0)));
        else if (opName == "chlo.erf")     set(0, mlxc::erf(operand(0)));
        else if (opName == "chlo.erf_inv") set(0, mlxc::erfinv(operand(0)));
        else if (opName == "chlo.lgamma")  set(0, HandleLgamma(operand(0)));

        // --- Binary arithmetic ---
        else if (opName == "stablehlo.add")       set(0, mlxc::add(operand(0), operand(1)));
        else if (opName == "stablehlo.subtract")  set(0, mlxc::subtract(operand(0), operand(1)));
        else if (opName == "stablehlo.multiply")  set(0, mlxc::multiply(operand(0), operand(1)));
        else if (opName == "stablehlo.divide")    set(0, mlxc::divide(operand(0), operand(1)));
        else if (opName == "stablehlo.maximum")   set(0, mlxc::maximum(operand(0), operand(1)));
        else if (opName == "stablehlo.minimum")   set(0, mlxc::minimum(operand(0), operand(1)));
        else if (opName == "stablehlo.power")     set(0, mlxc::power(operand(0), operand(1)));
        else if (opName == "stablehlo.remainder") set(0, truncRemainder(operand(0), operand(1)));
        else if (opName == "stablehlo.and")       set(0, mlxc::bitwise_and(operand(0), operand(1)));
        else if (opName == "stablehlo.or")        set(0, mlxc::bitwise_or(operand(0), operand(1)));
        else if (opName == "stablehlo.xor")       set(0, mlxc::bitwise_xor(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_left")
            set(0, shiftLeftLikeCpu(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_right_arithmetic")
            set(0, shiftRightArithmeticLikeCpu(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_right_logical")
            set(0, shiftRightLogicalLikeCpu(operand(0), operand(1)));
        else if (opName == "stablehlo.atan2" || opName == "chlo.next_after" ||
                 opName == "stablehlo.next_after") {
            if (opName == "stablehlo.atan2")
                set(0, mlxc::arctan2(operand(0), operand(1)));
            else {
                // Compile-safe next_after via MLX bit manipulation — no eval() calls.
                // Flatten to 1D first so mlxc::view gets contiguous bytes in logical order.
                auto x = operand(0);
                auto y = operand(1);
                if (x.dtype() != y.dtype()) {
                    return InterpResult::Error("chlo.next_after expects same dtype operands");
                }
                if (x.dtype() != mlxc::float32 && x.dtype() != mlxc::float64) {
                    return InterpResult::Error("chlo.next_after only supports f32/f64");
                }
                bool isF32 = (x.dtype() == mlxc::float32);
                auto itype  = isF32 ? mlxc::int32  : mlxc::int64;
                auto shape  = x.shape();
                auto xFlat  = mlxc::reshape(x, {static_cast<int>(x.size())});
                auto yFlat  = mlxc::reshape(y, {static_cast<int>(y.size())});
                auto x_bits = mlxc::view(xFlat, itype);
                auto y_bits = mlxc::view(yFlat, itype);
                auto zero_f = mlxc::zeros_like(xFlat);
                auto zero_i = mlxc::zeros_like(x_bits);
                auto one_i  = mlxc::ones_like(x_bits);
                auto neg1_i = mlxc::full(x_bits.shape(), -1, itype);
                auto x_eq_y    = mlxc::equal(xFlat, yFlat);
                auto x_is_0    = mlxc::equal(xFlat, zero_f);
                auto x_pos     = mlxc::greater_equal(x_bits, zero_i);
                auto y_pos     = mlxc::greater_equal(y_bits, zero_i);
                auto towards_y = mlxc::greater(yFlat, xFlat);
                auto dir       = mlxc::where(towards_y, one_i, neg1_i);
                // Same sign: increment bits; opposite sign (crossing zero): decrement.
                auto same_sign = mlxc::equal(x_pos, y_pos);
                auto nz_bits   = mlxc::where(same_sign,
                                             mlxc::add(x_bits, dir),
                                             mlxc::subtract(x_bits, dir));
                // x == 0: return smallest subnormal with sign of y.
                // f32: +subnormal = 0x00000001, -subnormal = 0x80000001
                // f64: +subnormal = 0x0000000000000001, -subnormal = 0x8000000000000001
                auto min_pos = mlxc::full(x_bits.shape(), 1, itype);
                auto min_neg = isF32
                    ? mlxc::full(x_bits.shape(), static_cast<int>(-2147483647),  itype)
                    : mlxc::full(x_bits.shape(), (long long)-9223372036854775807LL, itype);
                auto min_sub  = mlxc::where(y_pos, min_pos, min_neg);
                auto res_bits = mlxc::where(x_is_0, min_sub, nz_bits);
                res_bits      = mlxc::where(x_eq_y, y_bits, res_bits);
                set(0, mlxc::reshape(mlxc::view(res_bits, x.dtype()), shape));
            }
        }

        // --- Compare ---
        else if (opName == "stablehlo.compare") {
            auto cmpOp = mlir::cast<mlir::stablehlo::CompareOp>(op);
            auto lhs = operand(0);
            auto rhs = operand(1);
            if (cmpOp.getCompareType() == mlir::stablehlo::ComparisonType::UNSIGNED) {
                lhs = mlxc::astype(lhs, unsignedDtypeFor(lhs.dtype()));
                rhs = mlxc::astype(rhs, unsignedDtypeFor(rhs.dtype()));
            }
            switch (cmpOp.getComparisonDirection()) {
                case mlir::stablehlo::ComparisonDirection::EQ:
                    set(0, mlxc::equal(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::NE:
                    set(0, mlxc::not_equal(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::LT:
                    set(0, mlxc::less(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::LE:
                    set(0, mlxc::less_equal(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::GT:
                    set(0, mlxc::greater(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::GE:
                    set(0, mlxc::greater_equal(lhs, rhs)); break;
            }
        }

        // --- Select ---
        else if (opName == "stablehlo.select") {
            set(0, mlxc::where(operand(0), operand(1), operand(2)));
        }
        // --- Clamp ---
        else if (opName == "stablehlo.clamp") {
            // Operands: [min, operand, max]
            set(0, mlxc::minimum(mlxc::maximum(operand(1), operand(0)), operand(2)));
        }

        // --- Concatenate ---
        else if (opName == "stablehlo.concatenate") {
            auto catOp = mlir::cast<mlir::stablehlo::ConcatenateOp>(op);
            int axis = static_cast<int>(catOp.getDimension());
            std::vector<mlxc::array> arrs;
            for (auto v : catOp.getInputs()) arrs.push_back(vm.at(valKey(v)));
            set(0, mlxc::concatenate(arrs, axis));
        }

        // --- Slice ---
        else if (opName == "stablehlo.slice") {
            auto slOp = mlir::cast<mlir::stablehlo::SliceOp>(op);
            mlxc::Shape starts, stops, strides;
            for (int64_t v : slOp.getStartIndices())
                starts.push_back(static_cast<int>(v));
            for (int64_t v : slOp.getLimitIndices())
                stops.push_back(static_cast<int>(v));
            for (int64_t v : slOp.getStrides())
                strides.push_back(static_cast<int>(v));
            set(0, mlxc::slice(operand(0), starts, stops, strides));
        }

        // --- Dynamic slice ---
        // Compile-safe: use arange + take instead of item<>() to avoid eval barriers.
        else if (opName == "stablehlo.dynamic_slice") {
            auto dsOp = mlir::cast<mlir::stablehlo::DynamicSliceOp>(op);
            auto result = operand(0);
            int szi = 0;
            for (int64_t sz : dsOp.getSliceSizes()) {
                auto startArr =
                    mlxc::astype(vm.at(valKey(op.getOperand(1 + szi))), mlxc::int32);
                auto idx = mlxc::add(
                    mlxc::arange(0.0, static_cast<double>(sz), 1.0, mlxc::int32),
                    startArr);
                result = mlxc::take(result, idx, szi);
                szi++;
            }
            set(0, result);
        }

        // --- Dynamic update slice ---
        // Compile-safe: build N-D linear indices via broadcasting, use put_along_axis
        // on the flattened base instead of materializing scalar start indices.
        else if (opName == "stablehlo.dynamic_update_slice") {
            auto base   = operand(0);
            auto update = operand(1);
            int rank    = base.ndim();

            // Row-major strides for base.
            std::vector<int> rowStrides(static_cast<size_t>(rank), 1);
            for (int d = rank - 2; d >= 0; --d)
                rowStrides[static_cast<size_t>(d)] =
                    rowStrides[static_cast<size_t>(d + 1)] * base.shape()[d + 1];

            // Accumulate N-D linear index array via broadcasting:
            // linear[i0,i1,...] = sum_d ((arange(ud) + start_d) * rowStride_d)
            mlxc::array linear = mlxc::zeros({1}, mlxc::int32);
            for (int d = 0; d < rank; ++d) {
                int ud = update.shape()[d];
                auto start =
                    mlxc::astype(vm.at(valKey(op.getOperand(2 + d))), mlxc::int32);
                auto arange_d =
                    mlxc::arange(0.0, static_cast<double>(ud), 1.0, mlxc::int32);
                auto idx_d = mlxc::add(arange_d, start);  // shape [ud]
                // Reshape for broadcast at axis d: [1,...,1,ud,1,...,1]
                mlxc::Shape bshape(static_cast<size_t>(rank), 1);
                bshape[static_cast<size_t>(d)] = ud;
                idx_d = mlxc::reshape(idx_d, bshape);
                linear = mlxc::add(
                    linear,
                    mlxc::multiply(idx_d,
                                   mlxc::full({1}, rowStrides[static_cast<size_t>(d)],
                                              mlxc::int32)));
            }
            // linear shape: [u0, u1, ..., uR-1] via broadcasting.
            int totalUpdate = static_cast<int>(update.size());
            auto flatBase   = mlxc::reshape(base,   {static_cast<int>(base.size())});
            auto flatLinear = mlxc::reshape(linear, {totalUpdate});
            auto flatUpdate = mlxc::reshape(update, {totalUpdate});
            auto outFlat = mlxc::put_along_axis(flatBase, flatLinear, flatUpdate, 0);
            set(0, mlxc::reshape(outFlat, base.shape()));
        }

        // --- Pad ---
        else if (opName == "stablehlo.pad") {
            auto padOp = mlir::cast<mlir::stablehlo::PadOp>(op);
            int rank = operand(0).ndim();
            std::vector<int> lo(rank), hi(rank), interior(rank);
            int i = 0;
            for (int64_t v : padOp.getEdgePaddingLow())
                lo[i++] = static_cast<int>(v);
            i = 0;
            for (int64_t v : padOp.getEdgePaddingHigh())
                hi[i++] = static_cast<int>(v);
            i = 0;
            for (int64_t v : padOp.getInteriorPadding())
                interior[i++] = static_cast<int>(v);

            bool hasInterior =
                std::any_of(interior.begin(), interior.end(), [](int v) { return v != 0; });
            if (hasInterior) {
                return InterpResult::Error(
                    "stablehlo.pad with interior padding not yet implemented");
            }

            // Build pad widths
            std::vector<std::pair<int, int>> padWidths(rank);
            for (int j = 0; j < rank; j++) padWidths[j] = {lo[j], hi[j]};
            set(0, mlxc::pad(operand(0), padWidths, operand(1)));
        }

        // --- Reduce ---
        else if (opName == "stablehlo.reduce") {
            auto redOp = mlir::cast<mlir::stablehlo::ReduceOp>(op);
            std::string redFn = reductionOpName(redOp.getBody());
            std::vector<int> axes;
            for (int64_t d : redOp.getDimensions())
                axes.push_back(static_cast<int>(d));

            auto redInputs = redOp.getInputs();
            // Arg{max,min} tuple reduction: reduce(value, iota) -> (value, index)
            if (redInputs.size() == 2 && op.getNumResults() == 2 && axes.size() == 1) {
                auto cmpDir = reductionCompareDirection(redOp.getBody());
                if (cmpDir.has_value()) {
                    auto data = vm.at(valKey(redInputs[0]));
                    int axis = axes[0];
                    auto idx = *cmpDir == mlir::stablehlo::ComparisonDirection::GT
                                   ? mlxc::argmax(data, axis, false)
                                   : mlxc::argmin(data, axis, false);
                    auto vals = mlxc::take_along_axis(
                        data, mlxc::expand_dims(idx, {axis}), axis);
                    vals = mlxc::squeeze(vals, {axis});
                    auto idxType =
                        mlir::cast<mlir::RankedTensorType>(op.getResult(1).getType());
                    auto idxDtype = MlirTypeToMlx(idxType.getElementType());
                    vm.emplace(valKey(op.getResult(0)), vals);
                    vm.emplace(valKey(op.getResult(1)), mlxc::astype(idx, idxDtype));
                    continue;
                }
            }

            for (size_t i = 0; i < redInputs.size(); i++) {
                auto& inp = vm.at(valKey(redInputs[i]));
                mlxc::array result = mlxc::array(0.0f);  // default; overwritten below
                if (redFn == "stablehlo.add")
                    result = mlxc::sum(inp, axes, false);
                else if (redFn == "stablehlo.maximum")
                    result = mlxc::max(inp, axes, false);
                else if (redFn == "stablehlo.minimum")
                    result = mlxc::min(inp, axes, false);
                else if (redFn == "stablehlo.multiply")
                    result = mlxc::prod(inp, axes, false);
                else if (redFn == "stablehlo.or")
                    result = mlxc::any(inp, axes, false);
                else if (redFn == "stablehlo.and")
                    result = mlxc::all(inp, axes, false);
                else
                    return InterpResult::Error(
                        "Unsupported reduction op: " + redFn);
                vm.emplace(valKey(redOp.getResult(i)), result);
            }
        }

        // --- Dot general ---
        else if (opName == "stablehlo.dot_general") {
            auto dotOp = mlir::cast<mlir::stablehlo::DotGeneralOp>(op);
            set(0, dotGeneral(operand(0), operand(1), dotOp.getDotDimensionNumbers()));
        }
        else if (opName == "stablehlo.convolution") {
            auto convOp = mlir::cast<mlir::stablehlo::ConvolutionOp>(op);
            try {
                set(0, HandleConvolution(operand(0), operand(1), convOp));
            } catch (const std::exception& ex) {
                return InterpResult::Error(std::string("stablehlo.convolution lowering failed: ") + ex.what());
            }
        }
        else if (opName == "stablehlo.gather") {
            auto gatherOp = mlir::cast<mlir::stablehlo::GatherOp>(op);
            try {
                set(0, HandleGather(operand(0), operand(1), gatherOp));
            } catch (const std::exception& ex) {
                return InterpResult::Error(std::string("stablehlo.gather lowering failed: ") + ex.what());
            }
        }
        else if (opName == "stablehlo.sort") {
            auto sortOp = mlir::cast<mlir::stablehlo::SortOp>(op);
            std::vector<mlxc::array> inputs;
            inputs.reserve(op.getNumOperands());
            for (unsigned i = 0; i < op.getNumOperands(); ++i) {
                inputs.push_back(operand(i));
            }
            try {
                auto sorted = HandleSort(std::move(inputs), sortOp);
                if (sorted.size() != op.getNumResults()) {
                    return InterpResult::Error("stablehlo.sort result arity mismatch");
                }
                for (unsigned i = 0; i < op.getNumResults(); ++i) {
                    set(i, sorted[i]);
                }
            } catch (const std::exception& ex) {
                return InterpResult::Error(std::string("stablehlo.sort lowering failed: ") + ex.what());
            }
        }
        else if (opName == "stablehlo.scatter") {
            auto scatterOp = mlir::cast<mlir::stablehlo::ScatterOp>(op);
            try {
                set(0, HandleScatter(operand(0), operand(1), operand(2), scatterOp));
            } catch (const std::exception& ex) {
                return InterpResult::Error(std::string("stablehlo.scatter lowering failed: ") + ex.what());
            }
        }
        else if (opName == "stablehlo.cholesky") {
            auto choleskyOp = mlir::cast<mlir::stablehlo::CholeskyOp>(op);
            try {
                set(0, HandleCholesky(operand(0), choleskyOp));
            } catch (const std::exception& ex) {
                return InterpResult::Error(std::string("stablehlo.cholesky lowering failed: ") + ex.what());
            }
        }
        else if (opName == "stablehlo.triangular_solve") {
            auto triSolveOp = mlir::cast<mlir::stablehlo::TriangularSolveOp>(op);
            try {
                set(0, HandleTriangularSolve(operand(0), operand(1), triSolveOp));
            } catch (const std::exception& ex) {
                return InterpResult::Error(std::string("stablehlo.triangular_solve lowering failed: ") + ex.what());
            }
        }
        else if (opName == "stablehlo.fft") {
            auto fftOp = mlir::cast<mlir::stablehlo::FftOp>(op);
            try {
                set(0, HandleFft(operand(0), fftOp));
            } catch (const std::exception& ex) {
                return InterpResult::Error(std::string("stablehlo.fft lowering failed: ") + ex.what());
            }
        }

        // --- Complex ---
        else if (opName == "stablehlo.complex") {
            // Combine real + imag via view
            auto& re = operand(0);
            auto& im = operand(1);
            // Stack [re, im] along new last dim → [..., 2], then view as complex64
            int lastAxis = static_cast<int>(re.ndim());
            auto stacked = mlxc::stack({re, im}, lastAxis);
            auto flat = mlxc::reshape(stacked, {(int)stacked.size()});
            auto cplx_flat = mlxc::view(flat, mlxc::complex64);
            mlxc::Shape outShape(re.shape().begin(), re.shape().end());
            set(0, mlxc::reshape(cplx_flat, outShape));
        }

        // --- func.call: call another function in the module ---
        else if (opName == "func.call") {
            auto callOp = mlir::cast<mlir::func::CallOp>(op);
            auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            if (!callee)
                return InterpResult::Error("func.call: unknown function " +
                                           callOp.getCallee().str());
            std::vector<mlxc::array> callInputs;
            for (auto operand : callOp.getOperands())
                callInputs.push_back(vm.at(valKey(operand)));
            auto callResult = interpretFunction(callee, module, callInputs);
            if (!callResult.ok()) return callResult;
            for (size_t i = 0; i < callResult.outputs.size(); i++)
                vm.emplace(valKey(callOp.getResult(i)),
                           std::move(callResult.outputs[i]));
        }

        // --- stablehlo.count_leading_zeros ---
        else if (opName == "stablehlo.count_leading_zeros") {
            // For each bit position from MSB down, check if that bit is set;
            // clz = position of the first set bit from the top.
            auto& a = operand(0);
            int nbits = a.dtype().size() * 8;
            auto u = mlxc::astype(a, mlxc::uint32);
            // Iterate LSB→MSB; last found bit (highest bit) wins → correct clz
            auto clz = mlxc::full(a.shape(), nbits, mlxc::uint32);
            for (int i = 0; i < nbits; i++) {
                auto bit = mlxc::bitwise_and(mlxc::right_shift(u, mlxc::array((unsigned)i, mlxc::uint32)),
                                           mlxc::array(1u, mlxc::uint32));
                auto found = mlxc::equal(bit, mlxc::array(1u, mlxc::uint32));
                auto pos = mlxc::full(a.shape(), nbits - 1 - i, mlxc::uint32);
                clz = mlxc::where(found, pos, clz);
            }
            set(0, mlxc::astype(clz, a.dtype()));
        }

        // --- While loop ---
        else if (opName == "stablehlo.while") {
            auto whileOp = mlir::cast<mlir::stablehlo::WhileOp>(op);
            std::vector<mlxc::array> carried;
            carried.reserve(op.getNumOperands());
            for (auto v : op.getOperands()) carried.push_back(vm.at(valKey(v)));

            int64_t iter = 0;
            constexpr int64_t kMaxWhileIters = 1000000;
            while (true) {
                if (iter++ > kMaxWhileIters) {
                    return InterpResult::Error(
                        "stablehlo.while exceeded max iterations (" +
                        std::to_string(kMaxWhileIters) + ")");
                }
                auto condResult = interpretRegion(whileOp.getCond(), module, carried, vm);
                if (!condResult.ok()) return condResult;
                if (condResult.outputs.empty()) {
                    return InterpResult::Error(
                        "stablehlo.while condition returned no predicate");
                }
                bool pred = ScalarToInt64(condResult.outputs[0]) != 0;
                if (!pred) break;

                auto bodyResult = interpretRegion(whileOp.getBody(), module, carried, vm);
                if (!bodyResult.ok()) return bodyResult;
                if (bodyResult.outputs.size() != carried.size()) {
                    return InterpResult::Error(
                        "stablehlo.while body output arity mismatch");
                }
                carried = std::move(bodyResult.outputs);
            }
            if (carried.size() != op.getNumResults()) {
                return InterpResult::Error("stablehlo.while result arity mismatch");
            }
            for (size_t i = 0; i < carried.size(); i++)
                vm.emplace(valKey(op.getResult(i)), std::move(carried[i]));
        }

        // --- If ---
        else if (opName == "stablehlo.if") {
            auto ifOp = mlir::cast<mlir::stablehlo::IfOp>(op);
            bool pred = ScalarToInt64(operand(0)) != 0;
            std::vector<mlxc::array> branchInputs;
            branchInputs.reserve(op.getNumOperands() - 1);
            for (size_t i = 1; i < op.getNumOperands(); i++)
                branchInputs.push_back(vm.at(valKey(op.getOperand(i))));

            auto branchResult = interpretRegion(
                pred ? ifOp.getTrueBranch() : ifOp.getFalseBranch(),
                module, branchInputs, vm);
            if (!branchResult.ok()) return branchResult;
            if (branchResult.outputs.size() != op.getNumResults()) {
                return InterpResult::Error("stablehlo.if result arity mismatch");
            }
            for (size_t i = 0; i < branchResult.outputs.size(); i++)
                vm.emplace(valKey(op.getResult(i)),
                           std::move(branchResult.outputs[i]));
        }

        // --- Case ---
        else if (opName == "stablehlo.case") {
            auto caseOp = mlir::cast<mlir::stablehlo::CaseOp>(op);
            int64_t idx = ScalarToInt64(operand(0));
            int64_t numBranches = static_cast<int64_t>(caseOp->getNumRegions());
            if (numBranches <= 0) {
                return InterpResult::Error("stablehlo.case requires at least one branch");
            }
            idx = std::max<int64_t>(0, std::min<int64_t>(idx, numBranches - 1));

            std::vector<mlxc::array> branchInputs;
            branchInputs.reserve(op.getNumOperands() - 1);
            for (size_t i = 1; i < op.getNumOperands(); i++)
                branchInputs.push_back(vm.at(valKey(op.getOperand(i))));

            auto& region = caseOp->getRegion(static_cast<unsigned>(idx));
            auto branchResult = interpretRegion(region, module, branchInputs, vm);
            if (!branchResult.ok()) return branchResult;
            if (branchResult.outputs.size() != op.getNumResults()) {
                return InterpResult::Error("stablehlo.case result arity mismatch");
            }
            for (size_t i = 0; i < branchResult.outputs.size(); i++)
                vm.emplace(valKey(op.getResult(i)),
                           std::move(branchResult.outputs[i]));
        }
        // --- Custom call ---
        else if (opName == "stablehlo.custom_call") {
            auto ccOp = mlir::cast<mlir::stablehlo::CustomCallOp>(op);
            std::string target = ccOp.getCallTargetName().str();
            if (target.find("Sharding") != std::string::npos) {
                if (op.getNumOperands() == op.getNumResults()) {
                    for (size_t i = 0; i < op.getNumResults(); i++)
                        vm.emplace(valKey(op.getResult(i)), operand(i));
                } else if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
                    set(0, operand(0));
                } else {
                    return InterpResult::Error(
                        "stablehlo.custom_call(Sharding): unsupported operand/result arity");
                }
                continue;
            }
            if (target == "mhlo.erf") {
                if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
                    set(0, mlxc::erf(operand(0)));
                    continue;
                }
                return InterpResult::Error(
                    "stablehlo.custom_call(mhlo.erf): unsupported operand/result arity");
            }
            if (target == "mhlo.asin" || target == "mhlo.acos" || target == "mhlo.atan") {
                if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
                    if (target == "mhlo.asin") set(0, mlxc::arcsin(operand(0)));
                    else if (target == "mhlo.acos") set(0, mlxc::arccos(operand(0)));
                    else set(0, mlxc::arctan(operand(0)));
                    continue;
                }
                return InterpResult::Error(
                    "stablehlo.custom_call(" + target + "): unsupported operand/result arity");
            }
            if (target == "mhlo.asinh" || target == "mhlo.acosh" || target == "mhlo.atanh") {
                if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
                    if (target == "mhlo.asinh") set(0, mlxc::arcsinh(operand(0)));
                    else if (target == "mhlo.acosh") set(0, mlxc::arccosh(operand(0)));
                    else set(0, mlxc::arctanh(operand(0)));
                    continue;
                }
                return InterpResult::Error(
                    "stablehlo.custom_call(" + target + "): unsupported operand/result arity");
            }
            if (target == "mhlo.sinh" || target == "mhlo.cosh" || target == "mhlo.tanh") {
                if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
                    if (target == "mhlo.sinh") set(0, mlxc::sinh(operand(0)));
                    else if (target == "mhlo.cosh") set(0, mlxc::cosh(operand(0)));
                    else set(0, mlxc::tanh(operand(0)));
                    continue;
                }
                return InterpResult::Error(
                    "stablehlo.custom_call(" + target + "): unsupported operand/result arity");
            }
            if (target == "mhlo.topk") {
                if (op.getNumOperands() == 1 && op.getNumResults() == 2) {
                    // MLX argsort/topk on strided views can produce incorrect ordering;
                    // normalize to contiguous storage before sorting.
                    auto x = mlxc::contiguous(operand(0));
                    int axis = x.ndim() - 1;
                    auto outValsType =
                        mlir::cast<mlir::RankedTensorType>(op.getResult(0).getType());
                    auto outIdxType =
                        mlir::cast<mlir::RankedTensorType>(op.getResult(1).getType());
                    int k = static_cast<int>(outValsType.getShape().back());
                    if (k < 0 || k > x.shape()[axis]) {
                        return InterpResult::Error(
                            "stablehlo.custom_call(mhlo.topk): invalid k");
                    }

                    auto sortedIdx = ReverseAlongAxis(mlxc::argsort(x, axis), axis);
                    mlxc::Shape starts(static_cast<size_t>(sortedIdx.ndim()), 0);
                    mlxc::Shape stops(sortedIdx.shape().begin(), sortedIdx.shape().end());
                    mlxc::Shape strides(static_cast<size_t>(sortedIdx.ndim()), 1);
                    stops[static_cast<size_t>(axis)] = k;
                    auto topIdx = mlxc::slice(sortedIdx, starts, stops, strides);
                    auto topVals = mlxc::take_along_axis(x, topIdx, axis);
                    auto idxDtype = MlirTypeToMlx(outIdxType.getElementType());
                    set(0, topVals);
                    set(1, mlxc::astype(topIdx, idxDtype));
                    continue;
                }
                return InterpResult::Error(
                    "stablehlo.custom_call(mhlo.topk): unsupported operand/result arity");
            }
            return InterpResult::Error(
                jax_mlx::UnsupportedOpsMessage(
                    {"stablehlo.custom_call(" + target + ")"}));
        }

        // --- Unknown op ---
        else {
            return InterpResult::Error(jax_mlx::UnsupportedOpsMessage({opName.str()}));
        }
    }

    InterpResult result;
    result.outputs = std::move(outputs);
    return result;
}

// ============================================================================
// MlxExecutable implementation
// ============================================================================

MlxExecutable::MlxExecutable(MlxClient* client, mps::ParsedModule module)
    : client_(client) {
    if (!module.ok()) {
        error_ = "Invalid parsed module";
        return;
    }

    name_ = module.entry_func.getName().str();

    auto func_type = module.entry_func.getFunctionType();
    num_outputs_ = static_cast<int>(func_type.getNumResults());
    if (num_outputs_ == 0) num_outputs_ = 1;

    context_ = std::move(module.context);
    module_ = std::move(module.module);
    entry_func_ = module.entry_func;

    valid_ = true;
}

MlxExecutable::~MlxExecutable() {}

// Returns true if the module should skip mlxc::compile() wrapping.
// Two categories of barriers:
//
// 1. Hard eval barriers — ops whose handlers call eval() internally and
//    cannot be compiled at all.
//
// 2. Large-kernel ops — ops that already dispatch to a single heavy Metal
//    kernel (matmul, conv), where mlxc::compile() adds per-call overhead
//    without fusion benefit and causes measurable regressions.
//
// Elementwise/reduce-heavy functions (softmax, layernorm forward, etc.) do
// benefit from compilation; they don't contain dot_general or convolution.
//
// while/if/case: compile() would unroll loops; also requires eval() on
// condition scalars. These must remain barriers.
//
// Walks all functions in the module so func.call targets are also covered.
static bool HasHardEvalBarrier(mlir::ModuleOp module) {
    static const std::unordered_set<std::string_view> kBarriers = {
        // Hard eval barriers
        "stablehlo.cholesky",
        "stablehlo.triangular_solve",
        "chlo.lgamma",
        "stablehlo.while",
        "stablehlo.if",
        "stablehlo.case",
        // Large-kernel ops: already one Metal dispatch; compile() adds overhead
        "stablehlo.dot_general",
        "stablehlo.convolution",
    };
    bool found = false;
    module.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (kBarriers.count(op->getName().getStringRef())) {
            found = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return found;
}

ExecutionResult MlxExecutable::Execute(const std::vector<MlxBuffer*>& inputs,
                                        MlxDevice* device) {
    if (!valid_) return ExecutionResult::Error("Executable is not valid: " + error_);

    // Lazy compile on first call: wrap interpretFunction in mlxc::compile() so
    // MLX can cache the Metal kernel graph and fuse operations across calls.
    if (!compiled_fn_.has_value()) {
        if (HasHardEvalBarrier(*module_)) {
            // Sentinel: empty std::function means "use interpreter".
            compiled_fn_ = CompiledFn{};
        } else {
            mlir::func::FuncOp func = entry_func_;
            mlir::ModuleOp mod = *module_;
            compiled_fn_ = mlxc::compile(
                CompiledFn([func, mod](const std::vector<mlxc::array>& in) {
                    auto r = interpretFunction(func, mod, in);
                    if (!r.ok()) throw std::runtime_error(r.error);
                    return r.outputs;
                }));
        }
    }

    const auto& fn = *compiled_fn_;
    if (fn) {
        // Compiled path.  Catch Metal kernel compilation failures (e.g. MLX 0.31.0
        // ternary_ops `nan` issue) on first eval and fall back permanently.
        try {
            std::vector<mlxc::array> arrays;
            arrays.reserve(inputs.size());
            for (auto* buf : inputs) arrays.push_back(buf->array());
            auto outputs = fn(arrays);
            mlxc::eval(outputs);
            ExecutionResult result;
            for (auto& arr : outputs) {
                int pjrt_dtype = MlxDtypeToPjrt(arr.dtype());
                std::vector<int64_t> dims;
                for (int d : arr.shape()) dims.push_back(static_cast<int64_t>(d));
                result.buffers.push_back(
                    std::make_unique<MlxBuffer>(device, std::move(arr), pjrt_dtype, dims));
            }
            return result;
        } catch (const std::exception& e) {
            fprintf(stderr, "[jax-mlx] mlxc::compile eval failed (%s); falling back to interpreter\n",
                    e.what());
            compiled_fn_ = CompiledFn{};  // sentinel: use interpreter for this executable
        }
    }

    // Interpreter fallback (module has eval barriers, or compile failed above).
    return runFunction(entry_func_, *module_, inputs, device);
}

}  // namespace jax_mlx
