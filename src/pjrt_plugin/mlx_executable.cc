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

namespace mx = mlx::core;

namespace jax_mlx {

// ValueMap: MLIR Value → MLX array (lazy, not yet evaluated)
// Uses void* (v.getImpl()) as key since mx::array is not default-constructible.
using ValueMap = std::unordered_map<void*, mx::array>;

static void* valKey(mlir::Value v) {
    return v.getImpl();
}

// ============================================================================
// Helpers
// ============================================================================

static mx::Shape toMlxShape(mlir::RankedTensorType type) {
    mx::Shape s;
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
static mx::array permuteAxes(mx::array input, const std::vector<int>& perm) {
    if (perm.empty()) return input;
    return mx::transpose(input, perm);
}

// Build an MLX array from a DenseElementsAttr constant
static mx::array makeConstant(mlir::stablehlo::ConstantOp op) {
    auto denseAttr = mlir::cast<mlir::DenseElementsAttr>(op.getValue());
    auto tensorType = mlir::cast<mlir::RankedTensorType>(denseAttr.getType());

    mx::Shape shape = toMlxShape(tensorType);
    auto elemType = tensorType.getElementType();
    auto dtype = MlirTypeToMlx(elemType);

    // Handle bool (i1): MLIR stores packed bits, convert to uint8
    if (elemType.isInteger(1)) {
        std::vector<uint8_t> data;
        for (bool v : denseAttr.getValues<bool>()) data.push_back(v ? 1 : 0);
        return mx::array(data.data(), shape, mx::bool_);
    }

    // All other types: copy raw bytes and hand ownership to MLX
    auto rawData = denseAttr.getRawData();
    size_t nbytes = rawData.size();
    void* buf = std::malloc(nbytes > 0 ? nbytes : 1);
    if (nbytes > 0) std::memcpy(buf, rawData.data(), nbytes);
    return mx::array(buf, shape, dtype, [](void* p) { std::free(p); });
}

// stablehlo.broadcast_in_dim: insert size-1 dims then broadcast_to
static mx::array broadcastInDim(mx::array input,
                                 llvm::ArrayRef<int64_t> bdcDims,
                                 mx::Shape outShape) {
    if (input.ndim() == 0 && bdcDims.empty()) {
        // MLX broadcast_to on reshaped scalars can produce sparse-looking outputs.
        // Force scalar broadcasting via arithmetic expansion.
        return mx::add(mx::zeros(outShape, input.dtype()), input);
    }
    int outRank = static_cast<int>(outShape.size());
    mx::Shape interShape(outRank, 1);
    int inDimIdx = 0;
    for (int64_t d : bdcDims)
        interShape[static_cast<size_t>(d)] = input.shape()[inDimIdx++];
    auto reshaped = mx::reshape(input, interShape);
    return mx::add(mx::zeros(outShape, input.dtype()), reshaped);
}

// stablehlo.dot_general lowered through MLX einsum.
static mx::array dotGeneral(mx::array lhs, mx::array rhs,
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
        return mx::matmul(lhs, rhs);
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

    return mx::einsum(lhsSub + "," + rhsSub + "->" + outSub, {lhs, rhs});
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

static mx::array HandleConvolution(mx::array lhs, mx::array rhs,
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
    if (batchGroups != 1) {
        throw std::invalid_argument("stablehlo.convolution with batch_group_count != 1");
    }

    auto outCanonical = mx::conv_general(lhsCanonical,
                                         rhsCanonical,
                                         strides,
                                         padLo,
                                         padHi,
                                         rhsDil,
                                         lhsDil,
                                         static_cast<int>(convOp.getFeatureGroupCount()),
                                         flip);

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

static mx::array HandleGather(mx::array operand, mx::array startIndices,
                              mlir::stablehlo::GatherOp gatherOp) {
    auto dimNums = gatherOp.getDimensionNumbers();
    auto startIndexMap = dimNums.getStartIndexMap();
    auto collapsedSliceDims = dimNums.getCollapsedSliceDims();
    auto sliceSizes = gatherOp.getSliceSizes();

    // Handle the common "take along one axis" gather pattern used by embeddings.
    if (startIndexMap.size() != 1 || collapsedSliceDims.size() != 1) {
        throw std::invalid_argument("only single-axis gather is implemented");
    }
    int axis = static_cast<int>(startIndexMap[0]);
    if (collapsedSliceDims[0] != startIndexMap[0]) {
        throw std::invalid_argument("collapsed axis must match gathered axis");
    }
    if (sliceSizes[static_cast<size_t>(axis)] != 1) {
        throw std::invalid_argument("slice size for gathered axis must be 1");
    }
    for (int i = 0; i < operand.ndim(); ++i) {
        if (i != axis && sliceSizes[static_cast<size_t>(i)] != operand.shape()[i]) {
            throw std::invalid_argument("non-gather axes must use full slice");
        }
    }

    auto indices = startIndices;
    int indexVectorDim = static_cast<int>(dimNums.getIndexVectorDim());
    if (indexVectorDim < indices.ndim()) {
        if (indices.shape()[indexVectorDim] != 1) {
            throw std::invalid_argument("index_vector_dim width must be 1");
        }
        indices = mx::squeeze(indices, {indexVectorDim});
    }

    return mx::take(operand, indices, axis);
}

static mx::array HandleScatter(mx::array operand, mx::array scatterIndices, mx::array updates,
                               mlir::stablehlo::ScatterOp scatterOp) {
    auto dimNums = scatterOp.getScatterDimensionNumbers();
    auto scatterDimsToOperandDims = dimNums.getScatterDimsToOperandDims();
    int indexVectorDim = static_cast<int>(dimNums.getIndexVectorDim());
    auto insertedWindowDims = dimNums.getInsertedWindowDims();

    if (scatterDimsToOperandDims.size() != 1 || insertedWindowDims.size() != 1) {
        throw std::invalid_argument("only single-axis scatter is implemented");
    }
    int axis = static_cast<int>(scatterDimsToOperandDims[0]);
    if (insertedWindowDims[0] != scatterDimsToOperandDims[0]) {
        throw std::invalid_argument("inserted_window_dims must match scatter axis");
    }

    auto idx = scatterIndices;
    if (indexVectorDim < idx.ndim()) {
        if (idx.shape()[indexVectorDim] != 1) {
            throw std::invalid_argument("index_vector_dim width must be 1");
        }
        idx = mx::squeeze(idx, {indexVectorDim});
    }

    std::vector<mx::array> indices = {idx};
    std::vector<int> axes = {axis};
    auto updatesForMlx = updates;
    int indexRank = idx.ndim();
    // MLX scatter expects updates rank == index_rank + operand_rank and
    // includes singleton dims for inserted_window_dims.
    if (updatesForMlx.ndim() == indexRank + operand.ndim() - 1) {
        updatesForMlx = mx::expand_dims(updatesForMlx, indexRank + axis);
    }

    std::string scatterFn;
    for (auto& op : scatterOp.getUpdateComputation().front()) {
        auto nm = op.getName().getStringRef();
        if (nm != "stablehlo.return" && nm != "func.return") {
            scatterFn = nm.str();
            break;
        }
    }
    if (scatterFn == "stablehlo.add") {
        return mx::scatter_add(operand, indices, updatesForMlx, axes);
    }
    return mx::scatter(operand, indices, updatesForMlx, axes);
}

static mx::array HandleCholesky(mx::array a, mlir::stablehlo::CholeskyOp choleskyOp) {
    bool lower = true;
    if (choleskyOp.getLowerAttr()) lower = choleskyOp.getLower();

    if (a.ndim() < 2) {
        throw std::invalid_argument("cholesky expects rank >= 2");
    }
    int n = a.shape()[a.ndim() - 1];
    int m = a.shape()[a.ndim() - 2];
    if (n != m) throw std::invalid_argument("cholesky expects square matrices");
    if (a.dtype() != mx::float32 && a.dtype() != mx::float64) {
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

    auto makeOutput = [&](const auto* srcPtr, auto nanValue) -> mx::array {
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
        return mx::array(buf, a.shape(), a.dtype(), [](void* p) { std::free(p); });
    };

    if (a.dtype() == mx::float32) {
        return makeOutput(a.data<float>(), std::numeric_limits<float>::quiet_NaN());
    }
    return makeOutput(a.data<double>(), std::numeric_limits<double>::quiet_NaN());
}

static mx::array HandleTriangularSolve(mx::array a, mx::array b,
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
    if (a.dtype() != mx::float32 && a.dtype() != mx::float64) {
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

    auto solveImpl = [&](const auto* aPtr, const auto* bPtr) -> mx::array {
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
        return mx::array(buf, b.shape(), b.dtype(), [](void* p) { std::free(p); });
    };

    if (a.dtype() == mx::float32) {
        return solveImpl(a.data<float>(), b.data<float>());
    }
    return solveImpl(a.data<double>(), b.data<double>());
}

static mx::array HandleFft(mx::array input, mlir::stablehlo::FftOp fftOp) {
    auto fftLength = fftOp.getFftLength();
    int nAxes = static_cast<int>(fftLength.size());
    if (nAxes <= 0 || nAxes > input.ndim()) {
        throw std::invalid_argument("fft: invalid fft_length rank");
    }
    std::vector<int> axes;
    axes.reserve(static_cast<size_t>(nAxes));
    int startAxis = input.ndim() - nAxes;
    for (int i = 0; i < nAxes; ++i) axes.push_back(startAxis + i);

    mx::Shape lengths;
    lengths.reserve(static_cast<size_t>(nAxes));
    for (int64_t n : fftLength) lengths.push_back(static_cast<int>(n));

    switch (fftOp.getFftType()) {
        case mlir::stablehlo::FftType::FFT:
            return mx::fft::fftn(input, lengths, axes);
        case mlir::stablehlo::FftType::IFFT:
            return mx::fft::ifftn(input, lengths, axes);
        case mlir::stablehlo::FftType::RFFT:
            return mx::fft::rfftn(input, lengths, axes);
        case mlir::stablehlo::FftType::IRFFT:
            return mx::fft::irfftn(input, lengths, axes);
        default:
            throw std::invalid_argument("fft: unsupported fft type");
    }
}

// Inspect reduction body to find the operation name
static std::string reductionOpName(mlir::Region& body) {
    for (auto& op : body.front()) {
        auto nm = op.getName().getStringRef();
        if (nm != "stablehlo.return" && nm != "func.return") return nm.str();
    }
    return "";
}

// Bitwise NOT: implemented as XOR with all-ones for integer types, logical_not for bool
static mx::array bitwiseNot(mx::array a) {
    if (a.dtype() == mx::bool_) return mx::logical_not(a);
    // Use -1 (all-ones in two's complement) as the mask
    auto mask = mx::full(a.shape(), -1, a.dtype());
    return mx::bitwise_xor(a, mask);
}

// cbrt(x) = sign(x) * |x|^(1/3)
static mx::array cbrt(mx::array a) {
    auto absA = mx::abs(a);
    auto cbrtAbs = mx::power(absA, mx::array(1.0f / 3.0f, mx::float32));
    auto sgn = mx::sign(a);
    return mx::multiply(sgn, cbrtAbs);
}

// Population count via successive right-shifts
static mx::array popcnt(mx::array a) {
    int nbits = a.dtype().size() * 8;
    auto count = mx::zeros(a.shape(), mx::uint32);
    auto one = mx::array(1u, mx::uint32);
    for (int i = 0; i < nbits; i++) {
        auto shifted = mx::right_shift(mx::astype(a, mx::uint32), mx::array((unsigned)i, mx::uint32));
        count = mx::add(count, mx::bitwise_and(shifted, one));
    }
    return mx::astype(count, a.dtype());
}

// stablehlo.remainder follows truncating remainder semantics.
static mx::array truncRemainder(mx::array x, mx::array y) {
    auto r = mx::remainder(x, y);
    auto zero = mx::zeros(r.shape(), r.dtype());
    auto xNeg = mx::less(x, zero);
    auto yNeg = mx::less(y, zero);
    auto signDiff = mx::not_equal(xNeg, yNeg);
    auto nonZero = mx::not_equal(r, zero);
    auto adjust = mx::logical_and(signDiff, nonZero);
    return mx::where(adjust, mx::subtract(r, y), r);
}

static mx::Dtype unsignedDtypeFor(mx::Dtype dt) {
    if (dt == mx::int32 || dt == mx::uint32) return mx::uint32;
    if (dt == mx::int64 || dt == mx::uint64) return mx::uint64;
    return dt;
}

static mx::array invalidShiftMask(mx::array shift, mx::array value) {
    int bits = value.dtype().size() * 8;
    auto zero = mx::zeros(shift.shape(), shift.dtype());
    auto bitsArr = mx::full(shift.shape(), bits, shift.dtype());
    return mx::logical_or(mx::less(shift, zero), mx::greater_equal(shift, bitsArr));
}

static mx::array shiftLeftLikeCpu(mx::array value, mx::array shift) {
    auto invalid = invalidShiftMask(shift, value);
    auto zeroShift = mx::zeros(shift.shape(), shift.dtype());
    auto safeShift = mx::where(invalid, zeroShift, shift);
    auto shifted = mx::left_shift(value, safeShift);
    auto zeroOut = mx::zeros(value.shape(), value.dtype());
    return mx::where(invalid, zeroOut, shifted);
}

static mx::array shiftRightLogicalLikeCpu(mx::array value, mx::array shift) {
    auto invalid = invalidShiftMask(shift, value);
    auto zeroShift = mx::zeros(shift.shape(), shift.dtype());
    auto safeShift = mx::where(invalid, zeroShift, shift);
    auto u = mx::astype(value, unsignedDtypeFor(value.dtype()));
    auto shifted = mx::right_shift(u, safeShift);
    auto shiftedCast = mx::astype(shifted, value.dtype());
    auto zeroOut = mx::zeros(value.shape(), value.dtype());
    return mx::where(invalid, zeroOut, shiftedCast);
}

static mx::array shiftRightArithmeticLikeCpu(mx::array value, mx::array shift) {
    auto invalid = invalidShiftMask(shift, value);
    auto zeroShift = mx::zeros(shift.shape(), shift.dtype());
    auto safeShift = mx::where(invalid, zeroShift, shift);
    auto shifted = mx::right_shift(value, safeShift);
    auto zero = mx::zeros(value.shape(), value.dtype());
    auto signFill = mx::where(mx::less(value, zero), mx::full(value.shape(), -1, value.dtype()), zero);
    return mx::where(invalid, signFill, shifted);
}

// ============================================================================
// Main interpreter
// ============================================================================

// Result of the inner interpreter: either an error string or a list of arrays
struct InterpResult {
    std::vector<mx::array> outputs;
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
                                       const std::vector<mx::array>& inputs);

static void BindBlockArguments(mlir::Block& block,
                                const std::vector<mx::array>& inputs,
                                ValueMap& vm) {
    if (inputs.size() != block.getNumArguments()) return;
    for (size_t i = 0; i < inputs.size(); i++)
        vm.emplace(valKey(block.getArgument(i)), inputs[i]);
}

static InterpResult interpretRegion(mlir::Region& region,
                                     mlir::ModuleOp module,
                                     const std::vector<mx::array>& inputs,
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

static int64_t ScalarToInt64(mx::array a) {
    a.eval();
    if (a.dtype() == mx::int32) return static_cast<int64_t>(a.item<int32_t>());
    if (a.dtype() == mx::uint32) return static_cast<int64_t>(a.item<uint32_t>());
    if (a.dtype() == mx::uint64) return static_cast<int64_t>(a.item<uint64_t>());
    if (a.dtype() == mx::bool_) return a.item<bool>() ? 1 : 0;
    return a.item<int64_t>();
}

static ExecutionResult runFunction(mlir::func::FuncOp func,
                                    mlir::ModuleOp module,
                                    const std::vector<MlxBuffer*>& inputs,
                                    MlxDevice* device) {
    std::vector<mx::array> arrays;
    arrays.reserve(inputs.size());
    for (auto* buf : inputs) arrays.push_back(buf->array());

    auto interp = interpretFunction(func, module, arrays);
    if (!interp.ok()) return ExecutionResult::Error(interp.error);

    mx::eval(interp.outputs);
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
                                       const std::vector<mx::array>& inputs) {
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
    std::vector<mx::array> outputs;

    for (auto& op : entry) {
        auto opName = op.getName().getStringRef();

        // Helper: get the i-th operand array
        auto operand = [&](unsigned i) -> mx::array& {
            return vm.at(valKey(op.getOperand(i)));
        };

        // Helper: set the i-th result
        auto set = [&](unsigned i, mx::array arr) {
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
            set(0, mx::reshape(operand(0),
                               toMlxShape(mlir::cast<mlir::RankedTensorType>(reshOp.getType()))));
        } else if (opName == "stablehlo.broadcast_in_dim") {
            auto bdcOp = mlir::cast<mlir::stablehlo::BroadcastInDimOp>(op);
            set(0, broadcastInDim(operand(0), bdcOp.getBroadcastDimensions(),
                                   toMlxShape(mlir::cast<mlir::RankedTensorType>(bdcOp.getType()))));
        } else if (opName == "stablehlo.convert") {
            auto cvtOp = mlir::cast<mlir::stablehlo::ConvertOp>(op);
            auto dtype = MlirTypeToMlx(
                mlir::cast<mlir::RankedTensorType>(cvtOp.getType()).getElementType());
            set(0, mx::astype(operand(0), dtype));
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
            // Generic bitcast fallback via logical-order byte packing.
            in.eval();
            void* buf = std::malloc(static_cast<size_t>(outBytes > 0 ? outBytes : 1));
            if (outBytes > 0) {
                std::vector<int64_t> dims(in.shape().begin(), in.shape().end());
                std::vector<int64_t> rawStrides(in.strides().begin(), in.strides().end());
                auto strides = normalizeStridesToElements(dims,
                                                          rawStrides,
                                                          static_cast<int64_t>(in.data_size()),
                                                          static_cast<size_t>(in.dtype().size()));
                size_t dstOffset = 0;
                copyStridedToLinearBytes(in.data<uint8_t>(),
                                         static_cast<uint8_t*>(buf),
                                         dims,
                                         strides,
                                         static_cast<size_t>(in.dtype().size()),
                                         0,
                                         0,
                                         dstOffset);
            }
            set(0, mx::array(buf, outShape, outDtype, [](void* p) { std::free(p); }));
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
            auto range = mx::arange(0.0, static_cast<double>(n), 1.0, dtype);
            mx::Shape outShape = toMlxShape(resultType);
            mx::Shape rangeShape(outShape.size(), 1);
            rangeShape[iotaDim] = static_cast<int>(n);
            set(0, mx::broadcast_to(mx::reshape(range, rangeShape), outShape));
        } else if (opName == "stablehlo.copy") {
            set(0, operand(0));  // identity
        } else if (opName == "stablehlo.reverse") {
            auto revOp = mlir::cast<mlir::stablehlo::ReverseOp>(op);
            auto out = operand(0);
            for (int64_t d : revOp.getDimensions()) {
                int axis = static_cast<int>(d);
                int n = out.shape()[axis];
                auto idx = mx::arange(static_cast<double>(n - 1), -1.0, -1.0, mx::int32);
                out = mx::take(out, idx, axis);
            }
            set(0, out);
        }

        // --- Unary math ---
        else if (opName == "stablehlo.abs")               set(0, mx::abs(operand(0)));
        else if (opName == "stablehlo.negate")            set(0, mx::negative(operand(0)));
        else if (opName == "stablehlo.sign")              set(0, mx::sign(operand(0)));
        else if (opName == "stablehlo.not")               set(0, bitwiseNot(operand(0)));
        else if (opName == "stablehlo.exponential")       set(0, mx::exp(operand(0)));
        else if (opName == "stablehlo.exponential_minus_one") set(0, mx::expm1(operand(0)));
        else if (opName == "stablehlo.sqrt")              set(0, mx::sqrt(operand(0)));
        else if (opName == "stablehlo.rsqrt")             set(0, mx::rsqrt(operand(0)));
        else if (opName == "stablehlo.cbrt")              set(0, cbrt(operand(0)));
        else if (opName == "stablehlo.log")               set(0, mx::log(operand(0)));
        else if (opName == "stablehlo.log_plus_one")      set(0, mx::log1p(operand(0)));
        else if (opName == "stablehlo.logistic")          set(0, mx::sigmoid(operand(0)));
        else if (opName == "stablehlo.sine")              set(0, mx::sin(operand(0)));
        else if (opName == "stablehlo.cosine")            set(0, mx::cos(operand(0)));
        else if (opName == "stablehlo.tan")               set(0, mx::tan(operand(0)));
        else if (opName == "stablehlo.tanh")              set(0, mx::tanh(operand(0)));
        else if (opName == "stablehlo.floor")             set(0, mx::floor(operand(0)));
        else if (opName == "stablehlo.ceil")              set(0, mx::ceil(operand(0)));
        else if (opName == "stablehlo.round_nearest_afz") set(0, mx::round(operand(0)));
        else if (opName == "stablehlo.round_nearest_even") set(0, mx::round(operand(0)));
        else if (opName == "stablehlo.is_finite")         set(0, mx::isfinite(operand(0)));
        else if (opName == "stablehlo.real")              set(0, mx::real(operand(0)));
        else if (opName == "stablehlo.imag")              set(0, mx::imag(operand(0)));
        else if (opName == "stablehlo.popcnt")            set(0, popcnt(operand(0)));

        // --- CHLO unary ops ---
        else if (opName == "chlo.asin")    set(0, mx::arcsin(operand(0)));
        else if (opName == "chlo.acos")    set(0, mx::arccos(operand(0)));
        else if (opName == "chlo.atan")    set(0, mx::arctan(operand(0)));
        else if (opName == "chlo.asinh")   set(0, mx::arcsinh(operand(0)));
        else if (opName == "chlo.acosh")   set(0, mx::arccosh(operand(0)));
        else if (opName == "chlo.atanh")   set(0, mx::arctanh(operand(0)));
        else if (opName == "chlo.sinh")    set(0, mx::sinh(operand(0)));
        else if (opName == "chlo.cosh")    set(0, mx::cosh(operand(0)));
        else if (opName == "chlo.erf")     set(0, mx::erf(operand(0)));
        else if (opName == "chlo.erf_inv") set(0, mx::erfinv(operand(0)));

        // --- Binary arithmetic ---
        else if (opName == "stablehlo.add")       set(0, mx::add(operand(0), operand(1)));
        else if (opName == "stablehlo.subtract")  set(0, mx::subtract(operand(0), operand(1)));
        else if (opName == "stablehlo.multiply")  set(0, mx::multiply(operand(0), operand(1)));
        else if (opName == "stablehlo.divide")    set(0, mx::divide(operand(0), operand(1)));
        else if (opName == "stablehlo.maximum")   set(0, mx::maximum(operand(0), operand(1)));
        else if (opName == "stablehlo.minimum")   set(0, mx::minimum(operand(0), operand(1)));
        else if (opName == "stablehlo.power")     set(0, mx::power(operand(0), operand(1)));
        else if (opName == "stablehlo.remainder") set(0, truncRemainder(operand(0), operand(1)));
        else if (opName == "stablehlo.and")       set(0, mx::bitwise_and(operand(0), operand(1)));
        else if (opName == "stablehlo.or")        set(0, mx::bitwise_or(operand(0), operand(1)));
        else if (opName == "stablehlo.xor")       set(0, mx::bitwise_xor(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_left")
            set(0, shiftLeftLikeCpu(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_right_arithmetic")
            set(0, shiftRightArithmeticLikeCpu(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_right_logical")
            set(0, shiftRightLogicalLikeCpu(operand(0), operand(1)));
        else if (opName == "stablehlo.atan2" || opName == "chlo.next_after" ||
                 opName == "stablehlo.next_after") {
            if (opName == "stablehlo.atan2")
                set(0, mx::arctan2(operand(0), operand(1)));
            else {
                auto x = operand(0);
                auto y = operand(1);
                if (x.dtype() != y.dtype() || x.size() != y.size()) {
                    return InterpResult::Error("chlo.next_after expects same dtype/size operands");
                }
                mx::Shape shape(x.shape().begin(), x.shape().end());
                if (x.dtype() == mx::float32) {
                    auto xFlat = mx::reshape(
                        mx::add(mx::zeros(x.shape(), x.dtype()), x), {static_cast<int>(x.size())});
                    auto yFlat = mx::reshape(
                        mx::add(mx::zeros(y.shape(), y.dtype()), y), {static_cast<int>(y.size())});
                    xFlat.eval();
                    yFlat.eval();
                    std::vector<float> out(static_cast<size_t>(x.size()));
                    auto* xp = xFlat.data<float>();
                    auto* yp = yFlat.data<float>();
                    for (size_t i = 0; i < out.size(); ++i) out[i] = std::nextafter(xp[i], yp[i]);
                    size_t nbytes = out.size() * sizeof(float);
                    void* buf = std::malloc(nbytes > 0 ? nbytes : 1);
                    if (nbytes > 0) std::memcpy(buf, out.data(), nbytes);
                    set(0, mx::array(buf, shape, mx::float32, [](void* p) { std::free(p); }));
                } else if (x.dtype() == mx::float64) {
                    auto xFlat = mx::reshape(
                        mx::add(mx::zeros(x.shape(), x.dtype()), x), {static_cast<int>(x.size())});
                    auto yFlat = mx::reshape(
                        mx::add(mx::zeros(y.shape(), y.dtype()), y), {static_cast<int>(y.size())});
                    xFlat.eval();
                    yFlat.eval();
                    std::vector<double> out(static_cast<size_t>(x.size()));
                    auto* xp = xFlat.data<double>();
                    auto* yp = yFlat.data<double>();
                    for (size_t i = 0; i < out.size(); ++i) out[i] = std::nextafter(xp[i], yp[i]);
                    size_t nbytes = out.size() * sizeof(double);
                    void* buf = std::malloc(nbytes > 0 ? nbytes : 1);
                    if (nbytes > 0) std::memcpy(buf, out.data(), nbytes);
                    set(0, mx::array(buf, shape, mx::float64, [](void* p) { std::free(p); }));
                } else {
                    return InterpResult::Error("chlo.next_after only supports f32/f64");
                }
            }
        }

        // --- Compare ---
        else if (opName == "stablehlo.compare") {
            auto cmpOp = mlir::cast<mlir::stablehlo::CompareOp>(op);
            auto lhs = operand(0);
            auto rhs = operand(1);
            if (cmpOp.getCompareType() == mlir::stablehlo::ComparisonType::UNSIGNED) {
                lhs = mx::astype(lhs, unsignedDtypeFor(lhs.dtype()));
                rhs = mx::astype(rhs, unsignedDtypeFor(rhs.dtype()));
            }
            switch (cmpOp.getComparisonDirection()) {
                case mlir::stablehlo::ComparisonDirection::EQ:
                    set(0, mx::equal(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::NE:
                    set(0, mx::not_equal(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::LT:
                    set(0, mx::less(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::LE:
                    set(0, mx::less_equal(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::GT:
                    set(0, mx::greater(lhs, rhs)); break;
                case mlir::stablehlo::ComparisonDirection::GE:
                    set(0, mx::greater_equal(lhs, rhs)); break;
            }
        }

        // --- Select ---
        else if (opName == "stablehlo.select") {
            set(0, mx::where(operand(0), operand(1), operand(2)));
        }
        // --- Clamp ---
        else if (opName == "stablehlo.clamp") {
            // Operands: [min, operand, max]
            set(0, mx::minimum(mx::maximum(operand(1), operand(0)), operand(2)));
        }

        // --- Concatenate ---
        else if (opName == "stablehlo.concatenate") {
            auto catOp = mlir::cast<mlir::stablehlo::ConcatenateOp>(op);
            int axis = static_cast<int>(catOp.getDimension());
            std::vector<mx::array> arrs;
            for (auto v : catOp.getInputs()) arrs.push_back(vm.at(valKey(v)));
            set(0, mx::concatenate(arrs, axis));
        }

        // --- Slice ---
        else if (opName == "stablehlo.slice") {
            auto slOp = mlir::cast<mlir::stablehlo::SliceOp>(op);
            mx::Shape starts, stops, strides;
            for (int64_t v : slOp.getStartIndices())
                starts.push_back(static_cast<int>(v));
            for (int64_t v : slOp.getLimitIndices())
                stops.push_back(static_cast<int>(v));
            for (int64_t v : slOp.getStrides())
                strides.push_back(static_cast<int>(v));
            set(0, mx::slice(operand(0), starts, stops, strides));
        }

        // --- Dynamic slice ---
        else if (opName == "stablehlo.dynamic_slice") {
            auto dsOp = mlir::cast<mlir::stablehlo::DynamicSliceOp>(op);
            int rank = operand(0).ndim();
            mx::Shape starts(rank), stops(rank), strides(rank, 1);
            int szi = 0;
            for (int64_t sz : dsOp.getSliceSizes()) {
                auto startArr = vm.at(valKey(op.getOperand(1 + szi)));
                startArr.eval();
                int s = (startArr.dtype() == mx::int32) ? startArr.item<int32_t>()
                                                        : static_cast<int>(startArr.item<int64_t>());
                starts[szi] = s;
                stops[szi] = s + static_cast<int>(sz);
                szi++;
            }
            set(0, mx::slice(operand(0), starts, stops, strides));
        }

        // --- Dynamic update slice ---
        else if (opName == "stablehlo.dynamic_update_slice") {
            // operands: [operand, update, start_indices...]
            // Implement by building per-axis scatter indices then calling mx::scatter
            auto& base = operand(0);
            auto& update = operand(1);
            int rank = base.ndim();

            std::vector<int> starts(rank);
            for (int i = 0; i < rank; i++) {
                auto si = vm.at(valKey(op.getOperand(2 + i)));
                si.eval();
                starts[i] = (si.dtype() == mx::int32) ? si.item<int32_t>()
                                                       : static_cast<int>(si.item<int64_t>());
            }

            // Build per-axis index arrays matching the update shape, then scatter
            mx::Shape updShape(update.shape().begin(), update.shape().end());
            std::vector<mx::array> axisIndices;
            axisIndices.reserve(rank);
            std::vector<int> axesList(rank);
            for (int d = 0; d < rank; d++) {
                int64_t upd_size = update.shape()[d];
                auto range = mx::arange(starts[d], starts[d] + (int)upd_size, 1, mx::int32);
                mx::Shape rshp(rank, 1);
                rshp[d] = static_cast<int>(upd_size);
                axisIndices.push_back(mx::broadcast_to(mx::reshape(range, rshp), updShape));
                axesList[d] = d;
            }
            set(0, mx::scatter(base, axisIndices, update, axesList));
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
            set(0, mx::pad(operand(0), padWidths, operand(1)));
        }

        // --- Reduce ---
        else if (opName == "stablehlo.reduce") {
            auto redOp = mlir::cast<mlir::stablehlo::ReduceOp>(op);
            std::string redFn = reductionOpName(redOp.getBody());
            std::vector<int> axes;
            for (int64_t d : redOp.getDimensions())
                axes.push_back(static_cast<int>(d));

            auto redInputs = redOp.getInputs();
            for (size_t i = 0; i < redInputs.size(); i++) {
                auto& inp = vm.at(valKey(redInputs[i]));
                mx::array result = mx::array(0.0f);  // default; overwritten below
                if (redFn == "stablehlo.add")
                    result = mx::sum(inp, axes, false);
                else if (redFn == "stablehlo.maximum")
                    result = mx::max(inp, axes, false);
                else if (redFn == "stablehlo.minimum")
                    result = mx::min(inp, axes, false);
                else if (redFn == "stablehlo.multiply")
                    result = mx::prod(inp, axes, false);
                else if (redFn == "stablehlo.or")
                    result = mx::any(inp, axes, false);
                else if (redFn == "stablehlo.and")
                    result = mx::all(inp, axes, false);
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
            auto stacked = mx::stack({re, im}, lastAxis);
            auto flat = mx::reshape(stacked, {(int)stacked.size()});
            auto cplx_flat = mx::view(flat, mx::complex64);
            mx::Shape outShape(re.shape().begin(), re.shape().end());
            set(0, mx::reshape(cplx_flat, outShape));
        }

        // --- func.call: call another function in the module ---
        else if (opName == "func.call") {
            auto callOp = mlir::cast<mlir::func::CallOp>(op);
            auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            if (!callee)
                return InterpResult::Error("func.call: unknown function " +
                                           callOp.getCallee().str());
            std::vector<mx::array> callInputs;
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
            auto u = mx::astype(a, mx::uint32);
            // Iterate LSB→MSB; last found bit (highest bit) wins → correct clz
            auto clz = mx::full(a.shape(), nbits, mx::uint32);
            for (int i = 0; i < nbits; i++) {
                auto bit = mx::bitwise_and(mx::right_shift(u, mx::array((unsigned)i, mx::uint32)),
                                           mx::array(1u, mx::uint32));
                auto found = mx::equal(bit, mx::array(1u, mx::uint32));
                auto pos = mx::full(a.shape(), nbits - 1 - i, mx::uint32);
                clz = mx::where(found, pos, clz);
            }
            set(0, mx::astype(clz, a.dtype()));
        }

        // --- While loop ---
        else if (opName == "stablehlo.while") {
            auto whileOp = mlir::cast<mlir::stablehlo::WhileOp>(op);
            std::vector<mx::array> carried;
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
            std::vector<mx::array> branchInputs;
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

            std::vector<mx::array> branchInputs;
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

ExecutionResult MlxExecutable::Execute(const std::vector<MlxBuffer*>& inputs,
                                        MlxDevice* device) {
    if (!valid_) return ExecutionResult::Error("Executable is not valid: " + error_);
    return runFunction(entry_func_, *module_, inputs, device);
}

}  // namespace jax_mlx
