// MLX-backed PJRT executable: StableHLO → mlx::core op-by-op lowering

#include "pjrt_plugin/mlx_executable.h"

#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>
#include <sstream>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
    int outRank = static_cast<int>(outShape.size());
    mx::Shape interShape(outRank, 1);
    int inDimIdx = 0;
    for (int64_t d : bdcDims)
        interShape[static_cast<size_t>(d)] = input.shape()[inDimIdx++];
    return mx::broadcast_to(mx::reshape(input, interShape), outShape);
}

// stablehlo.dot_general → MLX matmul via permute+reshape
static mx::array dotGeneral(mx::array lhs, mx::array rhs,
                              mlir::stablehlo::DotDimensionNumbersAttr dimNums) {
    auto lhsBatchArr = dimNums.getLhsBatchingDimensions();
    auto rhsBatchArr = dimNums.getRhsBatchingDimensions();
    auto lhsContractArr = dimNums.getLhsContractingDimensions();
    auto rhsContractArr = dimNums.getRhsContractingDimensions();

    std::set<int> lhsBatchSet(lhsBatchArr.begin(), lhsBatchArr.end());
    std::set<int> lhsContractSet(lhsContractArr.begin(), lhsContractArr.end());
    std::set<int> rhsBatchSet(rhsBatchArr.begin(), rhsBatchArr.end());
    std::set<int> rhsContractSet(rhsContractArr.begin(), rhsContractArr.end());

    std::vector<int> lhsFreeDims, rhsFreeDims;
    for (int i = 0; i < lhs.ndim(); i++)
        if (!lhsBatchSet.count(i) && !lhsContractSet.count(i))
            lhsFreeDims.push_back(i);
    for (int i = 0; i < rhs.ndim(); i++)
        if (!rhsBatchSet.count(i) && !rhsContractSet.count(i))
            rhsFreeDims.push_back(i);

    // Permute lhs: [batch, free, contract]
    std::vector<int> lhsPerm, rhsPerm;
    for (auto d : lhsBatchArr) lhsPerm.push_back(d);
    for (auto d : lhsFreeDims) lhsPerm.push_back(d);
    for (auto d : lhsContractArr) lhsPerm.push_back(d);

    // Permute rhs: [batch, contract, free]
    for (auto d : rhsBatchArr) rhsPerm.push_back(d);
    for (auto d : rhsContractArr) rhsPerm.push_back(d);
    for (auto d : rhsFreeDims) rhsPerm.push_back(d);

    auto lhsP = mx::transpose(lhs, lhsPerm);
    auto rhsP = mx::transpose(rhs, rhsPerm);

    int batchN = static_cast<int>(lhsBatchArr.size());
    int freeL = static_cast<int>(lhsFreeDims.size());
    int freeR = static_cast<int>(rhsFreeDims.size());
    int contractN = static_cast<int>(lhsContractArr.size());

    int64_t batchSz = 1, M = 1, N = 1, K = 1;
    for (int i = 0; i < batchN; i++) batchSz *= lhsP.shape()[i];
    for (int i = batchN; i < batchN + freeL; i++) M *= lhsP.shape()[i];
    for (int i = batchN; i < batchN + contractN; i++) K *= lhsP.shape()[i];
    for (int i = batchN + contractN; i < batchN + contractN + freeR; i++)
        N *= rhsP.shape()[i];

    // Matmul on [batch, M, K] x [batch, K, N] = [batch, M, N]
    auto lhsR = mx::reshape(lhsP, {(int)batchSz, (int)M, (int)K});
    auto rhsR = mx::reshape(rhsP, {(int)batchSz, (int)K, (int)N});
    auto out = mx::matmul(lhsR, rhsR);  // [batch, M, N]

    // Reshape result to output shape
    mx::Shape outShape;
    for (int i = 0; i < batchN; i++) outShape.push_back(lhsP.shape()[i]);
    for (int i = batchN; i < batchN + freeL; i++) outShape.push_back(lhsP.shape()[i]);
    for (int i = batchN + contractN; i < rhsP.ndim(); i++) outShape.push_back(rhsP.shape()[i]);

    if (outShape.empty()) return mx::reshape(out, {});
    return mx::reshape(out, outShape);
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
static InterpResult interpretFunction(mlir::func::FuncOp func,
                                       mlir::ModuleOp module,
                                       const std::vector<mx::array>& inputs);

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
        if (opName == "func.return") {
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
        } else if (opName == "stablehlo.transpose") {
            auto transpOp = mlir::cast<mlir::stablehlo::TransposeOp>(op);
            std::vector<int> perm;
            for (int64_t d : transpOp.getPermutation())
                perm.push_back(static_cast<int>(d));
            set(0, mx::transpose(operand(0), perm));
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
        else if (opName == "stablehlo.remainder") set(0, mx::remainder(operand(0), operand(1)));
        else if (opName == "stablehlo.and")       set(0, mx::bitwise_and(operand(0), operand(1)));
        else if (opName == "stablehlo.or")        set(0, mx::bitwise_or(operand(0), operand(1)));
        else if (opName == "stablehlo.xor")       set(0, mx::bitwise_xor(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_left")
            set(0, mx::left_shift(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_right_arithmetic")
            set(0, mx::right_shift(operand(0), operand(1)));
        else if (opName == "stablehlo.shift_right_logical")
            set(0, mx::right_shift(operand(0), operand(1)));
        else if (opName == "stablehlo.atan2" || opName == "chlo.next_after") {
            if (opName == "stablehlo.atan2")
                set(0, mx::arctan2(operand(0), operand(1)));
            else  // chlo.next_after - not in MLX; fall through to error
                return InterpResult::Error("chlo.next_after not implemented");
        }

        // --- Compare ---
        else if (opName == "stablehlo.compare") {
            auto cmpOp = mlir::cast<mlir::stablehlo::CompareOp>(op);
            switch (cmpOp.getComparisonDirection()) {
                case mlir::stablehlo::ComparisonDirection::EQ:
                    set(0, mx::equal(operand(0), operand(1))); break;
                case mlir::stablehlo::ComparisonDirection::NE:
                    set(0, mx::not_equal(operand(0), operand(1))); break;
                case mlir::stablehlo::ComparisonDirection::LT:
                    set(0, mx::less(operand(0), operand(1))); break;
                case mlir::stablehlo::ComparisonDirection::LE:
                    set(0, mx::less_equal(operand(0), operand(1))); break;
                case mlir::stablehlo::ComparisonDirection::GT:
                    set(0, mx::greater(operand(0), operand(1))); break;
                case mlir::stablehlo::ComparisonDirection::GE:
                    set(0, mx::greater_equal(operand(0), operand(1))); break;
            }
        }

        // --- Select ---
        else if (opName == "stablehlo.select") {
            set(0, mx::where(operand(0), operand(1), operand(2)));
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
            return InterpResult::Error("stablehlo.while not yet implemented");
        }

        // --- If ---
        else if (opName == "stablehlo.if") {
            return InterpResult::Error("stablehlo.if not yet implemented");
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
