#include "pjrt_plugin/mlx_type_utils.h"

namespace jax_mlx {

mlx::core::Dtype PjrtDtypeToMlx(int pjrt_dtype) {
    switch (pjrt_dtype) {
        case PJRT_Buffer_Type_F32:
            return mlx::core::float32;
        case PJRT_Buffer_Type_F16:
            return mlx::core::float16;
        case PJRT_Buffer_Type_BF16:
            return mlx::core::bfloat16;
        case PJRT_Buffer_Type_F64:
            return mlx::core::float64;
        case PJRT_Buffer_Type_S8:
            return mlx::core::int8;
        case PJRT_Buffer_Type_S16:
            return mlx::core::int16;
        case PJRT_Buffer_Type_S32:
            return mlx::core::int32;
        case PJRT_Buffer_Type_S64:
            return mlx::core::int64;
        case PJRT_Buffer_Type_U8:
            return mlx::core::uint8;
        case PJRT_Buffer_Type_U16:
            return mlx::core::uint16;
        case PJRT_Buffer_Type_U32:
            return mlx::core::uint32;
        case PJRT_Buffer_Type_U64:
            return mlx::core::uint64;
        case PJRT_Buffer_Type_PRED:
            return mlx::core::bool_;
        case PJRT_Buffer_Type_C64:
        // MLX has no float64 complex — fall back to complex64
        case PJRT_Buffer_Type_C128:
            return mlx::core::complex64;
        default:
            return mlx::core::float32;
    }
}

int MlxDtypeToPjrt(mlx::core::Dtype dtype) {
    switch (dtype.val()) {
        case mlx::core::Dtype::Val::float32:
            return PJRT_Buffer_Type_F32;
        case mlx::core::Dtype::Val::float16:
            return PJRT_Buffer_Type_F16;
        case mlx::core::Dtype::Val::bfloat16:
            return PJRT_Buffer_Type_BF16;
        case mlx::core::Dtype::Val::float64:
            return PJRT_Buffer_Type_F64;
        case mlx::core::Dtype::Val::int8:
            return PJRT_Buffer_Type_S8;
        case mlx::core::Dtype::Val::int16:
            return PJRT_Buffer_Type_S16;
        case mlx::core::Dtype::Val::int32:
            return PJRT_Buffer_Type_S32;
        case mlx::core::Dtype::Val::int64:
            return PJRT_Buffer_Type_S64;
        case mlx::core::Dtype::Val::uint8:
            return PJRT_Buffer_Type_U8;
        case mlx::core::Dtype::Val::uint16:
            return PJRT_Buffer_Type_U16;
        case mlx::core::Dtype::Val::uint32:
            return PJRT_Buffer_Type_U32;
        case mlx::core::Dtype::Val::uint64:
            return PJRT_Buffer_Type_U64;
        case mlx::core::Dtype::Val::bool_:
            return PJRT_Buffer_Type_PRED;
        case mlx::core::Dtype::Val::complex64:
            return PJRT_Buffer_Type_C64;
        default:
            return PJRT_Buffer_Type_INVALID;
    }
}

int MlirTypeToPjrtDtype(mlir::Type type) {
    if (type.isF32())
        return PJRT_Buffer_Type_F32;
    if (type.isF16())
        return PJRT_Buffer_Type_F16;
    if (type.isBF16())
        return PJRT_Buffer_Type_BF16;
    if (type.isF64())
        return PJRT_Buffer_Type_F64;

    if (auto ct = mlir::dyn_cast<mlir::ComplexType>(type)) {
        mlir::Type inner = ct.getElementType();
        if (inner.isF32())
            return PJRT_Buffer_Type_C64;
        if (inner.isF64())
            return PJRT_Buffer_Type_C128;
        return -1;
    }

    if (auto it = mlir::dyn_cast<mlir::IntegerType>(type)) {
        unsigned w = it.getWidth();
        bool u = it.isUnsigned();
        if (w == 1)
            return PJRT_Buffer_Type_PRED;
        if (w == 8)
            return u ? PJRT_Buffer_Type_U8 : PJRT_Buffer_Type_S8;
        if (w == 16)
            return u ? PJRT_Buffer_Type_U16 : PJRT_Buffer_Type_S16;
        if (w == 32)
            return u ? PJRT_Buffer_Type_U32 : PJRT_Buffer_Type_S32;
        if (w == 64)
            return u ? PJRT_Buffer_Type_U64 : PJRT_Buffer_Type_S64;
    }

    return -1;
}

mlx::core::Dtype MlirTypeToMlx(mlir::Type type) {
    return PjrtDtypeToMlx(MlirTypeToPjrtDtype(type));
}

}  // namespace jax_mlx
