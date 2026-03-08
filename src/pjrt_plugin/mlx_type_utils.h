#pragma once

#include <xla/pjrt/c/pjrt_c_api.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlx/dtype.h"

namespace jax_mlx {

// Convert PJRT buffer type to MLX Dtype
mlx::core::Dtype PjrtDtypeToMlx(int pjrt_dtype);

// Convert MLX Dtype to PJRT buffer type
int MlxDtypeToPjrt(mlx::core::Dtype dtype);

// Convert MLIR element type to PJRT buffer type
int MlirTypeToPjrtDtype(mlir::Type type);

// Convert MLIR element type to MLX Dtype
mlx::core::Dtype MlirTypeToMlx(mlir::Type type);

}  // namespace jax_mlx
