#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlx/array.h"
#include "pjrt_plugin/mlx_buffer.h"

namespace mps {
struct ParsedModule;
}

namespace jax_mlx {

class MlxClient;
class MlxDevice;

struct ExecutionResult {
    std::vector<std::unique_ptr<MlxBuffer>> buffers;
    std::string error;

    bool ok() const { return error.empty(); }

    static ExecutionResult Error(const std::string& msg) {
        ExecutionResult r;
        r.error = msg;
        return r;
    }
};

// Compiled executable backed by MLX
class MlxExecutable {
public:
    // Takes ownership of ParsedModule
    MlxExecutable(MlxClient* client, mps::ParsedModule module);
    ~MlxExecutable();

    bool IsValid() const { return valid_; }
    const std::string& error() const { return error_; }

    ExecutionResult Execute(const std::vector<MlxBuffer*>& inputs, MlxDevice* device);

    const std::string& name() const { return name_; }
    int num_outputs() const { return num_outputs_; }

private:
    MlxClient* client_;
    std::string name_;
    std::string error_;
    int num_outputs_ = 1;
    bool valid_ = false;

    std::unique_ptr<mlir::MLIRContext> context_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
    mlir::func::FuncOp entry_func_;

    // Lazily-initialized compiled function. Set on first Execute():
    //   empty optional  → not yet initialized
    //   has_value() && !value() → has eval barriers; use interpreter
    //   has_value() && value()  → compiled path
    using CompiledFn =
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;
    std::optional<CompiledFn> compiled_fn_;

};

}  // namespace jax_mlx
