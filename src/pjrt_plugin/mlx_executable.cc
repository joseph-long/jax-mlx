#include "pjrt_plugin/mlx_executable.h"

#include "pjrt_plugin/stablehlo_parser.h"

namespace jax_mlx {

MlxExecutable::MlxExecutable(MlxClient* client, mps::ParsedModule module)
    : client_(client) {
    if (!module.ok()) {
        error_ = "Invalid parsed module";
        return;
    }

    name_ = module.entry_func.getName().str();

    // Count outputs from the entry function's return type
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
    return ExecutionResult::Error(
        "MLX execution not yet implemented. "
        "Op-by-op lowering from StableHLO to mlx::core is in progress.");
}

}  // namespace jax_mlx
