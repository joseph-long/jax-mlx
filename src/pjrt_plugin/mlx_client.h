#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pjrt_plugin/mlx_device.h"

namespace mps {
struct ParsedModule;
}

namespace jax_mlx {

class MlxBuffer;
class MlxExecutable;

// MLX-backed PJRT client
class MlxClient {
public:
    static std::unique_ptr<MlxClient> Create();
    ~MlxClient();

    const std::string& platform_name() const { return platform_name_; }
    const std::string& platform_version() const { return platform_version_; }
    int process_index() const { return 0; }

    int device_count() const;
    int addressable_device_count() const;
    MlxDevice* device(int index);
    MlxDevice* addressable_device(int index);
    MlxDevice* LookupDevice(int device_id);

    std::unique_ptr<MlxBuffer> BufferFromHostBuffer(
        const void* data,
        int pjrt_dtype,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& byte_strides,
        MlxDevice* device);

    std::unique_ptr<MlxExecutable> CompileStableHLO(mps::ParsedModule module,
                                                     MlxDevice* device);

private:
    MlxClient();
    bool Initialize();

    std::string platform_name_;
    std::string platform_version_;
    std::vector<std::unique_ptr<MlxDevice>> devices_;
};

}  // namespace jax_mlx
