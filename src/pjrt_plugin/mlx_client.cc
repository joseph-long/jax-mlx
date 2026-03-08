#include "pjrt_plugin/mlx_client.h"

#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/mlx_executable.h"
#include "pjrt_plugin/stablehlo_parser.h"

namespace jax_mlx {

std::unique_ptr<MlxClient> MlxClient::Create() {
    auto client = std::unique_ptr<MlxClient>(new MlxClient());
    if (!client->Initialize()) return nullptr;
    return client;
}

MlxClient::MlxClient() {}
MlxClient::~MlxClient() {}

bool MlxClient::Initialize() {
    platform_name_ = "mlx";
    platform_version_ = "mlx-0.1.0";
    devices_.push_back(std::make_unique<MlxDevice>(this, 0, "MLX GPU"));
    return true;
}

int MlxClient::device_count() const {
    return static_cast<int>(devices_.size());
}

int MlxClient::addressable_device_count() const {
    return device_count();
}

MlxDevice* MlxClient::device(int index) {
    return (index >= 0 && index < static_cast<int>(devices_.size()))
               ? devices_[index].get()
               : nullptr;
}

MlxDevice* MlxClient::addressable_device(int index) {
    return device(index);
}

MlxDevice* MlxClient::LookupDevice(int device_id) {
    for (auto& d : devices_) {
        if (d->id() == device_id) return d.get();
    }
    return nullptr;
}

std::unique_ptr<MlxBuffer> MlxClient::BufferFromHostBuffer(
    const void* data, int pjrt_dtype, const std::vector<int64_t>& dims,
    const std::vector<int64_t>& /*byte_strides*/, MlxDevice* device) {
    return MlxBuffer::FromHostBuffer(data, pjrt_dtype, dims, device);
}

std::unique_ptr<MlxExecutable> MlxClient::CompileStableHLO(mps::ParsedModule module,
                                                             MlxDevice* /*device*/) {
    return std::make_unique<MlxExecutable>(this, std::move(module));
}

}  // namespace jax_mlx
