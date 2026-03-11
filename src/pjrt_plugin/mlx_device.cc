#include "pjrt_plugin/mlx_device.h"

#include <utility>

namespace jax_mlx {

MlxDevice::MlxDevice(MlxClient* client, int id, std::string name)
    : client_(client), id_(id), device_kind_(std::move(name)) {
    debug_string_ = "MlxDevice(id=" + std::to_string(id) + ")";
}

MlxDevice::~MlxDevice() = default;

}  // namespace jax_mlx
