#include "pjrt_plugin/mlx_device.h"

namespace jax_mlx {

MlxDevice::MlxDevice(MlxClient* client, int id, const std::string& name)
    : client_(client), id_(id), device_kind_(name) {
    debug_string_ = "MlxDevice(id=" + std::to_string(id) + ")";
}

MlxDevice::~MlxDevice() {}

}  // namespace jax_mlx
