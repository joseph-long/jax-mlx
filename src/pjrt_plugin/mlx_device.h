#pragma once

#include <string>

namespace jax_mlx {

class MlxClient;

// Represents a single MLX device (GPU or CPU)
class MlxDevice {
public:
    MlxDevice(MlxClient* client, int id, const std::string& name);
    ~MlxDevice();

    int id() const { return id_; }
    int local_hardware_id() const { return id_; }
    const std::string& device_kind() const { return device_kind_; }
    const std::string& debug_string() const { return debug_string_; }
    const std::string& to_string() const { return debug_string_; }

    MlxClient* client() const { return client_; }
    bool IsAddressable() const { return true; }

private:
    MlxClient* client_;
    int id_;
    std::string device_kind_;
    std::string debug_string_;
};

}  // namespace jax_mlx
