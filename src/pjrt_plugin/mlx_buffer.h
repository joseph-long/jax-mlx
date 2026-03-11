#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "mlx/array.h"

namespace jax_mlx {

class MlxDevice;

// Represents a buffer backed by an mlx::core::array (unified memory)
class MlxBuffer {
public:
    MlxBuffer(MlxDevice* device, mlx::core::array array, int pjrt_dtype,
              const std::vector<int64_t>& dims);
    ~MlxBuffer();

    // Factory: create from a raw host pointer (copies the data)
    static std::unique_ptr<MlxBuffer> FromHostBuffer(const void* data, int pjrt_dtype,
                                                     const std::vector<int64_t>& dims,
                                                     MlxDevice* device);

    // Buffer metadata
    MlxDevice* device() const {
        return device_;
    }
    int dtype() const {
        return pjrt_dtype_;
    }
    const std::vector<int64_t>& dimensions() const {
        return dims_;
    }
    int64_t element_count() const;
    size_t byte_size() const;

    // Data access
    mlx::core::array& array() {
        return array_;
    }
    const mlx::core::array& array() const {
        return array_;
    }

    // Copy to host: evaluates the array then memcpy's to dst
    void ToHostBuffer(void* dst, std::function<void()> on_done);

    bool IsDeleted() const {
        return is_deleted_;
    }
    void Delete();

private:
    MlxDevice* device_;
    mlx::core::array array_;
    int pjrt_dtype_;
    std::vector<int64_t> dims_;
    bool is_deleted_ = false;
};

// Byte size of a PJRT dtype
size_t DtypeByteSize(int pjrt_dtype);

}  // namespace jax_mlx
