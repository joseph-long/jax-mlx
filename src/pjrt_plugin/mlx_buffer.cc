#include "pjrt_plugin/mlx_buffer.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "pjrt_plugin/mlx_type_utils.h"

namespace jax_mlx {

namespace {

std::vector<int64_t> NormalizeStridesToElements(const std::vector<int64_t>& dims,
                                                const std::vector<int64_t>& raw_strides,
                                                int64_t data_size_elems, size_t elem_size) {
    if (raw_strides.empty())
        return raw_strides;

    auto max_offset = [&](const std::vector<int64_t>& strides) {
        int64_t off = 0;
        for (long long dim : dims) {
            if (dim > 1)
                off += (dim - 1) * std::abs(dim);
        }
        return off;
    };

    // MLX exposes strides in element units. Some call paths may present byte
    // units; detect that and normalize.
    int64_t off = max_offset(raw_strides);
    if (off < data_size_elems || elem_size <= 1)
        return raw_strides;

    std::vector<int64_t> normalized = raw_strides;
    bool divisible = true;
    for (auto& s : normalized) {
        if (s % static_cast<int64_t>(elem_size) != 0) {
            divisible = false;
            break;
        }
        s /= static_cast<int64_t>(elem_size);
    }
    if (!divisible)
        return raw_strides;

    return max_offset(normalized) < data_size_elems ? normalized : raw_strides;
}

void CopyStridedToContiguous(const uint8_t* src, uint8_t* dst, const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides_elems, size_t elem_size,
                             size_t dim, int64_t src_index, size_t& dst_offset) {
    if (dim == dims.size()) {
        const auto* src_bytes = reinterpret_cast<const std::byte*>(src);
        std::ptrdiff_t byte_offset =
            static_cast<std::ptrdiff_t>(src_index) * static_cast<std::ptrdiff_t>(elem_size);
        std::memcpy(dst + dst_offset, src_bytes + byte_offset, elem_size);
        dst_offset += elem_size;
        return;
    }
    for (int64_t i = 0; i < dims[dim]; ++i) {
        CopyStridedToContiguous(src, dst, dims, strides_elems, elem_size, dim + 1,
                                src_index + i * strides_elems[dim], dst_offset);
    }
}

}  // namespace

MlxBuffer::MlxBuffer(MlxDevice* device, mlx::core::array array, int pjrt_dtype,
                     const std::vector<int64_t>& dims)
    : device_(device), array_(std::move(array)), pjrt_dtype_(pjrt_dtype), dims_(dims) {}

MlxBuffer::~MlxBuffer() = default;

std::unique_ptr<MlxBuffer> MlxBuffer::FromHostBuffer(const void* data, int pjrt_dtype,
                                                     const std::vector<int64_t>& dims,
                                                     MlxDevice* device) {
    mlx::core::Dtype mlx_dtype = PjrtDtypeToMlx(pjrt_dtype);

    // Convert dims to MLX Shape (int32_t)
    mlx::core::Shape mlx_shape;
    for (int64_t d : dims) {
        mlx_shape.push_back(static_cast<mlx::core::ShapeElem>(d));
    }

    // Compute total byte count
    int64_t nelems = 1;
    for (int64_t d : dims)
        nelems *= d;
    size_t nbytes = static_cast<size_t>(nelems) * mlx_dtype.size();

    // Copy data into a heap buffer that MLX will own via the deleter
    void* data_copy = std::malloc(nbytes > 0 ? nbytes : 1);
    if (nbytes > 0) {
        std::memcpy(data_copy, data, nbytes);
    }

    // Construct the MLX array, transferring ownership of data_copy
    auto arr = mlx::core::array(data_copy, mlx_shape, mlx_dtype, [](void* p) { std::free(p); });

    return std::make_unique<MlxBuffer>(device, std::move(arr), pjrt_dtype, dims);
}

int64_t MlxBuffer::element_count() const {
    if (dims_.empty())
        return 1;
    return std::accumulate(dims_.begin(), dims_.end(), int64_t{1},
                           [](int64_t a, int64_t b) { return a * b; });
}

size_t MlxBuffer::byte_size() const {
    return static_cast<size_t>(element_count()) * DtypeByteSize(pjrt_dtype_);
}

void MlxBuffer::ToHostBuffer(void* dst, std::function<void()> on_done) {
    array_.eval();
    size_t nbytes = byte_size();
    if (nbytes > 0) {
        if (array_.flags().row_contiguous) {
            std::memcpy(dst, array_.data<uint8_t>(), nbytes);
        } else {
            std::vector<int64_t> raw_strides;
            raw_strides.reserve(array_.strides().size());
            for (auto s : array_.strides())
                raw_strides.push_back(static_cast<int64_t>(s));

            auto strides = NormalizeStridesToElements(dims_, raw_strides,
                                                      static_cast<int64_t>(array_.data_size()),
                                                      DtypeByteSize(pjrt_dtype_));
            size_t dst_offset = 0;
            CopyStridedToContiguous(array_.data<uint8_t>(), static_cast<uint8_t*>(dst), dims_,
                                    strides, DtypeByteSize(pjrt_dtype_), 0, 0, dst_offset);
        }
    }
    if (on_done)
        on_done();
}

void MlxBuffer::Delete() {
    is_deleted_ = true;
}

size_t DtypeByteSize(int pjrt_dtype) {
    return PjrtDtypeToMlx(pjrt_dtype).size();
}

}  // namespace jax_mlx
