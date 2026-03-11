// PJRT Buffer API implementation for Metal backend

#include <cstring>
#include <vector>

#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/pjrt_types.h"

// ============================================================================
// Buffer API
// ============================================================================

PJRT_Error* MPS_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
    delete args->buffer;
    return nullptr;
}

PJRT_Error* MPS_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
    args->type = args->buffer && args->buffer->buffer
                     ? static_cast<PJRT_Buffer_Type>(args->buffer->buffer->dtype())
                     : PJRT_Buffer_Type_F32;
    return nullptr;
}

PJRT_Error* MPS_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        const auto& dims = args->buffer->buffer->dimensions();
        args->dims = dims.data();
        args->num_dims = dims.size();
    } else {
        args->dims = nullptr;
        args->num_dims = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        const auto& dims = args->buffer->buffer->dimensions();
        args->unpadded_dims = dims.data();
        args->num_dims = dims.size();
    } else {
        args->unpadded_dims = nullptr;
        args->num_dims = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_DynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* args) {
    args->dynamic_dim_indices = nullptr;
    args->num_dynamic_dims = 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args) {
    args->layout.type = PJRT_Buffer_MemoryLayout_Type_Strides;
    return nullptr;
}

PJRT_Error* MPS_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
    args->on_device_size_in_bytes =
        args->buffer && args->buffer->buffer ? args->buffer->buffer->byte_size() : 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_Device(PJRT_Buffer_Device_Args* args) {
    JAXPLUGIN_LOG_DEBUG(" PJRT_Buffer_Device called, buffer=%p\n", (void*)args->buffer);
    if (args->buffer && args->buffer->client && !args->buffer->client->devices.empty()) {
        args->device = args->buffer->client->devices[0];
        JAXPLUGIN_LOG_DEBUG(" PJRT_Buffer_Device: returning device=%p from client=%p\n",
                            (void*)args->device, (void*)args->buffer->client);
    } else {
        args->device = nullptr;
        JAXPLUGIN_LOG_DEBUG(" PJRT_Buffer_Device: returning nullptr (no client or devices)\n");
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
    JAXPLUGIN_LOG_DEBUG(" PJRT_Buffer_Memory called\n");
    // Return the default memory for the buffer's device
    if (args->buffer && args->buffer->client && !args->buffer->client->memories.empty()) {
        args->memory = args->buffer->client->memories[0];
    } else {
        args->memory = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        args->buffer->buffer->Delete();
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
    args->is_deleted =
        args->buffer && args->buffer->buffer ? args->buffer->buffer->IsDeleted() : true;
    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
    fprintf(stderr, "[jax-mlx] MPS_Buffer_CopyToDevice called\n");
    // We only have one device (MLX GPU), so copy is a shallow copy of the MLX array.
    // MLX arrays are copy-on-write, so this is safe and cheap.
    if (!args->buffer || !args->buffer->buffer) {
        return MakeError("CopyToDevice: null source buffer");
    }
    auto* src = args->buffer->buffer.get();
    auto* dst = new PJRT_Buffer();
    dst->buffer = std::make_unique<jax_mlx::MlxBuffer>(src->device(), src->array(), src->dtype(),
                                                       src->dimensions());
    dst->client = args->buffer->client;
    args->dst_buffer = dst;
    fprintf(stderr, "[jax-mlx] MPS_Buffer_CopyToDevice done\n");
    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args* args) {
    if (!args || !args->buffer || !args->buffer->buffer) {
        return MakeError("CopyToMemory: null source buffer", PJRT_Error_Code_INVALID_ARGUMENT);
    }
    if (!args->dst_memory || !args->dst_memory->device) {
        return MakeError("CopyToMemory: null destination memory/device",
                         PJRT_Error_Code_INVALID_ARGUMENT);
    }

    // Unified-memory architecture: memory-to-memory copy on the same device is
    // equivalent to creating another buffer view/copy-on-write handle.
    auto* src = args->buffer->buffer.get();
    auto* dst = new PJRT_Buffer();
    dst->buffer = std::make_unique<jax_mlx::MlxBuffer>(src->device(), src->array(), src->dtype(),
                                                       src->dimensions());
    dst->client = args->buffer->client;
    args->dst_buffer = dst;
    return nullptr;
}

PJRT_Error* MPS_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
    if (args->src && args->src->buffer && args->dst) {
        args->src->buffer->ToHostBuffer(args->dst, nullptr);
    }

    auto* event = new PJRT_Event();
    event->ready = true;
    args->event = event;

    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyRawToHost(PJRT_Buffer_CopyRawToHost_Args* args) {
    if (!args || !args->buffer || !args->buffer->buffer || !args->dst) {
        return MakeError("CopyRawToHost: invalid arguments", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    size_t total = args->buffer->buffer->byte_size();
    if (args->offset < 0 || args->transfer_size < 0) {
        return MakeError("CopyRawToHost: negative offset or size",
                         PJRT_Error_Code_INVALID_ARGUMENT);
    }
    size_t offset = static_cast<size_t>(args->offset);
    size_t size = static_cast<size_t>(args->transfer_size);
    if (offset + size > total) {
        return MakeError("CopyRawToHost: requested range out of bounds",
                         PJRT_Error_Code_INVALID_ARGUMENT);
    }

    if (offset == 0 && size == total) {
        args->buffer->buffer->ToHostBuffer(args->dst, nullptr);
    } else {
        std::vector<uint8_t> tmp(total);
        args->buffer->buffer->ToHostBuffer(tmp.data(), nullptr);
        std::memcpy(args->dst, tmp.data() + offset, size);
    }

    auto* event = new PJRT_Event();
    event->ready = true;
    args->event = event;
    return nullptr;
}

PJRT_Error* MPS_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
    args->is_on_cpu = false;
    return nullptr;
}

PJRT_Error* MPS_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
    auto* event = new PJRT_Event();
    event->ready = true;
    args->event = event;
    return nullptr;
}

PJRT_Error* MPS_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args) {
    args->buffer_pointer = 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) {
    args->device_memory_ptr = nullptr;
    return nullptr;
}

// ============================================================================
// CopyToDeviceStream API
// ============================================================================

PJRT_Error* MPS_CopyToDeviceStream_Destroy(PJRT_CopyToDeviceStream_Destroy_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_AddChunk(PJRT_CopyToDeviceStream_AddChunk_Args* args) {
    return MakeError("CopyToDeviceStream not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_CopyToDeviceStream_TotalBytes(PJRT_CopyToDeviceStream_TotalBytes_Args* args) {
    args->total_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_GranuleSize(PJRT_CopyToDeviceStream_GranuleSize_Args* args) {
    args->granule_size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_CurrentBytes(PJRT_CopyToDeviceStream_CurrentBytes_Args* args) {
    args->current_bytes = 0;
    return nullptr;
}
