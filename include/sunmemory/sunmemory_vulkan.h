/* -----------------------------------------------------------------
 * Vulkan memory helper header (Kompute-backed).
 * -----------------------------------------------------------------*/

#ifndef _SUNDIALS_VULKANMEMORY_H
#define _SUNDIALS_VULKANMEMORY_H

#include <kompute/Kompute.hpp>
#include <sundials/sundials_memory.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

SUNDIALS_EXPORT
SUNMemoryHelper SUNMemoryHelper_Vulkan(SUNContext sunctx);

SUNDIALS_EXPORT
SUNErrCode SUNMemoryHelper_Alloc_Vulkan(SUNMemoryHelper helper,
                                        SUNMemory* memptr, size_t mem_size,
                                        SUNMemoryType mem_type, void* queue);

SUNDIALS_EXPORT
SUNErrCode SUNMemoryHelper_AllocStrided_Vulkan(SUNMemoryHelper helper,
                                               SUNMemory* memptr,
                                               size_t mem_size, size_t stride,
                                               SUNMemoryType mem_type,
                                               void* queue);

SUNDIALS_EXPORT SUNMemoryHelper SUNMemoryHelper_Clone_Vulkan(SUNMemoryHelper helper);

SUNDIALS_EXPORT
SUNErrCode SUNMemoryHelper_Dealloc_Vulkan(SUNMemoryHelper helper, SUNMemory mem,
                                          void* queue);

SUNDIALS_EXPORT
SUNErrCode SUNMemoryHelper_Copy_Vulkan(SUNMemoryHelper helper, SUNMemory dst,
                                       SUNMemory src, size_t memory_size,
                                       void* queue);

SUNDIALS_EXPORT
SUNErrCode SUNMemoryHelper_CopyAsync_Vulkan(SUNMemoryHelper helper,
                                            SUNMemory dst, SUNMemory src,
                                            size_t memory_size, void* queue);

SUNDIALS_EXPORT
SUNErrCode SUNMemoryHelper_Destroy_Vulkan(SUNMemoryHelper helper);

SUNDIALS_EXPORT
SUNErrCode SUNMemoryHelper_GetAllocStats_Vulkan(SUNMemoryHelper helper,
                                                SUNMemoryType mem_type,
                                                unsigned long* num_allocations,
                                                unsigned long* num_deallocations,
                                                size_t* bytes_allocated,
                                                size_t* bytes_high_watermark);

#ifdef __cplusplus
}
#endif

#endif
