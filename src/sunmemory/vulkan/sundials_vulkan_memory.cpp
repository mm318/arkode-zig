/* -----------------------------------------------------------------
 * Minimal Kompute-backed SUNMemory helper for Vulkan NVECTOR.
 * -----------------------------------------------------------------*/

#include <cstdlib>
#include <cstring>
#include <memory>

#include <kompute/Kompute.hpp>
#include <sundials/sundials_math.h>
#include <sunmemory/sunmemory_vulkan.h>

#include "sundials/priv/sundials_errors_impl.h"
#include "sundials/sundials_errors.h"
#include "sundials_vulkan.h"

struct SUNMemoryHelper_Content_Vulkan_
{
  unsigned long num_allocations_host{0};
  unsigned long num_deallocations_host{0};
  unsigned long num_allocations_device{0};
  unsigned long num_deallocations_device{0};
  size_t bytes_allocated_host{0};
  size_t bytes_high_watermark_host{0};
  size_t bytes_allocated_device{0};
  size_t bytes_high_watermark_device{0};
  std::shared_ptr<kp::Manager> manager;
};

typedef struct SUNMemoryHelper_Content_Vulkan_ SUNMemoryHelper_Content_Vulkan;
#define SUNHELPER_CONTENT(h) ((SUNMemoryHelper_Content_Vulkan*)h->content)

static std::shared_ptr<kp::Manager> GetHelperManager()
{
  static std::shared_ptr<kp::Manager> mgr = std::make_shared<kp::Manager>();
  return mgr;
}

extern "C" {

SUNMemoryHelper SUNMemoryHelper_Vulkan(SUNContext sunctx)
{
  SUNFunctionBegin(sunctx);

  SUNMemoryHelper helper = SUNMemoryHelper_NewEmpty(sunctx);
  SUNCheckLastErrNull();

  helper->ops->alloc         = SUNMemoryHelper_Alloc_Vulkan;
  helper->ops->allocstrided  = SUNMemoryHelper_AllocStrided_Vulkan;
  helper->ops->dealloc       = SUNMemoryHelper_Dealloc_Vulkan;
  helper->ops->copy          = SUNMemoryHelper_Copy_Vulkan;
  helper->ops->copyasync     = SUNMemoryHelper_CopyAsync_Vulkan;
  helper->ops->getallocstats = SUNMemoryHelper_GetAllocStats_Vulkan;
  helper->ops->clone         = SUNMemoryHelper_Clone_Vulkan;
  helper->ops->destroy       = SUNMemoryHelper_Destroy_Vulkan;

  helper->content =
    (SUNMemoryHelper_Content_Vulkan*)malloc(sizeof(SUNMemoryHelper_Content_Vulkan));
  SUNAssertNull(helper->content, SUN_ERR_MALLOC_FAIL);

  SUNHELPER_CONTENT(helper)->manager = GetHelperManager();
  return helper;
}

SUNMemoryHelper SUNMemoryHelper_Clone_Vulkan(SUNMemoryHelper helper)
{
  SUNFunctionBegin(helper->sunctx);
  return SUNMemoryHelper_Vulkan(helper->sunctx);
}

SUNErrCode SUNMemoryHelper_Alloc_Vulkan(SUNMemoryHelper helper, SUNMemory* memptr,
                                        size_t mem_size, SUNMemoryType mem_type,
                                        void* /*queue*/)
{
  SUNFunctionBegin(helper->sunctx);

  SUNMemory mem = SUNMemoryNewEmpty(helper->sunctx);
  SUNCheckLastErr();

  mem->bytes = mem_size;
  mem->type  = mem_type;
  mem->own   = SUNTRUE;
  mem->ptr   = NULL;

  if (mem_type == SUNMEMTYPE_HOST || mem_type == SUNMEMTYPE_PINNED ||
      mem_type == SUNMEMTYPE_UVM || mem_type == SUNMEMTYPE_DEVICE)
  {
    mem->ptr = malloc(mem_size);
    SUNAssert(mem->ptr, SUN_ERR_MALLOC_FAIL);
    if (mem_type == SUNMEMTYPE_DEVICE)
    {
      SUNHELPER_CONTENT(helper)->bytes_allocated_device += mem_size;
      SUNHELPER_CONTENT(helper)->bytes_high_watermark_device =
        SUNMAX(SUNHELPER_CONTENT(helper)->bytes_allocated_device,
               SUNHELPER_CONTENT(helper)->bytes_high_watermark_device);
      SUNHELPER_CONTENT(helper)->num_allocations_device++;
    }
    else
    {
      SUNHELPER_CONTENT(helper)->bytes_allocated_host += mem_size;
      SUNHELPER_CONTENT(helper)->bytes_high_watermark_host =
        SUNMAX(SUNHELPER_CONTENT(helper)->bytes_allocated_host,
               SUNHELPER_CONTENT(helper)->bytes_high_watermark_host);
      SUNHELPER_CONTENT(helper)->num_allocations_host++;
    }
  }
  else
  {
    free(mem);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  *memptr = mem;
  return SUN_SUCCESS;
}

SUNErrCode SUNMemoryHelper_AllocStrided_Vulkan(SUNMemoryHelper helper,
                                               SUNMemory* memptr, size_t mem_size,
                                               size_t stride, SUNMemoryType mem_type,
                                               void* queue)
{
  // Allocate contiguous storage; stride unused in this minimal helper.
  return SUNMemoryHelper_Alloc_Vulkan(helper, memptr, mem_size * stride, mem_type,
                                      queue);
}

SUNErrCode SUNMemoryHelper_Dealloc_Vulkan(SUNMemoryHelper helper, SUNMemory mem,
                                          void* /*queue*/)
{
  SUNFunctionBegin(helper->sunctx);
  if (mem == NULL) { return SUN_SUCCESS; }

  if (mem->own && mem->ptr)
  {
    free(mem->ptr);
    if (mem->type == SUNMEMTYPE_DEVICE)
    {
      SUNHELPER_CONTENT(helper)->num_deallocations_device++;
    }
    else
    {
      SUNHELPER_CONTENT(helper)->num_deallocations_host++;
    }
  }

  free(mem);
  return SUN_SUCCESS;
}

SUNErrCode SUNMemoryHelper_Copy_Vulkan(SUNMemoryHelper helper, SUNMemory dst,
                                       SUNMemory src, size_t memory_size,
                                       void* /*queue*/)
{
  SUNFunctionBegin(helper->sunctx);
  if (dst == NULL || src == NULL) { return SUN_ERR_ARG_OUTOFRANGE; }

  std::memcpy(dst->ptr, src->ptr, memory_size);
  return SUN_SUCCESS;
}

SUNErrCode SUNMemoryHelper_CopyAsync_Vulkan(SUNMemoryHelper helper, SUNMemory dst,
                                            SUNMemory src, size_t memory_size,
                                            void* queue)
{
  // No async path; fall back to sync copy.
  return SUNMemoryHelper_Copy_Vulkan(helper, dst, src, memory_size, queue);
}

SUNErrCode SUNMemoryHelper_Destroy_Vulkan(SUNMemoryHelper helper)
{
  if (helper == NULL) { return SUN_SUCCESS; }
  free(helper->content);
  free(helper);
  return SUN_SUCCESS;
}

SUNErrCode SUNMemoryHelper_GetAllocStats_Vulkan(SUNMemoryHelper helper,
                                                SUNMemoryType mem_type,
                                                unsigned long* num_allocations,
                                                unsigned long* num_deallocations,
                                                size_t* bytes_allocated,
                                                size_t* bytes_high_watermark)
{
  if (mem_type == SUNMEMTYPE_DEVICE)
  {
    *num_allocations   = SUNHELPER_CONTENT(helper)->num_allocations_device;
    *num_deallocations = SUNHELPER_CONTENT(helper)->num_deallocations_device;
    *bytes_allocated   = SUNHELPER_CONTENT(helper)->bytes_allocated_device;
    *bytes_high_watermark =
      SUNHELPER_CONTENT(helper)->bytes_high_watermark_device;
  }
  else
  {
    *num_allocations   = SUNHELPER_CONTENT(helper)->num_allocations_host;
    *num_deallocations = SUNHELPER_CONTENT(helper)->num_deallocations_host;
    *bytes_allocated   = SUNHELPER_CONTENT(helper)->bytes_allocated_host;
    *bytes_high_watermark =
      SUNHELPER_CONTENT(helper)->bytes_high_watermark_host;
  }
  return SUN_SUCCESS;
}

} // extern "C"

