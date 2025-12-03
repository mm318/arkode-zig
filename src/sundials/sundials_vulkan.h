/*
 * -----------------------------------------------------------------
 * Vulkan utility helpers for NVECTOR Kompute backend.
 * -----------------------------------------------------------------
 */

#ifndef _SUNDIALS_VULKAN_H
#define _SUNDIALS_VULKAN_H

#include <cassert>
#include <kompute/Kompute.hpp>
#include <sundials/sundials_types.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

#define SUNDIALS_VK_VERIFY(vkerr) SUNDIALS_VK_Assert(vkerr, __FILE__, __LINE__)

inline sunbooleantype SUNDIALS_VK_Assert(VkResult vkerr, const char* file,
                                         int line)
{
  if (vkerr != VK_SUCCESS)
  {
#ifdef SUNDIALS_DEBUG
    fprintf(stderr, "ERROR in Vulkan operation: %d %s:%d\n", vkerr, file, line);
#ifdef SUNDIALS_DEBUG_ASSERT
    assert(false);
#endif
#endif
    return SUNFALSE;
  }
  return SUNTRUE;
}

#ifdef __cplusplus /* wrapper to enable C++ usage */
}
#endif

#endif
