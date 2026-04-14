/*
 * -----------------------------------------------------------------
 * Vulkan utility helpers for NVECTOR Kompute backend.
 * -----------------------------------------------------------------
 */

#ifndef _SUNDIALS_VULKAN_H
#define _SUNDIALS_VULKAN_H

#include <cassert>
#include <kompute/Kompute.hpp>
#include <memory>
#include <sundials/sundials_types.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */

// Shared Kompute manager used by all Vulkan helper components.
inline kp::Manager* SUNDIALS_VK_GetManager()
{
  // kp::Manager must be a singleton
  static kp::Manager mgr;
  return &mgr;
}

// Alias the shared manager without taking ownership.
inline std::shared_ptr<kp::Manager> SUNDIALS_VK_GetSharedManager()
{
  static std::shared_ptr<kp::Manager> mgr(SUNDIALS_VK_GetManager(),
                                          [](kp::Manager*) {});
  return mgr;
}

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
