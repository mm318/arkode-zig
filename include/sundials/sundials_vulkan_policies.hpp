/*
 * -----------------------------------------------------------------
 * Programmer(s): ChatGPT (Vulkan port of CUDA exec policies)
 * -----------------------------------------------------------------
 * This header defines the ExecPolicy classes that determine Vulkan
 * compute dispatch geometry when launching Kompute algorithms.
 * -----------------------------------------------------------------
 */

#ifndef _SUNDIALS_VULKANEXECPOLICIES_HPP
#define _SUNDIALS_VULKANEXECPOLICIES_HPP

#include <cstddef>
#include <cstdint>

namespace sundials {
namespace vulkan {

class ExecPolicy
{
public:
  explicit ExecPolicy(uint32_t workgroupSize = 256u)
    : workgroupSize_(workgroupSize)
  {}

  ExecPolicy(const ExecPolicy& other) : workgroupSize_(other.workgroupSize_) {}

  virtual ~ExecPolicy() = default;

  virtual uint32_t blockSize(std::size_t /*numWorkUnits*/ = 0,
                             std::size_t /*gridDim*/      = 0) const
  {
    return workgroupSize_;
  }

  virtual uint32_t gridSize(std::size_t numWorkUnits = 0,
                            std::size_t /*blockDim*/ = 0) const
  {
    if (numWorkUnits == 0) { return 1; }
    return static_cast<uint32_t>((numWorkUnits + workgroupSize_ - 1) /
                                 workgroupSize_);
  }

  virtual bool atomic() const { return false; }

  virtual ExecPolicy* clone() const { return new ExecPolicy(*this); }

protected:
  uint32_t workgroupSize_;
};

class AtomicReduceExecPolicy : public ExecPolicy
{
public:
  explicit AtomicReduceExecPolicy(uint32_t workgroupSize = 256u,
                                  uint32_t gridSize      = 0u)
    : ExecPolicy(workgroupSize), gridSize_(gridSize)
  {}

  AtomicReduceExecPolicy(const AtomicReduceExecPolicy& other)
    : ExecPolicy(other), gridSize_(other.gridSize_)
  {}

  uint32_t gridSize(std::size_t numWorkUnits = 0,
                    std::size_t /*blockDim*/ = 0) const override
  {
    if (gridSize_ != 0u) { return gridSize_; }
    if (numWorkUnits == 0) { return 1; }
    return static_cast<uint32_t>((numWorkUnits + (blockSize() * 2 - 1)) /
                                 (blockSize() * 2));
  }

  bool atomic() const override { return true; }

  ExecPolicy* clone() const override
  {
    return new AtomicReduceExecPolicy(*this);
  }

private:
  uint32_t gridSize_;
};

} // namespace vulkan
} // namespace sundials

using SUNVulkanExecPolicy       = sundials::vulkan::ExecPolicy;
using SUNVulkanAtomicExecPolicy = sundials::vulkan::AtomicReduceExecPolicy;

#endif
