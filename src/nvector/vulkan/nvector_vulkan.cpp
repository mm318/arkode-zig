/*
 * -----------------------------------------------------------------
 * Vulkan NVECTOR implementation using Kompute and Slang (slangc).
 * NOTE: This backend mirrors the CUDA API but executes on Vulkan.
 * -----------------------------------------------------------------
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <new>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <kompute/Kompute.hpp>
#include <nvector/nvector_vulkan.h>
#include <sundials/priv/sundials_context_impl.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_errors.h>
#include <sunmemory/sunmemory_vulkan.h>

#include "sundials_vulkan.h"

#define ZERO   SUN_RCONST(0.0)
#define HALF   SUN_RCONST(0.5)
#define ONE    SUN_RCONST(1.0)
#define ONEPT5 SUN_RCONST(1.5)
#define TWO    SUN_RCONST(2.0)

using namespace sundials;
using namespace sundials::vulkan;

// Macros to access vector content
#define NVEC_VULKAN_CONTENT(x) ((N_VectorContent_Vulkan)(x->content))
#define NVEC_VULKAN_LENGTH(x)  (NVEC_VULKAN_CONTENT(x)->length)

struct PrivateVectorContent_Vulkan
{
  std::shared_ptr<kp::Manager> manager; // shared Kompute manager for queues/tensors

  std::variant<sunrealtype*, std::vector<sunrealtype>> host_data = nullptr;
  bool host_needs_update = false; // device is newer and must be copied to host

  std::shared_ptr<kp::Tensor> device_data; // device buffer that backs the vector
  bool device_needs_update = true; // host is newer and must be copied to device
};

#define NVEC_VULKAN_PRIVATE(x) \
  (static_cast<PrivateVectorContent_Vulkan*>(NVEC_VULKAN_CONTENT(x)->priv))

using ShaderFloat = float;

static constexpr bool kShaderMatchesSunreal =
  std::is_same_v<ShaderFloat, sunrealtype>;

static inline std::span<sunrealtype> HostData(N_Vector v)
{
  return {N_VGetHostArrayPointer_Vulkan(v),
          static_cast<size_t>(NVEC_VULKAN_LENGTH(v))};
}

// Shader runs in single precision; convert at the boundaries when sunrealtype differs.
template<typename T>
static std::vector<T> ToShaderBuffer(std::span<sunrealtype> src)
{
  if constexpr (std::is_same_v<T, sunrealtype>) { return src; }
  else
  {
    std::vector<T> dst(src.size());
    std::transform(src.begin(), src.end(), dst.begin(),
                   [](sunrealtype v) { return static_cast<T>(v); });
    return dst;
  }
}

template<typename T>
static void FromShaderBuffer(std::span<T> src, std::span<sunrealtype> dst)
{
  assert(dst.size() >= src.size());
  if constexpr (std::is_same_v<T, sunrealtype>)
  {
    std::copy(src.begin(), src.end(), dst.begin());
  }
  else
  {
    std::transform(src.begin(), src.end(), dst.begin(),
                   [](T v) { return static_cast<sunrealtype>(v); });
  }
}

// Ensure host_data has at least new_length capacity.
// If current storage is sufficient, just updates the nvector length.
// Otherwise: pointer -> convert to vector; vector -> resize.
static void EnsureHostDataLength(N_Vector v, sunindextype new_length)
{
  auto* priv = NVEC_VULKAN_PRIVATE(v);

  if (auto* vec = std::get_if<std::vector<sunrealtype>>(&priv->host_data))
  {
    if (vec->size() >= static_cast<size_t>(new_length))
    {
      // Sufficient storage - just update length
      NVEC_VULKAN_LENGTH(v) = new_length;
    }
    else
    {
      // Need to resize the vector
      vec->resize(new_length);
      NVEC_VULKAN_LENGTH(v) = new_length;
    }
  }
  else if (auto* ptr = std::get_if<sunrealtype*>(&priv->host_data))
  {
    // It's a pointer - we don't know its allocated size
    // If we need more than current length, convert to vector
    sunindextype old_length = NVEC_VULKAN_LENGTH(v);
    if (new_length <= old_length)
    {
      // Assume pointer has at least old_length capacity
      NVEC_VULKAN_LENGTH(v) = new_length;
    }
    else
    {
      // Need more storage - convert to vector
      std::vector<sunrealtype> new_vec(new_length);
      if (*ptr && old_length > 0)
      {
        std::copy(*ptr, *ptr + old_length, new_vec.begin());
      }
      priv->host_data       = std::move(new_vec);
      NVEC_VULKAN_LENGTH(v) = new_length;
    }
  }
}

static void EnsureTensor(N_Vector v)
{
  auto* priv = NVEC_VULKAN_PRIVATE(v);
  if (!priv->device_data)
  {
    if constexpr (kShaderMatchesSunreal)
    {
      priv->device_data =
        priv->manager->tensor(N_VGetHostArrayPointer_Vulkan(v),
                              static_cast<uint32_t>(NVEC_VULKAN_LENGTH(v)),
                              sizeof(ShaderFloat),
                              kp::Memory::dataType<ShaderFloat>(),
                              kp::Memory::MemoryTypes::eDevice);
    }
    else
    {
      std::vector<ShaderFloat> shader_init =
        ToShaderBuffer<ShaderFloat>(HostData(v));
      priv->device_data =
        priv->manager->tensor(shader_init.empty() ? nullptr : shader_init.data(),
                              static_cast<uint32_t>(NVEC_VULKAN_LENGTH(v)),
                              sizeof(ShaderFloat),
                              kp::Memory::dataType<ShaderFloat>(),
                              kp::Memory::MemoryTypes::eDevice);
    }
  }
}

static void MarkDeviceNeedsUpdate(N_Vector v)
{
  NVEC_VULKAN_PRIVATE(v)->device_needs_update = true;
  NVEC_VULKAN_PRIVATE(v)->host_needs_update   = false;
}

static void MarkHostNeedsUpdate(N_Vector v)
{
  NVEC_VULKAN_PRIVATE(v)->host_needs_update   = true;
  NVEC_VULKAN_PRIVATE(v)->device_needs_update = false;
}

void N_VCopyToDevice_Vulkan(N_Vector v)
{
  auto* priv = NVEC_VULKAN_PRIVATE(v);
  if (!priv->device_needs_update) { return; }
  EnsureTensor(v);
  if constexpr (kShaderMatchesSunreal)
  {
    std::span<sunrealtype> to_shader = HostData(v);
    priv->device_data->setData(to_shader.data(), to_shader.size());
  }
  else
  {
    std::vector<ShaderFloat> to_shader = ToShaderBuffer<ShaderFloat>(HostData(v));
    priv->device_data->setData(to_shader);
  }
  auto seq = priv->manager->sequence();
  seq->record<kp::OpSyncDevice>(
    {std::static_pointer_cast<kp::Memory>(priv->device_data)});
  seq->eval();
  priv->device_needs_update = false;
  priv->host_needs_update   = false;
}

void N_VCopyFromDevice_Vulkan(N_Vector v)
{
  auto* priv = NVEC_VULKAN_PRIVATE(v);
  if (!priv->host_needs_update) { return; }
  EnsureTensor(v);
  auto seq = priv->manager->sequence();
  seq->record<kp::OpSyncLocal>(
    {std::static_pointer_cast<kp::Memory>(priv->device_data)});
  seq->eval();
  ShaderFloat* from_shader = priv->device_data->data<ShaderFloat>();
  FromShaderBuffer<ShaderFloat>({from_shader, priv->device_data->size()},
                                HostData(v));
  priv->device_needs_update = false;
  priv->host_needs_update   = false;
}

// ---------------------------------------------------------------------------
// Slang compilation helpers
// ---------------------------------------------------------------------------

// Element-wise shader
static const std::string ElementwiseShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
// Elementwise operations controlled via push constants.
struct Params {{
    uint op;
    float a;
    float b;
    uint n;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{}> X;
[[vk::binding(1,0)]] RWStructuredBuffer<{}> Y;
[[vk::binding(2,0)]] RWStructuredBuffer<{}> Z;

[numthreads(LOCAL_SIZE_X, LOCAL_SIZE_Y, LOCAL_SIZE_Z)]
void main(uint3 dtid : SV_DispatchThreadID)
{{
    uint i = dtid.x;
    if (i >= params.n) return;

    {} a = ({})(params.a);
    {} b = ({})(params.b);

    switch (params.op)
    {{
    case 0: Z[i] = a * X[i] + b * Y[i]; break; // linear sum
    case 1: Z[i] = a; break; // const
    case 2: Z[i] = X[i] * Y[i]; break; // prod
    case 3: Z[i] = X[i] / Y[i]; break; // div
    case 4: Z[i] = a * X[i]; break; // scale
    case 5: Z[i] = abs(X[i]); break; // abs
    case 6: Z[i] = ({})(1.0) / X[i]; break; // inv
    case 7: Z[i] = X[i] + a; break; // add const
    case 8: Z[i] = (abs(X[i]) >= a) 
                    ? ({})(1.0)
                    : ({})(0.0);
                    break; // compare
    default: Z[i] = X[i]; break;
    }}
}}
                  )",
                                real, real, real, real, real, real, real, real,
                                real, real);

  return src;
}

// Dot product shader
static const std::string DotProdShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  // Parallel reduction shader for dot product.
  // Each workgroup computes a partial sum using shared memory reduction.
  // Each thread processes two elements, then we reduce within the workgroup.
  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint numGroups;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> X;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> Y;
[[vk::binding(2,0)]] RWStructuredBuffer<{0}> partialSums;

groupshared {0} sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{{
    uint tid = gtid.x;
    uint i = gid.x * (LOCAL_SIZE_X * 2) + tid;
    uint gridSize = LOCAL_SIZE_X * 2 * params.numGroups;

    // Grid-stride loop to handle arbitrarily large inputs
    {0} sum = ({0})0.0;
    while (i < params.n) {{
        sum += X[i] * Y[i];
        if (i + LOCAL_SIZE_X < params.n) {{
            sum += X[i + LOCAL_SIZE_X] * Y[i + LOCAL_SIZE_X];
        }}
        i += gridSize;
    }}
    sdata[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    // Reduction in shared memory
    if (LOCAL_SIZE_X >= 512) {{ if (tid < 256) {{ sdata[tid] += sdata[tid + 256]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 256) {{ if (tid < 128) {{ sdata[tid] += sdata[tid + 128]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 128) {{ if (tid < 64) {{ sdata[tid] += sdata[tid + 64]; }} GroupMemoryBarrierWithGroupSync(); }}

    // Warp-level reduction (Vulkan/SPIR-V requires explicit sync for correctness)
    if (tid < 32) {{
        if (LOCAL_SIZE_X >= 64) {{ sdata[tid] += sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 32) {{ sdata[tid] += sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 16) {{ sdata[tid] += sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 8) {{ sdata[tid] += sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 4) {{ sdata[tid] += sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 2) {{ sdata[tid] += sdata[tid + 1]; }}
    }}

    // Thread 0 writes this workgroup's partial sum
    if (tid == 0) {{
        partialSums[gid.x] = sdata[0];
    }}
}}
)",
                                real);

  return src;
}

// MaxNorm reduction shader (computes max(abs(x)))
static const std::string MaxNormShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint numGroups;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> X;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> partialMax;

groupshared {0} sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{{
    uint tid = gtid.x;
    uint i = gid.x * (LOCAL_SIZE_X * 2) + tid;
    uint gridSize = LOCAL_SIZE_X * 2 * params.numGroups;

    // Grid-stride loop to find max absolute value
    {0} maxVal = ({0})0.0;
    while (i < params.n) {{
        {0} val = abs(X[i]);
        if (val > maxVal) maxVal = val;
        if (i + LOCAL_SIZE_X < params.n) {{
            val = abs(X[i + LOCAL_SIZE_X]);
            if (val > maxVal) maxVal = val;
        }}
        i += gridSize;
    }}
    sdata[tid] = maxVal;
    GroupMemoryBarrierWithGroupSync();

    // Reduction in shared memory using max
    if (LOCAL_SIZE_X >= 512) {{ if (tid < 256) {{ if (sdata[tid + 256] > sdata[tid]) sdata[tid] = sdata[tid + 256]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 256) {{ if (tid < 128) {{ if (sdata[tid + 128] > sdata[tid]) sdata[tid] = sdata[tid + 128]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 128) {{ if (tid < 64) {{ if (sdata[tid + 64] > sdata[tid]) sdata[tid] = sdata[tid + 64]; }} GroupMemoryBarrierWithGroupSync(); }}

    if (tid < 32) {{
        if (LOCAL_SIZE_X >= 64) {{ if (sdata[tid + 32] > sdata[tid]) sdata[tid] = sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 32) {{ if (sdata[tid + 16] > sdata[tid]) sdata[tid] = sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 16) {{ if (sdata[tid + 8] > sdata[tid]) sdata[tid] = sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 8) {{ if (sdata[tid + 4] > sdata[tid]) sdata[tid] = sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 4) {{ if (sdata[tid + 2] > sdata[tid]) sdata[tid] = sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 2) {{ if (sdata[tid + 1] > sdata[tid]) sdata[tid] = sdata[tid + 1]; }}
    }}

    if (tid == 0) {{
        partialMax[gid.x] = sdata[0];
    }}
}}
)",
                                real);

  return src;
}

// Weighted squared sum reduction shader (for WrmsNorm, WL2Norm, WSqrSum)
static const std::string WSqrSumShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint numGroups;
    uint useMask;  // 0 = no mask, 1 = use mask
    uint pad;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> X;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> W;
[[vk::binding(2,0)]] RWStructuredBuffer<{0}> ID;  // mask (only used if useMask=1)
[[vk::binding(3,0)]] RWStructuredBuffer<{0}> partialSums;

groupshared {0} sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{{
    uint tid = gtid.x;
    uint i = gid.x * (LOCAL_SIZE_X * 2) + tid;
    uint gridSize = LOCAL_SIZE_X * 2 * params.numGroups;

    {0} sum = ({0})0.0;
    while (i < params.n) {{
        if (params.useMask == 0 || ID[i] > ({0})0.0) {{
            {0} v = X[i] * W[i];
            sum += v * v;
        }}
        if (i + LOCAL_SIZE_X < params.n) {{
            if (params.useMask == 0 || ID[i + LOCAL_SIZE_X] > ({0})0.0) {{
                {0} v = X[i + LOCAL_SIZE_X] * W[i + LOCAL_SIZE_X];
                sum += v * v;
            }}
        }}
        i += gridSize;
    }}
    sdata[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    if (LOCAL_SIZE_X >= 512) {{ if (tid < 256) {{ sdata[tid] += sdata[tid + 256]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 256) {{ if (tid < 128) {{ sdata[tid] += sdata[tid + 128]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 128) {{ if (tid < 64) {{ sdata[tid] += sdata[tid + 64]; }} GroupMemoryBarrierWithGroupSync(); }}

    if (tid < 32) {{
        if (LOCAL_SIZE_X >= 64) {{ sdata[tid] += sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 32) {{ sdata[tid] += sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 16) {{ sdata[tid] += sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 8) {{ sdata[tid] += sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 4) {{ sdata[tid] += sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 2) {{ sdata[tid] += sdata[tid + 1]; }}
    }}

    if (tid == 0) {{
        partialSums[gid.x] = sdata[0];
    }}
}}
)",
                                real);

  return src;
}

// Min quotient reduction shader (finds min(num/denom) where denom != 0)
static const std::string MinQuotientShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";
  const char* big = (sizeof(ShaderFloat) == sizeof(double)) ? "1.0e308"
                                                            : "3.4e38";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint numGroups;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> Num;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> Denom;
[[vk::binding(2,0)]] RWStructuredBuffer<{0}> partialMin;

groupshared {0} sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{{
    uint tid = gtid.x;
    uint i = gid.x * (LOCAL_SIZE_X * 2) + tid;
    uint gridSize = LOCAL_SIZE_X * 2 * params.numGroups;

    {0} minVal = ({0}){1};
    while (i < params.n) {{
        if (Denom[i] != ({0})0.0) {{
            {0} q = Num[i] / Denom[i];
            if (q < minVal) minVal = q;
        }}
        if (i + LOCAL_SIZE_X < params.n && Denom[i + LOCAL_SIZE_X] != ({0})0.0) {{
            {0} q = Num[i + LOCAL_SIZE_X] / Denom[i + LOCAL_SIZE_X];
            if (q < minVal) minVal = q;
        }}
        i += gridSize;
    }}
    sdata[tid] = minVal;
    GroupMemoryBarrierWithGroupSync();

    if (LOCAL_SIZE_X >= 512) {{ if (tid < 256) {{ if (sdata[tid + 256] < sdata[tid]) sdata[tid] = sdata[tid + 256]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 256) {{ if (tid < 128) {{ if (sdata[tid + 128] < sdata[tid]) sdata[tid] = sdata[tid + 128]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 128) {{ if (tid < 64) {{ if (sdata[tid + 64] < sdata[tid]) sdata[tid] = sdata[tid + 64]; }} GroupMemoryBarrierWithGroupSync(); }}

    if (tid < 32) {{
        if (LOCAL_SIZE_X >= 64) {{ if (sdata[tid + 32] < sdata[tid]) sdata[tid] = sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 32) {{ if (sdata[tid + 16] < sdata[tid]) sdata[tid] = sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 16) {{ if (sdata[tid + 8] < sdata[tid]) sdata[tid] = sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 8) {{ if (sdata[tid + 4] < sdata[tid]) sdata[tid] = sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 4) {{ if (sdata[tid + 2] < sdata[tid]) sdata[tid] = sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 2) {{ if (sdata[tid + 1] < sdata[tid]) sdata[tid] = sdata[tid + 1]; }}
    }}

    if (tid == 0) {{
        partialMin[gid.x] = sdata[0];
    }}
}}
)",
                                real, big);

  return src;
}

// InvTest shader (computes z = 1/x, returns flag if any x == 0)
static const std::string InvTestShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint numGroups;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> X;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> Z;
[[vk::binding(2,0)]] RWStructuredBuffer<uint> hasZero;  // per-workgroup flag

groupshared uint sHasZero;

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{{
    uint tid = gtid.x;
    uint i = dtid.x;

    if (tid == 0) sHasZero = 0;
    GroupMemoryBarrierWithGroupSync();

    if (i < params.n) {{
        if (X[i] == ({0})0.0) {{
            Z[i] = ({0})0.0;
            InterlockedOr(sHasZero, 1);
        }} else {{
            Z[i] = ({0})1.0 / X[i];
        }}
    }}
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {{
        hasZero[gid.x] = sHasZero;
    }}
}}
)",
                                real);

  return src;
}

static const std::string ReduceOrShaderSource()
{
  std::string src = R"(
struct Params {
    uint n;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<uint> flags;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> result;

groupshared uint sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint tid = gtid.x;

    uint val = 0;
    if (tid < params.n) {
        val = flags[tid];
    }
    for (uint i = tid + LOCAL_SIZE_X; i < params.n; i += LOCAL_SIZE_X) {
        val |= flags[i];
    }
    sdata[tid] = val;
    GroupMemoryBarrierWithGroupSync();

    if (LOCAL_SIZE_X >= 512) { if (tid < 256) { sdata[tid] |= sdata[tid + 256]; } GroupMemoryBarrierWithGroupSync(); }
    if (LOCAL_SIZE_X >= 256) { if (tid < 128) { sdata[tid] |= sdata[tid + 128]; } GroupMemoryBarrierWithGroupSync(); }
    if (LOCAL_SIZE_X >= 128) { if (tid < 64) { sdata[tid] |= sdata[tid + 64]; } GroupMemoryBarrierWithGroupSync(); }

    if (tid < 32) {
        if (LOCAL_SIZE_X >= 64) { sdata[tid] |= sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }
        if (LOCAL_SIZE_X >= 32) { sdata[tid] |= sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }
        if (LOCAL_SIZE_X >= 16) { sdata[tid] |= sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }
        if (LOCAL_SIZE_X >= 8) { sdata[tid] |= sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }
        if (LOCAL_SIZE_X >= 4) { sdata[tid] |= sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }
        if (LOCAL_SIZE_X >= 2) { sdata[tid] |= sdata[tid + 1]; }
    }

    if (tid == 0) {
        result[0] = sdata[0];
    }
}
)";

  return src;
}

// ConstrMask shader (checks constraints and sets mask)
static const std::string ConstrMaskShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint numGroups;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> C;  // constraints
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> X;  // values
[[vk::binding(2,0)]] RWStructuredBuffer<{0}> M;  // mask output
[[vk::binding(3,0)]] RWStructuredBuffer<uint> hasViolation;

groupshared uint sViolation;

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{{
    uint tid = gtid.x;
    uint i = dtid.x;

    if (tid == 0) sViolation = 0;
    GroupMemoryBarrierWithGroupSync();

    if (i < params.n) {{
        M[i] = ({0})0.0;

        {0} c = C[i];
        {0} x = X[i];

        if (c != ({0})0.0) {{
            {0} absC = abs(c);
            {0} prod = x * c;
            // |c| > 1.5 means c = +/-2: strict inequality (x*c must be > 0)
            // |c| > 0.5 means c = +/-1 or +/-2: non-strict (x*c must be >= 0)
            bool violated = (absC > ({0})1.5 && prod <= ({0})0.0) ||
                           (absC > ({0})0.5 && prod < ({0})0.0);
            if (violated) {{
                M[i] = ({0})1.0;
                InterlockedOr(sViolation, 1);
            }}
        }}
    }}
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {{
        hasViolation[gid.x] = sViolation;
    }}
}}
)",
                                real);

  return src;
}

// Linear combination shader (z = sum of c[i] * X[i])
static const std::string LinearCombShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  // This shader handles up to 8 vectors. For more, we call it multiple times.
  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint nvec;
    uint zIsX0;  // 1 if Z aliases X[0]
    uint pad;
    float c0, c1, c2, c3, c4, c5, c6, c7;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> X0;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> X1;
[[vk::binding(2,0)]] RWStructuredBuffer<{0}> X2;
[[vk::binding(3,0)]] RWStructuredBuffer<{0}> X3;
[[vk::binding(4,0)]] RWStructuredBuffer<{0}> X4;
[[vk::binding(5,0)]] RWStructuredBuffer<{0}> X5;
[[vk::binding(6,0)]] RWStructuredBuffer<{0}> X6;
[[vk::binding(7,0)]] RWStructuredBuffer<{0}> X7;
[[vk::binding(8,0)]] RWStructuredBuffer<{0}> Z;

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{{
    uint i = dtid.x;
    if (i >= params.n) return;

    {0} sum = ({0})0.0;

    // If Z aliases X[0] and c[0] == 1, we're accumulating into Z
    if (params.zIsX0 == 1) {{
        sum = Z[i];
        if (params.nvec > 1) sum += ({0})params.c1 * X1[i];
        if (params.nvec > 2) sum += ({0})params.c2 * X2[i];
        if (params.nvec > 3) sum += ({0})params.c3 * X3[i];
        if (params.nvec > 4) sum += ({0})params.c4 * X4[i];
        if (params.nvec > 5) sum += ({0})params.c5 * X5[i];
        if (params.nvec > 6) sum += ({0})params.c6 * X6[i];
        if (params.nvec > 7) sum += ({0})params.c7 * X7[i];
    }} else {{
        sum = ({0})params.c0 * X0[i];
        if (params.nvec > 1) sum += ({0})params.c1 * X1[i];
        if (params.nvec > 2) sum += ({0})params.c2 * X2[i];
        if (params.nvec > 3) sum += ({0})params.c3 * X3[i];
        if (params.nvec > 4) sum += ({0})params.c4 * X4[i];
        if (params.nvec > 5) sum += ({0})params.c5 * X5[i];
        if (params.nvec > 6) sum += ({0})params.c6 * X6[i];
        if (params.nvec > 7) sum += ({0})params.c7 * X7[i];
    }}

    Z[i] = sum;
}}
)",
                                real);

  return src;
}

// ScaleAddMulti shader (Z[j] = c[j] * X + Y[j])
static const std::string ScaleAddShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    float c;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> X;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> Y;
[[vk::binding(2,0)]] RWStructuredBuffer<{0}> Z;

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{{
    uint i = dtid.x;
    if (i >= params.n) return;

    Z[i] = ({0})params.c * X[i] + Y[i];
}}
)",
                                real);

  return src;
}

// Final max reduction shader: finds max of partial results
static const std::string FinalMaxReduceShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint pad0;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> partialMax;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> result;

groupshared {0} sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{{
    uint tid = gtid.x;

    // Load partial max values into shared memory
    {0} maxVal = ({0})0.0;
    if (tid < params.n) {{
        maxVal = partialMax[tid];
    }}
    for (uint i = tid + LOCAL_SIZE_X; i < params.n; i += LOCAL_SIZE_X) {{
        if (partialMax[i] > maxVal) maxVal = partialMax[i];
    }}
    sdata[tid] = maxVal;
    GroupMemoryBarrierWithGroupSync();

    // Reduction using max
    if (LOCAL_SIZE_X >= 512) {{ if (tid < 256) {{ if (sdata[tid + 256] > sdata[tid]) sdata[tid] = sdata[tid + 256]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 256) {{ if (tid < 128) {{ if (sdata[tid + 128] > sdata[tid]) sdata[tid] = sdata[tid + 128]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 128) {{ if (tid < 64) {{ if (sdata[tid + 64] > sdata[tid]) sdata[tid] = sdata[tid + 64]; }} GroupMemoryBarrierWithGroupSync(); }}

    if (tid < 32) {{
        if (LOCAL_SIZE_X >= 64) {{ if (sdata[tid + 32] > sdata[tid]) sdata[tid] = sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 32) {{ if (sdata[tid + 16] > sdata[tid]) sdata[tid] = sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 16) {{ if (sdata[tid + 8] > sdata[tid]) sdata[tid] = sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 8) {{ if (sdata[tid + 4] > sdata[tid]) sdata[tid] = sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 4) {{ if (sdata[tid + 2] > sdata[tid]) sdata[tid] = sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 2) {{ if (sdata[tid + 1] > sdata[tid]) sdata[tid] = sdata[tid + 1]; }}
    }}

    if (tid == 0) {{
        result[0] = sdata[0];
    }}
}}
)",
                                real);

  return src;
}

static const std::string FinalMinReduceShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";
  const char* big = (sizeof(ShaderFloat) == sizeof(double)) ? "1.0e308"
                                                            : "3.4e38";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;
    uint pad0;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> partialMin;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> result;

groupshared {0} sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{{
    uint tid = gtid.x;

    {0} minVal = ({0}){1};
    if (tid < params.n) {{
        minVal = partialMin[tid];
    }}
    for (uint i = tid + LOCAL_SIZE_X; i < params.n; i += LOCAL_SIZE_X) {{
        if (partialMin[i] < minVal) minVal = partialMin[i];
    }}
    sdata[tid] = minVal;
    GroupMemoryBarrierWithGroupSync();

    if (LOCAL_SIZE_X >= 512) {{ if (tid < 256) {{ if (sdata[tid + 256] < sdata[tid]) sdata[tid] = sdata[tid + 256]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 256) {{ if (tid < 128) {{ if (sdata[tid + 128] < sdata[tid]) sdata[tid] = sdata[tid + 128]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 128) {{ if (tid < 64) {{ if (sdata[tid + 64] < sdata[tid]) sdata[tid] = sdata[tid + 64]; }} GroupMemoryBarrierWithGroupSync(); }}

    if (tid < 32) {{
        if (LOCAL_SIZE_X >= 64) {{ if (sdata[tid + 32] < sdata[tid]) sdata[tid] = sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 32) {{ if (sdata[tid + 16] < sdata[tid]) sdata[tid] = sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 16) {{ if (sdata[tid + 8] < sdata[tid]) sdata[tid] = sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 8) {{ if (sdata[tid + 4] < sdata[tid]) sdata[tid] = sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 4) {{ if (sdata[tid + 2] < sdata[tid]) sdata[tid] = sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 2) {{ if (sdata[tid + 1] < sdata[tid]) sdata[tid] = sdata[tid + 1]; }}
    }}

    if (tid == 0) {{
        result[0] = sdata[0];
    }}
}}
)",
                                real, big);

  return src;
}

// Reduction shader: sums partial results from first pass
static const std::string FinalSumReduceShaderSource()
{
  const char* real = (sizeof(ShaderFloat) == sizeof(double)) ? "double" : "float";

  std::string src = fmt::format(R"(
struct Params {{
    uint n;        // number of partial sums to reduce
    uint pad0;
    uint pad1;
    uint pad2;
}};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0,0)]] RWStructuredBuffer<{0}> partialSums;
[[vk::binding(1,0)]] RWStructuredBuffer<{0}> result;

groupshared {0} sdata[LOCAL_SIZE_X];

[numthreads(LOCAL_SIZE_X, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID)
{{
    uint tid = gtid.x;

    // Load partial sums into shared memory
    {0} sum = ({0})0.0;
    if (tid < params.n) {{
        sum = partialSums[tid];
    }}
    // Handle case where we have more partial sums than threads
    for (uint i = tid + LOCAL_SIZE_X; i < params.n; i += LOCAL_SIZE_X) {{
        sum += partialSums[i];
    }}
    sdata[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    // Reduction in shared memory
    if (LOCAL_SIZE_X >= 512) {{ if (tid < 256) {{ sdata[tid] += sdata[tid + 256]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 256) {{ if (tid < 128) {{ sdata[tid] += sdata[tid + 128]; }} GroupMemoryBarrierWithGroupSync(); }}
    if (LOCAL_SIZE_X >= 128) {{ if (tid < 64) {{ sdata[tid] += sdata[tid + 64]; }} GroupMemoryBarrierWithGroupSync(); }}

    if (tid < 32) {{
        if (LOCAL_SIZE_X >= 64) {{ sdata[tid] += sdata[tid + 32]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 32) {{ sdata[tid] += sdata[tid + 16]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 16) {{ sdata[tid] += sdata[tid + 8]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 8) {{ sdata[tid] += sdata[tid + 4]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 4) {{ sdata[tid] += sdata[tid + 2]; GroupMemoryBarrierWithGroupSync(); }}
        if (LOCAL_SIZE_X >= 2) {{ sdata[tid] += sdata[tid + 1]; }}
    }}

    // Thread 0 writes final result
    if (tid == 0) {{
        result[0] = sdata[0];
    }}
}}
)",
                                real);

  return src;
}

static std::vector<uint32_t> CompileSlangToSpirv(
  const std::string& source, const std::string& entry,
  const std::array<uint32_t, 3>& localSizes)
{
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  std::filesystem::path src    = tmpdir / "nvector_vulkan_tmp.slang";
  std::filesystem::path spv    = tmpdir / "nvector_vulkan_tmp.spv";

  {
    std::ofstream out(src);
    out << source;
  }

  std::stringstream cmd;
  cmd << "slangc -target spirv -profile cs_6_2 -entry " << entry << " "
      << "-DLOCAL_SIZE_X=" << localSizes[0] << " "
      << "-DLOCAL_SIZE_Y=" << localSizes[1] << " "
      << "-DLOCAL_SIZE_Z=" << localSizes[2] << " "
      << "-o " << spv << " " << src;

  int rc = std::system(cmd.str().c_str());
  if (rc != 0)
  {
    throw std::runtime_error(
      "slangc failed when compiling Vulkan NVECTOR shaders");
  }

  std::ifstream spv_in(spv, std::ios::binary);
  std::vector<char> spirv_bytes((std::istreambuf_iterator<char>(spv_in)),
                                std::istreambuf_iterator<char>());

  // Cleanup temp files
  // std::filesystem::remove(src);
  // std::filesystem::remove(spv);

  // convert byte stream to uint32_t vector
  std::vector<uint32_t> words;
  words.resize(spirv_bytes.size() / sizeof(uint32_t));
  std::memcpy(words.data(), spirv_bytes.data(), spirv_bytes.size());
  return words;
}

static const std::vector<uint32_t>& GetElementwiseSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(ElementwiseShaderSource(), "main", kLocalSizes);
  return spirv;
}

static const std::vector<uint32_t>& GetDotProdSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(DotProdShaderSource(), "main", kLocalSizes);
  return spirv;
}

static const std::vector<uint32_t>& GetFinalSumReduceSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(FinalSumReduceShaderSource(), "main", kLocalSizes);
  return spirv;
}

static const std::vector<uint32_t>& GetMaxNormSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(MaxNormShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetFinalMaxReduceSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(FinalMaxReduceShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetMinQuotientSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(MinQuotientShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetFinalMinReduceSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(FinalMinReduceShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetInvTestSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(InvTestShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetReduceOrSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(ReduceOrShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetWSqrSumSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(WSqrSumShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetConstrMaskSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(ConstrMaskShaderSource(), "main", {256, 1, 1});
  return spirv;
}

static const std::vector<uint32_t>& GetScaleAddSpirv()
{
  static std::vector<uint32_t> spirv =
    CompileSlangToSpirv(ScaleAddShaderSource(), "main", {256, 1, 1});
  return spirv;
}

// ---------------------------------------------------------------------------
// Kompute dispatch helpers
// ---------------------------------------------------------------------------

enum class ElementwiseOp : uint32_t
{
  LinearSum = 0,
  Const     = 1,
  Prod      = 2,
  Div       = 3,
  Scale     = 4,
  Abs       = 5,
  Inv       = 6,
  AddConst  = 7,
  Compare   = 8
};

static void DispatchElementwise(ElementwiseOp op, sunrealtype a, sunrealtype b,
                                N_Vector x, N_Vector y, N_Vector z)
{
  auto privX = NVEC_VULKAN_PRIVATE(x);
  auto privY = y ? NVEC_VULKAN_PRIVATE(y) : nullptr;
  auto privZ = NVEC_VULKAN_PRIVATE(z);

  N_VCopyToDevice_Vulkan(x);
  if (y) { N_VCopyToDevice_Vulkan(y); }
  EnsureTensor(z);

  const auto& spirv = GetElementwiseSpirv();
  auto seq          = privZ->manager->sequence();

  // Sync inputs and outputs
  std::vector<std::shared_ptr<kp::Memory>> memObjects;
  memObjects.reserve(3);
  memObjects.push_back(privX->device_data);
  // Always provide three bindings to match the shader layout; reuse X when Y is absent.
  if (y) { memObjects.push_back(privY->device_data); }
  else { memObjects.push_back(privX->device_data); }
  memObjects.push_back(privZ->device_data);
  seq->record<kp::OpSyncDevice>(memObjects);

  struct Push
  {
    uint32_t op;
    float a;
    float b;
    uint32_t n;
  };

  Push push{static_cast<uint32_t>(op), static_cast<float>(a),
            static_cast<float>(b), static_cast<uint32_t>(NVEC_VULKAN_LENGTH(z))};

  std::vector<uint8_t> pushConstants(sizeof(Push));
  std::memcpy(pushConstants.data(), &push, sizeof(Push));

  auto stream_policy = NVEC_VULKAN_CONTENT(z)->stream_exec_policy;
  auto algo          = privZ->manager->algorithm(memObjects, spirv,
                                                 {stream_policy->gridSize(push.n), 1, 1},
                                                 std::vector<uint32_t>{}, pushConstants);

  seq->record<kp::OpAlgoDispatch>(algo);
  seq->record<kp::OpSyncLocal>(
    {std::static_pointer_cast<kp::Memory>(privZ->device_data)});
  seq->eval();

  // Copy results back to host - match the pattern in N_VCopyFromDevice_Vulkan
  ShaderFloat* from_shader = privZ->device_data->data<ShaderFloat>();
  FromShaderBuffer<ShaderFloat>({from_shader, privZ->device_data->size()},
                                HostData(z));
  privZ->device_needs_update = false;
  privZ->host_needs_update   = false;
}

// GPU-accelerated dot product using parallel reduction
static sunrealtype DispatchDotProdReduction(N_Vector x, N_Vector y)
{
  auto* privX = NVEC_VULKAN_PRIVATE(x);
  auto* privY = NVEC_VULKAN_PRIVATE(y);

  N_VCopyToDevice_Vulkan(x);
  N_VCopyToDevice_Vulkan(y);

  const uint32_t n = static_cast<uint32_t>(NVEC_VULKAN_LENGTH(x));

  // Use reduce_exec_policy for reduction operations (optimized for reductions).
  // The policy's blockSize() must match the shader's LOCAL_SIZE_X (256).
  auto* reduce_policy      = NVEC_VULKAN_CONTENT(x)->reduce_exec_policy;
  const uint32_t blockSize = reduce_policy->blockSize(n);
  const uint32_t numGroups = reduce_policy->gridSize(n);

  // Sanity check: shader was compiled with LOCAL_SIZE_X=256
  assert(blockSize == kLocalSizes[0] &&
         "reduce_exec_policy blockSize must match shader LOCAL_SIZE_X");

  // Create tensor for partial sums (one per workgroup)
  auto partialSumsTensor =
    privX->manager->tensor(nullptr, numGroups, sizeof(ShaderFloat),
                           kp::Memory::dataType<ShaderFloat>(),
                           kp::Memory::MemoryTypes::eDevice);

  // Create tensor for final result (single element)
  auto resultTensor = privX->manager->tensor(nullptr, 1, sizeof(ShaderFloat),
                                             kp::Memory::dataType<ShaderFloat>(),
                                             kp::Memory::MemoryTypes::eDevice);

  // First pass: compute partial sums per workgroup
  {
    const auto& spirv = GetDotProdSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(privX->device_data);
    memObjects.push_back(privY->device_data);
    memObjects.push_back(partialSumsTensor);

    struct Push
    {
      uint32_t n;
      uint32_t numGroups;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{n, numGroups, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();
    seq->record<kp::OpSyncDevice>(memObjects);

    auto algo = privX->manager->algorithm(memObjects, spirv, {numGroups, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->eval();
  }

  // Second pass: reduce partial sums to final result
  {
    const auto& spirv = GetFinalSumReduceSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(partialSumsTensor);
    memObjects.push_back(resultTensor);

    struct Push
    {
      uint32_t n;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{numGroups, 0, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();

    auto algo = privX->manager->algorithm(memObjects, spirv, {1, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(resultTensor)});
    seq->eval();
  }

  // Read result back
  ShaderFloat result = resultTensor->data<ShaderFloat>()[0];
  return static_cast<sunrealtype>(result);
}

// GPU-accelerated max norm using parallel reduction
static sunrealtype DispatchMaxNormReduction(N_Vector x)
{
  auto* privX = NVEC_VULKAN_PRIVATE(x);

  N_VCopyToDevice_Vulkan(x);

  const uint32_t n = static_cast<uint32_t>(NVEC_VULKAN_LENGTH(x));

  auto* reduce_policy      = NVEC_VULKAN_CONTENT(x)->reduce_exec_policy;
  const uint32_t blockSize = reduce_policy->blockSize(n);
  const uint32_t numGroups = reduce_policy->gridSize(n);

  assert(blockSize == kLocalSizes[0] &&
         "reduce_exec_policy blockSize must match shader LOCAL_SIZE_X");

  // Create tensor for partial max values (one per workgroup)
  auto partialMaxTensor =
    privX->manager->tensor(nullptr, numGroups, sizeof(ShaderFloat),
                           kp::Memory::dataType<ShaderFloat>(),
                           kp::Memory::MemoryTypes::eDevice);

  // Create tensor for final result (single element)
  auto resultTensor = privX->manager->tensor(nullptr, 1, sizeof(ShaderFloat),
                                             kp::Memory::dataType<ShaderFloat>(),
                                             kp::Memory::MemoryTypes::eDevice);

  // First pass: compute partial max per workgroup
  {
    const auto& spirv = GetMaxNormSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(privX->device_data);
    memObjects.push_back(partialMaxTensor);

    struct Push
    {
      uint32_t n;
      uint32_t numGroups;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{n, numGroups, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();
    seq->record<kp::OpSyncDevice>(memObjects);

    auto algo = privX->manager->algorithm(memObjects, spirv, {numGroups, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->eval();
  }

  // Second pass: reduce partial max values to final result
  {
    const auto& spirv = GetFinalMaxReduceSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(partialMaxTensor);
    memObjects.push_back(resultTensor);

    struct Push
    {
      uint32_t n;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{numGroups, 0, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();

    auto algo = privX->manager->algorithm(memObjects, spirv, {1, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(resultTensor)});
    seq->eval();
  }

  // Read result back
  ShaderFloat result = resultTensor->data<ShaderFloat>()[0];
  return static_cast<sunrealtype>(result);
}

// GPU-accelerated weighted squared sum (used by WrmsNorm, WL2Norm, WSqrSum)
static sunrealtype DispatchWSqrSumReduction(N_Vector x, N_Vector w, N_Vector id,
                                            bool useMask)
{
  auto* privX = NVEC_VULKAN_PRIVATE(x);
  auto* privW = NVEC_VULKAN_PRIVATE(w);

  N_VCopyToDevice_Vulkan(x);
  N_VCopyToDevice_Vulkan(w);
  if (useMask) { N_VCopyToDevice_Vulkan(id); }

  const uint32_t n = static_cast<uint32_t>(NVEC_VULKAN_LENGTH(x));

  auto* reduce_policy      = NVEC_VULKAN_CONTENT(x)->reduce_exec_policy;
  const uint32_t blockSize = reduce_policy->blockSize(n);
  const uint32_t numGroups = reduce_policy->gridSize(n);

  assert(blockSize == kLocalSizes[0] &&
         "reduce_exec_policy blockSize must match shader LOCAL_SIZE_X");

  auto partialSumsTensor =
    privX->manager->tensor(nullptr, numGroups, sizeof(ShaderFloat),
                           kp::Memory::dataType<ShaderFloat>(),
                           kp::Memory::MemoryTypes::eDevice);

  auto resultTensor = privX->manager->tensor(nullptr, 1, sizeof(ShaderFloat),
                                             kp::Memory::dataType<ShaderFloat>(),
                                             kp::Memory::MemoryTypes::eDevice);

  // First pass
  {
    const auto& spirv = GetWSqrSumSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(privX->device_data);
    memObjects.push_back(privW->device_data);
    // For mask: use id if provided, otherwise reuse X (shader ignores it when useMask=0)
    if (useMask) { memObjects.push_back(NVEC_VULKAN_PRIVATE(id)->device_data); }
    else { memObjects.push_back(privX->device_data); }
    memObjects.push_back(partialSumsTensor);

    struct Push
    {
      uint32_t n;
      uint32_t numGroups;
      uint32_t useMask;
      uint32_t pad;
    };

    Push push{n, numGroups, useMask ? 1u : 0u, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();
    seq->record<kp::OpSyncDevice>(memObjects);

    auto algo = privX->manager->algorithm(memObjects, spirv, {numGroups, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->eval();
  }

  // Second pass
  {
    const auto& spirv = GetFinalSumReduceSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(partialSumsTensor);
    memObjects.push_back(resultTensor);

    struct Push
    {
      uint32_t n;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{numGroups, 0, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();

    auto algo = privX->manager->algorithm(memObjects, spirv, {1, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(resultTensor)});
    seq->eval();
  }

  ShaderFloat result = resultTensor->data<ShaderFloat>()[0];
  return static_cast<sunrealtype>(result);
}

// GPU-accelerated min quotient reduction
static sunrealtype DispatchMinQuotientReduction(N_Vector num, N_Vector denom)
{
  auto* privN = NVEC_VULKAN_PRIVATE(num);
  auto* privD = NVEC_VULKAN_PRIVATE(denom);

  N_VCopyToDevice_Vulkan(num);
  N_VCopyToDevice_Vulkan(denom);

  const uint32_t n = static_cast<uint32_t>(NVEC_VULKAN_LENGTH(num));

  auto* reduce_policy      = NVEC_VULKAN_CONTENT(num)->reduce_exec_policy;
  const uint32_t blockSize = reduce_policy->blockSize(n);
  const uint32_t numGroups = reduce_policy->gridSize(n);

  assert(blockSize == kLocalSizes[0] &&
         "reduce_exec_policy blockSize must match shader LOCAL_SIZE_X");

  auto partialMinTensor =
    privN->manager->tensor(nullptr, numGroups, sizeof(ShaderFloat),
                           kp::Memory::dataType<ShaderFloat>(),
                           kp::Memory::MemoryTypes::eDevice);

  auto resultTensor = privN->manager->tensor(nullptr, 1, sizeof(ShaderFloat),
                                             kp::Memory::dataType<ShaderFloat>(),
                                             kp::Memory::MemoryTypes::eDevice);

  // First pass
  {
    const auto& spirv = GetMinQuotientSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(privN->device_data);
    memObjects.push_back(privD->device_data);
    memObjects.push_back(partialMinTensor);

    struct Push
    {
      uint32_t n;
      uint32_t numGroups;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{n, numGroups, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privN->manager->sequence();
    seq->record<kp::OpSyncDevice>(memObjects);

    auto algo = privN->manager->algorithm(memObjects, spirv, {numGroups, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->eval();
  }

  // Second pass
  {
    const auto& spirv = GetFinalMinReduceSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(partialMinTensor);
    memObjects.push_back(resultTensor);

    struct Push
    {
      uint32_t n;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{numGroups, 0, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privN->manager->sequence();

    auto algo = privN->manager->algorithm(memObjects, spirv, {1, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(resultTensor)});
    seq->eval();
  }

  ShaderFloat result = resultTensor->data<ShaderFloat>()[0];
  return static_cast<sunrealtype>(result);
}

// GPU-accelerated inverse test (z = 1/x, returns false if any x == 0)
static sunbooleantype DispatchInvTest(N_Vector x, N_Vector z)
{
  auto* privX = NVEC_VULKAN_PRIVATE(x);
  auto* privZ = NVEC_VULKAN_PRIVATE(z);

  N_VCopyToDevice_Vulkan(x);
  EnsureTensor(z);

  const uint32_t n = static_cast<uint32_t>(NVEC_VULKAN_LENGTH(x));

  auto* stream_policy      = NVEC_VULKAN_CONTENT(x)->stream_exec_policy;
  const uint32_t numGroups = stream_policy->gridSize(n);

  // Tensor for per-workgroup zero flags
  auto hasZeroTensor = privX->manager->tensor(nullptr, numGroups,
                                              sizeof(uint32_t),
                                              kp::Memory::dataType<uint32_t>(),
                                              kp::Memory::MemoryTypes::eDevice);

  auto resultTensor = privX->manager->tensor(nullptr, 1, sizeof(uint32_t),
                                             kp::Memory::dataType<uint32_t>(),
                                             kp::Memory::MemoryTypes::eDevice);

  // First pass: compute inverses and flag zeros
  {
    const auto& spirv = GetInvTestSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(privX->device_data);
    memObjects.push_back(privZ->device_data);
    memObjects.push_back(hasZeroTensor);

    struct Push
    {
      uint32_t n;
      uint32_t numGroups;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{n, numGroups, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();
    seq->record<kp::OpSyncDevice>(memObjects);

    auto algo = privX->manager->algorithm(memObjects, spirv, {numGroups, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->eval();
  }

  // Second pass: reduce OR of flags
  {
    const auto& spirv = GetReduceOrSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(hasZeroTensor);
    memObjects.push_back(resultTensor);

    struct Push
    {
      uint32_t n;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{numGroups, 0, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();

    auto algo = privX->manager->algorithm(memObjects, spirv, {1, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(resultTensor)});
    seq->eval();
  }

  // Sync Z back to host
  {
    auto seq = privZ->manager->sequence();
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(privZ->device_data)});
    seq->eval();

    ShaderFloat* from_shader = privZ->device_data->data<ShaderFloat>();
    FromShaderBuffer<ShaderFloat>({from_shader, privZ->device_data->size()},
                                  HostData(z));
    privZ->device_needs_update = false;
    privZ->host_needs_update   = false;
  }

  uint32_t hasZero = resultTensor->data<uint32_t>()[0];
  return hasZero ? SUNFALSE : SUNTRUE;
}

// GPU-accelerated constraint mask check
static sunbooleantype DispatchConstrMask(N_Vector c, N_Vector x, N_Vector m)
{
  auto* privC = NVEC_VULKAN_PRIVATE(c);
  auto* privX = NVEC_VULKAN_PRIVATE(x);
  auto* privM = NVEC_VULKAN_PRIVATE(m);

  N_VCopyToDevice_Vulkan(c);
  N_VCopyToDevice_Vulkan(x);
  EnsureTensor(m);

  const uint32_t n = static_cast<uint32_t>(NVEC_VULKAN_LENGTH(x));

  auto* stream_policy      = NVEC_VULKAN_CONTENT(x)->stream_exec_policy;
  const uint32_t numGroups = stream_policy->gridSize(n);

  auto hasViolationTensor =
    privX->manager->tensor(nullptr, numGroups, sizeof(uint32_t),
                           kp::Memory::dataType<uint32_t>(),
                           kp::Memory::MemoryTypes::eDevice);

  auto resultTensor = privX->manager->tensor(nullptr, 1, sizeof(uint32_t),
                                             kp::Memory::dataType<uint32_t>(),
                                             kp::Memory::MemoryTypes::eDevice);

  // First pass: check constraints and set mask
  {
    const auto& spirv = GetConstrMaskSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(privC->device_data);
    memObjects.push_back(privX->device_data);
    memObjects.push_back(privM->device_data);
    memObjects.push_back(hasViolationTensor);

    struct Push
    {
      uint32_t n;
      uint32_t numGroups;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{n, numGroups, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();
    seq->record<kp::OpSyncDevice>(memObjects);

    auto algo = privX->manager->algorithm(memObjects, spirv, {numGroups, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->eval();
  }

  // Second pass: reduce OR of violation flags
  {
    const auto& spirv = GetReduceOrSpirv();

    std::vector<std::shared_ptr<kp::Memory>> memObjects;
    memObjects.push_back(hasViolationTensor);
    memObjects.push_back(resultTensor);

    struct Push
    {
      uint32_t n;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    Push push{numGroups, 0, 0, 0};

    std::vector<uint8_t> pushConstants(sizeof(Push));
    std::memcpy(pushConstants.data(), &push, sizeof(Push));

    auto seq = privX->manager->sequence();

    auto algo = privX->manager->algorithm(memObjects, spirv, {1, 1, 1},
                                          std::vector<uint32_t>{}, pushConstants);

    seq->record<kp::OpAlgoDispatch>(algo);
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(resultTensor)});
    seq->eval();
  }

  // Sync M back to host
  {
    auto seq = privM->manager->sequence();
    seq->record<kp::OpSyncLocal>(
      {std::static_pointer_cast<kp::Memory>(privM->device_data)});
    seq->eval();

    ShaderFloat* from_shader = privM->device_data->data<ShaderFloat>();
    FromShaderBuffer<ShaderFloat>({from_shader, privM->device_data->size()},
                                  HostData(m));
    privM->device_needs_update = false;
    privM->host_needs_update   = false;
  }

  uint32_t hasViolation = resultTensor->data<uint32_t>()[0];
  return hasViolation ? SUNFALSE : SUNTRUE;
}

// GPU-accelerated scale-add (Z = c * X + Y)
static void DispatchScaleAdd(sunrealtype c, N_Vector x, N_Vector y, N_Vector z)
{
  auto* privX = NVEC_VULKAN_PRIVATE(x);
  auto* privY = NVEC_VULKAN_PRIVATE(y);
  auto* privZ = NVEC_VULKAN_PRIVATE(z);

  N_VCopyToDevice_Vulkan(x);
  N_VCopyToDevice_Vulkan(y);
  EnsureTensor(z);

  const uint32_t n = static_cast<uint32_t>(NVEC_VULKAN_LENGTH(x));

  const auto& spirv        = GetScaleAddSpirv();
  auto* stream_policy      = NVEC_VULKAN_CONTENT(x)->stream_exec_policy;
  const uint32_t numGroups = stream_policy->gridSize(n);

  std::vector<std::shared_ptr<kp::Memory>> memObjects;
  memObjects.push_back(privX->device_data);
  memObjects.push_back(privY->device_data);
  memObjects.push_back(privZ->device_data);

  struct Push
  {
    uint32_t n;
    float c;
    uint32_t pad1;
    uint32_t pad2;
  };

  Push push{n, static_cast<float>(c), 0, 0};

  std::vector<uint8_t> pushConstants(sizeof(Push));
  std::memcpy(pushConstants.data(), &push, sizeof(Push));

  auto seq = privZ->manager->sequence();
  seq->record<kp::OpSyncDevice>(memObjects);

  auto algo = privZ->manager->algorithm(memObjects, spirv, {numGroups, 1, 1},
                                        std::vector<uint32_t>{}, pushConstants);

  seq->record<kp::OpAlgoDispatch>(algo);
  seq->record<kp::OpSyncLocal>(
    {std::static_pointer_cast<kp::Memory>(privZ->device_data)});
  seq->eval();

  ShaderFloat* from_shader = privZ->device_data->data<ShaderFloat>();
  FromShaderBuffer<ShaderFloat>({from_shader, privZ->device_data->size()},
                                HostData(z));
  privZ->device_needs_update = false;
  privZ->host_needs_update   = false;
}

// ---------------------------------------------------------------------------
// NVECTOR creation / destruction
// ---------------------------------------------------------------------------

extern "C" {

N_Vector N_VNewEmpty_Vulkan(SUNContext sunctx)
{
  N_Vector v = N_VNewEmpty(sunctx);
  if (v == NULL) { return NULL; }

  v->content = (N_VectorContent_Vulkan)malloc(sizeof(_N_VectorContent_Vulkan));
  if (v->content == NULL)
  {
    N_VDestroy(v);
    return NULL;
  }

  auto* priv = new (std::nothrow) PrivateVectorContent_Vulkan();
  if (priv == nullptr)
  {
    N_VDestroy(v);
    return NULL;
  }
  NVEC_VULKAN_CONTENT(v)->priv = priv;

  NVEC_VULKAN_LENGTH(v)                      = 0;
  NVEC_VULKAN_CONTENT(v)->stream_exec_policy = new ExecPolicy(256);
  NVEC_VULKAN_CONTENT(v)->reduce_exec_policy = new AtomicReduceExecPolicy(256);

  priv->manager = SUNDIALS_VK_GetSharedManager();

  // Attach operations
  v->ops->nvgetvectorid           = N_VGetVectorID_Vulkan;
  v->ops->nvclone                 = N_VClone_Vulkan;
  v->ops->nvcloneempty            = N_VCloneEmpty_Vulkan;
  v->ops->nvdestroy               = N_VDestroy_Vulkan;
  v->ops->nvspace                 = N_VSpace_Vulkan;
  v->ops->nvgetlength             = N_VGetLength_Vulkan;
  v->ops->nvgetarraypointer       = N_VGetHostArrayPointer_Vulkan;
  v->ops->nvgetdevicearraypointer = N_VGetDeviceArrayPointer_Vulkan;
  v->ops->nvsetarraypointer       = N_VSetHostArrayPointer_Vulkan;

  v->ops->nvlinearsum    = N_VLinearSum_Vulkan;
  v->ops->nvconst        = N_VConst_Vulkan;
  v->ops->nvprod         = N_VProd_Vulkan;
  v->ops->nvdiv          = N_VDiv_Vulkan;
  v->ops->nvscale        = N_VScale_Vulkan;
  v->ops->nvabs          = N_VAbs_Vulkan;
  v->ops->nvinv          = N_VInv_Vulkan;
  v->ops->nvaddconst     = N_VAddConst_Vulkan;
  v->ops->nvdotprod      = N_VDotProd_Vulkan;
  v->ops->nvmaxnorm      = N_VMaxNorm_Vulkan;
  v->ops->nvmin          = N_VMin_Vulkan;
  v->ops->nvl1norm       = N_VL1Norm_Vulkan;
  v->ops->nvinvtest      = N_VInvTest_Vulkan;
  v->ops->nvconstrmask   = N_VConstrMask_Vulkan;
  v->ops->nvminquotient  = N_VMinQuotient_Vulkan;
  v->ops->nvwrmsnorm     = N_VWrmsNorm_Vulkan;
  v->ops->nvwrmsnormmask = N_VWrmsNormMask_Vulkan;
  v->ops->nvwl2norm      = N_VWL2Norm_Vulkan;
  v->ops->nvcompare      = N_VCompare_Vulkan;

  // fused ops
  v->ops->nvlinearcombination = N_VLinearCombination_Vulkan;
  v->ops->nvscaleaddmulti     = N_VScaleAddMulti_Vulkan;
  v->ops->nvdotprodmulti      = N_VDotProdMulti_Vulkan;

  // vector array operations
  v->ops->nvlinearsumvectorarray     = N_VLinearSumVectorArray_Vulkan;
  v->ops->nvscalevectorarray         = N_VScaleVectorArray_Vulkan;
  v->ops->nvconstvectorarray         = N_VConstVectorArray_Vulkan;
  v->ops->nvscaleaddmultivectorarray = N_VScaleAddMultiVectorArray_Vulkan;
  v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_Vulkan;
  v->ops->nvwrmsnormvectorarray     = N_VWrmsNormVectorArray_Vulkan;
  v->ops->nvwrmsnormmaskvectorarray = N_VWrmsNormMaskVectorArray_Vulkan;

  // optional reductions
  v->ops->nvwsqrsumlocal     = N_VWSqrSumLocal_Vulkan;
  v->ops->nvwsqrsummasklocal = N_VWSqrSumMaskLocal_Vulkan;

  // XBraid support
  v->ops->nvbufsize   = N_VBufSize_Vulkan;
  v->ops->nvbufpack   = N_VBufPack_Vulkan;
  v->ops->nvbufunpack = N_VBufUnpack_Vulkan;

  v->ops->nvprint     = N_VPrint_Vulkan;
  v->ops->nvprintfile = N_VPrintFile_Vulkan;

  return v;
}

N_Vector N_VNew_Vulkan(sunindextype length, SUNContext sunctx)
{
  N_Vector v = N_VNewEmpty_Vulkan(sunctx);
  if (v == NULL) { return NULL; }

  NVEC_VULKAN_LENGTH(v) = length;
  NVEC_VULKAN_PRIVATE(v)->host_data =
    std::vector(length, static_cast<sunrealtype>(ZERO));

  return v;
}

N_Vector N_VMake_Vulkan(sunindextype length, sunrealtype* h_vdata,
                        sunrealtype* d_vdata, SUNContext sunctx)
{
  N_Vector v = N_VNewEmpty_Vulkan(sunctx);
  if (v == NULL) { return NULL; }

  NVEC_VULKAN_LENGTH(v) = length;

  // Set up host buffer
  if (h_vdata != nullptr)
  {
    NVEC_VULKAN_PRIVATE(v)->host_data = std::vector(h_vdata, h_vdata + length);
    MarkDeviceNeedsUpdate(v);
  }
  else
  {
    NVEC_VULKAN_PRIVATE(v)->host_data =
      std::vector(length, static_cast<sunrealtype>(ZERO));
  }

  if (d_vdata != NULL)
  {
    auto* priv = NVEC_VULKAN_PRIVATE(v);
    if constexpr (kShaderMatchesSunreal)
    {
      priv->device_data =
        priv->manager->tensor(d_vdata, static_cast<uint32_t>(length),
                              sizeof(ShaderFloat),
                              kp::Memory::dataType<ShaderFloat>(),
                              kp::Memory::MemoryTypes::eDevice);
    }
    else
    {
      std::vector<ShaderFloat> shader_init =
        ToShaderBuffer<ShaderFloat>(HostData(v));
      priv->device_data =
        priv->manager->tensor(shader_init.empty() ? nullptr : shader_init.data(),
                              static_cast<uint32_t>(length), sizeof(ShaderFloat),
                              kp::Memory::dataType<ShaderFloat>(),
                              kp::Memory::MemoryTypes::eDevice);
    }
    NVEC_VULKAN_PRIVATE(v)->device_needs_update = false;
  }

  return v;
}

void N_VSetHostArrayPointer_Vulkan(sunrealtype* h_vdata, N_Vector v)
{
  auto* priv = NVEC_VULKAN_PRIVATE(v);

  if (h_vdata == nullptr)
  {
    priv->host_data = std::vector(NVEC_VULKAN_LENGTH(v),
                                  static_cast<sunrealtype>(ZERO));
  }
  else
  {
    // Check if h_vdata points to our internal vector's data - if so, don't
    // replace the variant (which would destroy the vector and make h_vdata
    // a dangling pointer)
    if (auto* vec = std::get_if<std::vector<sunrealtype>>(&priv->host_data))
    {
      if (vec->data() != h_vdata) { priv->host_data = h_vdata; }
    }
    else { priv->host_data = h_vdata; }
  }

  MarkDeviceNeedsUpdate(v);
}

sunrealtype* N_VGetHostArrayPointer_Vulkan(N_Vector x)
{
  PrivateVectorContent_Vulkan* priv = NVEC_VULKAN_PRIVATE(x);
  if (sunrealtype* const* ptr = std::get_if<sunrealtype*>(&priv->host_data))
  {
    return *ptr;
  }
  else if (std::vector<sunrealtype>* data =
             std::get_if<std::vector<sunrealtype>>(&priv->host_data))
  {
    assert(data->size() >= static_cast<size_t>(NVEC_VULKAN_LENGTH(x)));
    return data->data();
  }
  else { return nullptr; }
}

sunrealtype* N_VGetDeviceArrayPointer_Vulkan(N_Vector x)
{
  if constexpr (kShaderMatchesSunreal)
  {
    auto* priv = NVEC_VULKAN_PRIVATE(x);
    return static_cast<sunrealtype*>(priv->device_data->rawData());
  }
  else { return nullptr; }
}

SUNErrCode N_VSetKernelExecPolicy_Vulkan(N_Vector x,
                                         SUNVulkanExecPolicy* stream_exec_policy,
                                         SUNVulkanExecPolicy* reduce_exec_policy)
{
  if (stream_exec_policy == NULL || reduce_exec_policy == NULL)
  {
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  delete NVEC_VULKAN_CONTENT(x)->stream_exec_policy;
  delete NVEC_VULKAN_CONTENT(x)->reduce_exec_policy;

  NVEC_VULKAN_CONTENT(x)->stream_exec_policy = stream_exec_policy->clone();
  NVEC_VULKAN_CONTENT(x)->reduce_exec_policy = reduce_exec_policy->clone();
  return SUN_SUCCESS;
}

N_Vector N_VCloneEmpty_Vulkan(N_Vector w)
{
  N_Vector v = N_VNewEmpty_Vulkan(w->sunctx);
  if (v == NULL) { return NULL; }

  NVEC_VULKAN_LENGTH(v)           = NVEC_VULKAN_LENGTH(w);
  NVEC_VULKAN_PRIVATE(v)->manager = NVEC_VULKAN_PRIVATE(w)->manager;

  return v;
}

N_Vector N_VClone_Vulkan(N_Vector w)
{
  N_Vector v = N_VCloneEmpty_Vulkan(w);
  if (v == NULL) { return NULL; }

  // Deep copy: if source is a pointer, we must copy the data to v's vector
  auto* priv_w = NVEC_VULKAN_PRIVATE(w);
  if (std::holds_alternative<std::vector<sunrealtype>>(priv_w->host_data))
  {
    // Source is a vector - variant assignment does deep copy
    NVEC_VULKAN_PRIVATE(v)->host_data = priv_w->host_data;
  }
  else
  {
    // Source is a pointer - allocate vector and copy data
    NVEC_VULKAN_PRIVATE(v)->host_data =
      std::vector<sunrealtype>(NVEC_VULKAN_LENGTH(w));
    const auto hw = HostData(w);
    auto hv       = HostData(v);
    std::copy(hw.begin(), hw.end(), hv.begin());
  }
  MarkDeviceNeedsUpdate(v);

  return v;
}

void N_VDestroy_Vulkan(N_Vector v)
{
  if (v == NULL) { return; }
  if (v->content)
  {
    delete NVEC_VULKAN_PRIVATE(v);

    delete NVEC_VULKAN_CONTENT(v)->stream_exec_policy;
    delete NVEC_VULKAN_CONTENT(v)->reduce_exec_policy;

    free(v->content);
  }
  N_VFreeEmpty(v);
}

void N_VSpace_Vulkan(N_Vector v, sunindextype* lrw, sunindextype* liw)
{
  *lrw = NVEC_VULKAN_LENGTH(v);
  *liw = 0;
}

// ---------------------------------------------------------------------------
// Core vector operations
// ---------------------------------------------------------------------------

void N_VLinearSum_Vulkan(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y,
                         N_Vector z)
{
  DispatchElementwise(ElementwiseOp::LinearSum, a, b, x, y, z);
}

void N_VConst_Vulkan(sunrealtype c, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::Const, c, 0.0, z, nullptr, z);
}

void N_VProd_Vulkan(N_Vector x, N_Vector y, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::Prod, 0.0, 0.0, x, y, z);
}

void N_VDiv_Vulkan(N_Vector x, N_Vector y, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::Div, 0.0, 0.0, x, y, z);
}

void N_VScale_Vulkan(sunrealtype c, N_Vector x, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::Scale, c, 0.0, x, nullptr, z);
}

void N_VAbs_Vulkan(N_Vector x, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::Abs, 0.0, 0.0, x, nullptr, z);
}

void N_VInv_Vulkan(N_Vector x, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::Inv, 0.0, 0.0, x, nullptr, z);
}

void N_VAddConst_Vulkan(N_Vector x, sunrealtype b, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::AddConst, b, 0.0, x, nullptr, z);
}

sunrealtype N_VDotProd_Vulkan(N_Vector x, N_Vector y)
{
  return DispatchDotProdReduction(x, y);
}

sunrealtype N_VMaxNorm_Vulkan(N_Vector x)
{
  return DispatchMaxNormReduction(x);
}

sunrealtype N_VWrmsNorm_Vulkan(N_Vector x, N_Vector w)
{
  sunrealtype sum = DispatchWSqrSumReduction(x, w, nullptr, false);
  return std::sqrt(sum / NVEC_VULKAN_LENGTH(x));
}

sunrealtype N_VWrmsNormMask_Vulkan(N_Vector x, N_Vector w, N_Vector id)
{
  sunrealtype sum = DispatchWSqrSumReduction(x, w, id, true);
  return std::sqrt(sum / NVEC_VULKAN_LENGTH(x));
}

sunrealtype N_VMin_Vulkan(N_Vector x)
{
  N_VCopyFromDevice_Vulkan(x);
  const auto hx = HostData(x);
  return *std::min_element(hx.begin(), hx.end());
}

sunrealtype N_VWL2Norm_Vulkan(N_Vector x, N_Vector w)
{
  sunrealtype sum = DispatchWSqrSumReduction(x, w, nullptr, false);
  return std::sqrt(sum);
}

sunrealtype N_VL1Norm_Vulkan(N_Vector x)
{
  N_VCopyFromDevice_Vulkan(x);
  const auto hx   = HostData(x);
  sunrealtype sum = ZERO;
  for (auto v : hx) sum += std::abs(v);
  return sum;
}

void N_VCompare_Vulkan(sunrealtype c, N_Vector x, N_Vector z)
{
  DispatchElementwise(ElementwiseOp::Compare, c, 0.0, x, nullptr, z);
}

sunbooleantype N_VInvTest_Vulkan(N_Vector x, N_Vector z)
{
  EnsureHostDataLength(z, NVEC_VULKAN_LENGTH(x));
  return DispatchInvTest(x, z);
}

sunbooleantype N_VConstrMask_Vulkan(N_Vector c, N_Vector x, N_Vector m)
{
  EnsureHostDataLength(m, NVEC_VULKAN_LENGTH(x));
  return DispatchConstrMask(c, x, m);
}

sunrealtype N_VMinQuotient_Vulkan(N_Vector num, N_Vector denom)
{
  return DispatchMinQuotientReduction(num, denom);
}

// ---------------------------------------------------------------------------
// Fused ops
// ---------------------------------------------------------------------------

SUNErrCode N_VLinearCombination_Vulkan(int nvec, sunrealtype* c, N_Vector* X,
                                       N_Vector Z)
{
  if (nvec <= 0 || c == NULL || X == NULL) { return SUN_ERR_ARG_OUTOFRANGE; }

  // nvec == 1: should have called N_VScale
  if (nvec == 1)
  {
    N_VScale_Vulkan(c[0], X[0], Z);
    return SUN_SUCCESS;
  }

  // nvec == 2: should have called N_VLinearSum
  if (nvec == 2)
  {
    N_VLinearSum_Vulkan(c[0], X[0], c[1], X[1], Z);
    return SUN_SUCCESS;
  }

  // For nvec > 2, use GPU-accelerated pairwise linear sums
  // Z = c[0]*X[0] + c[1]*X[1]
  N_VLinearSum_Vulkan(c[0], X[0], c[1], X[1], Z);

  // Z += c[j]*X[j] for j = 2, ..., nvec-1
  for (int j = 2; j < nvec; j++) { N_VLinearSum_Vulkan(ONE, Z, c[j], X[j], Z); }

  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMulti_Vulkan(int nvec, sunrealtype* c, N_Vector X,
                                   N_Vector* Y, N_Vector* Z)
{
  if (nvec <= 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  // Use GPU-accelerated scale-add for each vector pair
  for (int j = 0; j < nvec; j++)
  {
    EnsureHostDataLength(Z[j], NVEC_VULKAN_LENGTH(X));
    DispatchScaleAdd(c[j], X, Y[j], Z[j]);
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VDotProdMulti_Vulkan(int nvec, N_Vector x, N_Vector* Y,
                                  sunrealtype* dotprods)
{
  // Use GPU-accelerated dot product for each vector pair
  for (int j = 0; j < nvec; j++)
  {
    dotprods[j] = DispatchDotProdReduction(x, Y[j]);
  }
  return SUN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Vector array operations
// ---------------------------------------------------------------------------

SUNErrCode N_VLinearSumVectorArray_Vulkan(int nvec, sunrealtype a, N_Vector* X,
                                          sunrealtype b, N_Vector* Y, N_Vector* Z)
{
  for (int j = 0; j < nvec; j++)
  {
    N_VLinearSum_Vulkan(a, X[j], b, Y[j], Z[j]);
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VScaleVectorArray_Vulkan(int nvec, sunrealtype* c, N_Vector* X,
                                      N_Vector* Z)
{
  for (int j = 0; j < nvec; j++) { N_VScale_Vulkan(c[j], X[j], Z[j]); }
  return SUN_SUCCESS;
}

SUNErrCode N_VConstVectorArray_Vulkan(int nvec, sunrealtype c, N_Vector* Z)
{
  for (int j = 0; j < nvec; j++) { N_VConst_Vulkan(c, Z[j]); }
  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMultiVectorArray_Vulkan(int nvec, int nsum,
                                              sunrealtype* a, N_Vector* X,
                                              N_Vector** Y, N_Vector** Z)
{
  if (nvec < 1 || nsum < 1) { return SUN_ERR_ARG_OUTOFRANGE; }

  // Special case for nvec == 1
  if (nvec == 1)
  {
    // Should have called N_VLinearSum
    if (nsum == 1)
    {
      N_VLinearSum_Vulkan(a[0], X[0], ONE, Y[0][0], Z[0][0]);
      return SUN_SUCCESS;
    }

    // Should have called N_VScaleAddMulti
    // Build temporary arrays: YY[j] = Y[j][0], ZZ[j] = Z[j][0]
    std::vector<N_Vector> YY(nsum), ZZ(nsum);
    for (int j = 0; j < nsum; j++)
    {
      YY[j] = Y[j][0];
      ZZ[j] = Z[j][0];
    }
    return N_VScaleAddMulti_Vulkan(nsum, a, X[0], YY.data(), ZZ.data());
  }

  // Special case for nsum == 1: should have called N_VLinearSumVectorArray
  if (nsum == 1)
  {
    return N_VLinearSumVectorArray_Vulkan(nvec, a[0], X, ONE, Y[0], Z[0]);
  }

  // General case: compute multiple linear sums
  sunindextype N = NVEC_VULKAN_LENGTH(X[0]);

  // Y[i][j] += a[i] * x[j] when Y == Z (in-place)
  if (Y == Z)
  {
    for (int i = 0; i < nvec; i++)
    {
      N_VCopyFromDevice_Vulkan(X[i]);
      const auto hx = HostData(X[i]);
      for (int j = 0; j < nsum; j++)
      {
        N_VCopyFromDevice_Vulkan(Y[j][i]);
        EnsureHostDataLength(Y[j][i], N);
        auto hy = HostData(Y[j][i]);
        for (sunindextype k = 0; k < N; k++) { hy[k] += a[j] * hx[k]; }
        MarkDeviceNeedsUpdate(Y[j][i]);
      }
    }
    return SUN_SUCCESS;
  }

  // Z[i][j] = a[i] * x[j] + y[i][j]
  for (int i = 0; i < nvec; i++)
  {
    N_VCopyFromDevice_Vulkan(X[i]);
    const auto hx = HostData(X[i]);
    for (int j = 0; j < nsum; j++)
    {
      N_VCopyFromDevice_Vulkan(Y[j][i]);
      EnsureHostDataLength(Z[j][i], N);
      const auto hy = HostData(Y[j][i]);
      auto hz       = HostData(Z[j][i]);
      for (sunindextype k = 0; k < N; k++) { hz[k] = a[j] * hx[k] + hy[k]; }
      MarkDeviceNeedsUpdate(Z[j][i]);
    }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VLinearCombinationVectorArray_Vulkan(int nvec, int nsum,
                                                  sunrealtype* c, N_Vector** X,
                                                  N_Vector* Z)
{
  if (nvec < 1 || nsum < 1) { return SUN_ERR_ARG_OUTOFRANGE; }

  // Special case for nvec == 1
  if (nvec == 1)
  {
    // Should have called N_VScale
    if (nsum == 1)
    {
      N_VScale_Vulkan(c[0], X[0][0], Z[0]);
      return SUN_SUCCESS;
    }

    // Should have called N_VLinearSum
    if (nsum == 2)
    {
      N_VLinearSum_Vulkan(c[0], X[0][0], c[1], X[1][0], Z[0]);
      return SUN_SUCCESS;
    }

    // Should have called N_VLinearCombination
    // Build temporary array: Y[i] = X[i][0]
    std::vector<N_Vector> Y(nsum);
    for (int i = 0; i < nsum; i++) { Y[i] = X[i][0]; }
    return N_VLinearCombination_Vulkan(nsum, c, Y.data(), Z[0]);
  }

  // Special case for nsum == 1: should have called N_VScaleVectorArray
  if (nsum == 1)
  {
    std::vector<sunrealtype> ctmp(nvec, c[0]);
    return N_VScaleVectorArray_Vulkan(nvec, ctmp.data(), X[0], Z);
  }

  // Special case for nsum == 2: should have called N_VLinearSumVectorArray
  if (nsum == 2)
  {
    return N_VLinearSumVectorArray_Vulkan(nvec, c[0], X[0], c[1], X[1], Z);
  }

  // General case: compute linear combination
  sunindextype N = NVEC_VULKAN_LENGTH(Z[0]);

  // X[0][j] += c[i]*X[i][j], i = 1,...,nsum-1 when X[0] == Z and c[0] == 1
  if ((X[0] == Z) && (c[0] == ONE))
  {
    for (int j = 0; j < nvec; j++)
    {
      N_VCopyFromDevice_Vulkan(Z[j]);
      EnsureHostDataLength(Z[j], N);
      auto hz = HostData(Z[j]);
      for (int i = 1; i < nsum; i++)
      {
        N_VCopyFromDevice_Vulkan(X[i][j]);
        const auto hx = HostData(X[i][j]);
        for (sunindextype k = 0; k < N; k++) { hz[k] += c[i] * hx[k]; }
      }
      MarkDeviceNeedsUpdate(Z[j]);
    }
    return SUN_SUCCESS;
  }

  // X[0][j] = c[0] * X[0][j] + sum{ c[i] * X[i][j] }, i = 1,...,nsum-1
  if (X[0] == Z)
  {
    for (int j = 0; j < nvec; j++)
    {
      N_VCopyFromDevice_Vulkan(Z[j]);
      EnsureHostDataLength(Z[j], N);
      auto hz = HostData(Z[j]);
      for (sunindextype k = 0; k < N; k++) { hz[k] *= c[0]; }
      for (int i = 1; i < nsum; i++)
      {
        N_VCopyFromDevice_Vulkan(X[i][j]);
        const auto hx = HostData(X[i][j]);
        for (sunindextype k = 0; k < N; k++) { hz[k] += c[i] * hx[k]; }
      }
      MarkDeviceNeedsUpdate(Z[j]);
    }
    return SUN_SUCCESS;
  }

  // Z[j] = sum{ c[i] * X[i][j] }, i = 0,...,nsum-1
  for (int j = 0; j < nvec; j++)
  {
    N_VCopyFromDevice_Vulkan(X[0][j]);
    EnsureHostDataLength(Z[j], N);
    const auto hx0 = HostData(X[0][j]);
    auto hz        = HostData(Z[j]);
    for (sunindextype k = 0; k < N; k++) { hz[k] = c[0] * hx0[k]; }
    for (int i = 1; i < nsum; i++)
    {
      N_VCopyFromDevice_Vulkan(X[i][j]);
      const auto hx = HostData(X[i][j]);
      for (sunindextype k = 0; k < N; k++) { hz[k] += c[i] * hx[k]; }
    }
    MarkDeviceNeedsUpdate(Z[j]);
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VWrmsNormVectorArray_Vulkan(int nvec, N_Vector* X, N_Vector* W,
                                         sunrealtype* nrm)
{
  for (int j = 0; j < nvec; j++) { nrm[j] = N_VWrmsNorm_Vulkan(X[j], W[j]); }
  return SUN_SUCCESS;
}

SUNErrCode N_VWrmsNormMaskVectorArray_Vulkan(int nvec, N_Vector* X, N_Vector* W,
                                             N_Vector id, sunrealtype* nrm)
{
  for (int j = 0; j < nvec; j++)
  {
    nrm[j] = N_VWrmsNormMask_Vulkan(X[j], W[j], id);
  }
  return SUN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Local reductions
// ---------------------------------------------------------------------------

sunrealtype N_VWSqrSumLocal_Vulkan(N_Vector x, N_Vector w)
{
  return DispatchWSqrSumReduction(x, w, nullptr, false);
}

sunrealtype N_VWSqrSumMaskLocal_Vulkan(N_Vector x, N_Vector w, N_Vector id)
{
  return DispatchWSqrSumReduction(x, w, id, true);
}

// ---------------------------------------------------------------------------
// XBraid buffer ops (host copies)
// ---------------------------------------------------------------------------

SUNErrCode N_VBufSize_Vulkan(N_Vector x, sunindextype* size)
{
  *size = NVEC_VULKAN_LENGTH(x) * sizeof(sunrealtype);
  return SUN_SUCCESS;
}

SUNErrCode N_VBufPack_Vulkan(N_Vector x, void* buf)
{
  N_VCopyFromDevice_Vulkan(x);
  const auto hx = HostData(x);
  std::memcpy(buf, hx.data(), hx.size() * sizeof(sunrealtype));
  return SUN_SUCCESS;
}

SUNErrCode N_VBufUnpack_Vulkan(N_Vector x, void* buf)
{
  EnsureHostDataLength(x, NVEC_VULKAN_LENGTH(x));
  auto hx = HostData(x);
  std::memcpy(hx.data(), buf, hx.size() * sizeof(sunrealtype));
  MarkDeviceNeedsUpdate(x);

  return SUN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Debug print
// ---------------------------------------------------------------------------

void N_VPrint_Vulkan(N_Vector v) { N_VPrintFile_Vulkan(v, stdout); }

void N_VPrintFile_Vulkan(N_Vector v, FILE* outfile)
{
  N_VCopyFromDevice_Vulkan(v);
  const auto hv = HostData(v);
  for (auto val : hv) { fprintf(outfile, "%g ", val); }
  fprintf(outfile, "\n");
}

// ---------------------------------------------------------------------------
// Enable fused ops (all implemented)
// ---------------------------------------------------------------------------

SUNErrCode N_VEnableFusedOps_Vulkan(N_Vector /*v*/, sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableLinearCombination_Vulkan(N_Vector /*v*/, sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableScaleAddMulti_Vulkan(N_Vector /*v*/, sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableDotProdMulti_Vulkan(N_Vector /*v*/, sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableLinearSumVectorArray_Vulkan(N_Vector /*v*/,
                                                sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableScaleVectorArray_Vulkan(N_Vector /*v*/, sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableConstVectorArray_Vulkan(N_Vector /*v*/, sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableWrmsNormVectorArray_Vulkan(N_Vector /*v*/,
                                               sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableWrmsNormMaskVectorArray_Vulkan(N_Vector /*v*/,
                                                   sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableScaleAddMultiVectorArray_Vulkan(N_Vector /*v*/,
                                                    sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

SUNErrCode N_VEnableLinearCombinationVectorArray_Vulkan(N_Vector /*v*/,
                                                        sunbooleantype /*tf*/)
{
  return SUN_SUCCESS;
}

} // extern "C"
