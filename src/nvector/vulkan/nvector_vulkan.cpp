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
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
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

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)

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
  if constexpr (std::is_same_v<T, sunrealtype>) { dst = src; }
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
      priv->host_data                = std::move(new_vec);
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

static std::vector<uint32_t> CompileSlangToSpirv(
  const std::string& source, const std::string& entry,
  const std::array<uint32_t, 3>& localSize)
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
      << "-DLOCAL_SIZE_X=" << localSize[0] << " "
      << "-DLOCAL_SIZE_Y=" << localSize[1] << " "
      << "-DLOCAL_SIZE_Z=" << localSize[2] << " "
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
  std::filesystem::remove(src);
  std::filesystem::remove(spv);

  // convert byte stream to uint32_t vector
  std::vector<uint32_t> words;
  words.resize(spirv_bytes.size() / sizeof(uint32_t));
  std::memcpy(words.data(), spirv_bytes.data(), spirv_bytes.size());
  return words;
}

static const std::string& ElementwiseShaderSource()
{
  static std::once_flag once;

  static std::string src;

  std::call_once(once,
                 []()
                 {
                   const char* real = (sizeof(ShaderFloat) == sizeof(double))
                                        ? "double"
                                        : "float";

                   src = fmt::format(R"(
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
                                     real, real, real, real, real, real, real,
                                     real, real, real);
                 });

  return src;
}

struct ShaderCache
{
  std::vector<uint32_t> elementwise_spv;
  std::mutex mutex;
};

static ShaderCache& GetShaderCache()
{
  static ShaderCache cache;
  return cache;
}

static const std::vector<uint32_t>& GetElementwiseSpirv()
{
  auto& cache = GetShaderCache();
  if (!cache.elementwise_spv.empty()) { return cache.elementwise_spv; }
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (cache.elementwise_spv.empty())
  {
    cache.elementwise_spv = CompileSlangToSpirv(ElementwiseShaderSource(),
                                                "main", {256, 1, 1});
  }
  return cache.elementwise_spv;
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

  auto spirv = GetElementwiseSpirv();
  auto seq   = privZ->manager->sequence();

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

  auto from_shader = privZ->device_data->vector<ShaderFloat>();
  FromShaderBuffer<ShaderFloat>(from_shader, HostData(z));
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

  NVEC_VULKAN_LENGTH(v)             = 0;
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
  else { priv->host_data = h_vdata; }

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
  N_Vector v = N_VNew_Vulkan(NVEC_VULKAN_LENGTH(w), w->sunctx);
  if (v == NULL) { return NULL; }

  NVEC_VULKAN_PRIVATE(v)->manager = NVEC_VULKAN_PRIVATE(w)->manager;

  return v;
}

N_Vector N_VClone_Vulkan(N_Vector w)
{
  N_Vector v = N_VCloneEmpty_Vulkan(w);
  if (v == NULL) { return NULL; }

  // Deep copy: if source is a pointer, we must copy the data to v's vector
  // (v was already created with a vector via N_VCloneEmpty_Vulkan)
  auto* priv_w = NVEC_VULKAN_PRIVATE(w);
  if (std::holds_alternative<std::vector<sunrealtype>>(priv_w->host_data))
  {
    // Source is a vector - variant assignment does deep copy
    NVEC_VULKAN_PRIVATE(v)->host_data = priv_w->host_data;
  }
  else
  {
    // Source is a pointer - copy data into v's existing vector
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
  N_VCopyFromDevice_Vulkan(x);
  N_VCopyFromDevice_Vulkan(y);
  const auto hx = HostData(x);
  const auto hy = HostData(y);
  return std::inner_product(hx.begin(), hx.end(), hy.begin(), ZERO);
}

sunrealtype N_VMaxNorm_Vulkan(N_Vector x)
{
  N_VCopyFromDevice_Vulkan(x);
  const auto hx = HostData(x);
  sunrealtype m = ZERO;
  for (auto v : hx) m = std::max(m, std::abs(v));
  return m;
}

sunrealtype N_VWrmsNorm_Vulkan(N_Vector x, N_Vector w)
{
  N_VCopyFromDevice_Vulkan(x);
  N_VCopyFromDevice_Vulkan(w);
  const auto hx   = HostData(x);
  const auto hw   = HostData(w);
  sunrealtype sum = ZERO;
  for (size_t i = 0; i < hx.size(); ++i)
  {
    sunrealtype v = hx[i] * hw[i];
    sum += v * v;
  }
  return std::sqrt(sum / hx.size());
}

sunrealtype N_VWrmsNormMask_Vulkan(N_Vector x, N_Vector w, N_Vector id)
{
  N_VCopyFromDevice_Vulkan(x);
  N_VCopyFromDevice_Vulkan(w);
  N_VCopyFromDevice_Vulkan(id);
  const auto hx    = HostData(x);
  const auto hw    = HostData(w);
  const auto hid   = HostData(id);
  sunrealtype sum  = ZERO;
  sunindextype cnt = 0;
  for (size_t i = 0; i < hx.size(); ++i)
  {
    if (hid[i] > ZERO)
    {
      sunrealtype v = hx[i] * hw[i];
      sum += v * v;
      cnt++;
    }
  }
  return cnt == 0 ? ZERO : std::sqrt(sum / cnt);
}

sunrealtype N_VMin_Vulkan(N_Vector x)
{
  N_VCopyFromDevice_Vulkan(x);
  const auto hx = HostData(x);
  return *std::min_element(hx.begin(), hx.end());
}

sunrealtype N_VWL2Norm_Vulkan(N_Vector x, N_Vector w)
{
  N_VCopyFromDevice_Vulkan(x);
  N_VCopyFromDevice_Vulkan(w);
  const auto hx   = HostData(x);
  const auto hw   = HostData(w);
  sunrealtype sum = ZERO;
  for (size_t i = 0; i < hx.size(); ++i) sum += hx[i] * hx[i] * hw[i] * hw[i];
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
  N_VCopyFromDevice_Vulkan(x);
  EnsureHostDataLength(z, NVEC_VULKAN_LENGTH(x));
  const auto hx         = HostData(x);
  auto hz               = HostData(z);
  sunbooleantype result = SUNTRUE;
  for (size_t i = 0; i < hx.size(); ++i)
  {
    if (hx[i] == ZERO)
    {
      hz[i]  = ZERO;
      result = SUNFALSE;
    }
    else { hz[i] = ONE / hx[i]; }
  }
  MarkDeviceNeedsUpdate(z);

  return result;
}

sunbooleantype N_VConstrMask_Vulkan(N_Vector c, N_Vector x, N_Vector m)
{
  N_VCopyFromDevice_Vulkan(c);
  N_VCopyFromDevice_Vulkan(x);
  EnsureHostDataLength(m, NVEC_VULKAN_LENGTH(x));
  auto hm       = HostData(m);
  const auto hc = HostData(c);
  const auto hx = HostData(x);

  // Initialize m to zero
  std::fill(hm.begin(), hm.end(), ZERO);

  sunbooleantype test = SUNTRUE;
  for (size_t i = 0; i < hx.size(); ++i)
  {
    if ((hc[i] == -ONE && hx[i] <= ZERO) || (hc[i] == ONE && hx[i] >= ZERO) ||
        (hc[i] == TWO && hx[i] == ZERO))
    {
      hm[i] = ONE;
      test  = SUNFALSE;
    }
  }
  MarkDeviceNeedsUpdate(m);

  return test;
}

sunrealtype N_VMinQuotient_Vulkan(N_Vector num, N_Vector denom)
{
  N_VCopyFromDevice_Vulkan(num);
  N_VCopyFromDevice_Vulkan(denom);
  const auto hn   = HostData(num);
  const auto hd   = HostData(denom);
  sunrealtype min = std::numeric_limits<sunrealtype>::infinity();
  for (size_t i = 0; i < hn.size(); ++i)
  {
    if (hd[i] != ZERO) { min = std::min(min, hn[i] / hd[i]); }
  }
  return min;
}

// ---------------------------------------------------------------------------
// Fused ops
// ---------------------------------------------------------------------------

SUNErrCode N_VLinearCombination_Vulkan(int nvec, sunrealtype* c, N_Vector* X,
                                       N_Vector Z)
{
  if (nvec <= 0 || c == NULL || X == NULL) { return SUN_ERR_ARG_OUTOFRANGE; }

  EnsureHostDataLength(Z, NVEC_VULKAN_LENGTH(Z));
  auto hz = HostData(Z);
  std::fill(hz.begin(), hz.end(), ZERO);
  for (int j = 0; j < nvec; j++)
  {
    N_VCopyFromDevice_Vulkan(X[j]);
    const auto hx = HostData(X[j]);
    for (size_t i = 0; i < hz.size(); ++i) hz[i] += c[j] * hx[i];
  }
  MarkDeviceNeedsUpdate(Z);

  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMulti_Vulkan(int nvec, sunrealtype* c, N_Vector X,
                                   N_Vector* Y, N_Vector* Z)
{
  if (nvec <= 0) { return SUN_ERR_ARG_OUTOFRANGE; }
  N_VCopyFromDevice_Vulkan(X);
  const auto hx = HostData(X);
  for (int j = 0; j < nvec; j++)
  {
    N_VCopyFromDevice_Vulkan(Y[j]);
    EnsureHostDataLength(Z[j], static_cast<sunindextype>(hx.size()));
    auto hz       = HostData(Z[j]);
    const auto hy = HostData(Y[j]);
    for (size_t i = 0; i < hx.size(); ++i) hz[i] = c[j] * hx[i] + hy[i];
    MarkDeviceNeedsUpdate(Z[j]);
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VDotProdMulti_Vulkan(int nvec, N_Vector x, N_Vector* Y,
                                  sunrealtype* dotprods)
{
  N_VCopyFromDevice_Vulkan(x);
  const auto hx = HostData(x);
  for (int j = 0; j < nvec; j++)
  {
    N_VCopyFromDevice_Vulkan(Y[j]);
    const auto hy = HostData(Y[j]);
    dotprods[j]   = std::inner_product(hx.begin(), hx.end(), hy.begin(), ZERO);
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
  for (int j = 0; j < nvec; j++)
  {
    N_VScaleAddMulti_Vulkan(nsum, a, X[j], Y[j], Z[j]);
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VLinearCombinationVectorArray_Vulkan(int nvec, int nsum,
                                                  sunrealtype* c, N_Vector** X,
                                                  N_Vector* Z)
{
  for (int j = 0; j < nvec; j++)
  {
    N_VLinearCombination_Vulkan(nsum, c, X[j], Z[j]);
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
  N_VCopyFromDevice_Vulkan(x);
  N_VCopyFromDevice_Vulkan(w);
  const auto hx   = HostData(x);
  const auto hw   = HostData(w);
  sunrealtype sum = ZERO;
  for (size_t i = 0; i < hx.size(); ++i) sum += hx[i] * hx[i] * hw[i] * hw[i];
  return sum;
}

sunrealtype N_VWSqrSumMaskLocal_Vulkan(N_Vector x, N_Vector w, N_Vector id)
{
  N_VCopyFromDevice_Vulkan(x);
  N_VCopyFromDevice_Vulkan(w);
  N_VCopyFromDevice_Vulkan(id);
  const auto hx   = HostData(x);
  const auto hw   = HostData(w);
  const auto hid  = HostData(id);
  sunrealtype sum = ZERO;
  for (size_t i = 0; i < hx.size(); ++i)
  {
    if (hid[i] > ZERO) sum += hx[i] * hx[i] * hw[i] * hw[i];
  }
  return sum;
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
