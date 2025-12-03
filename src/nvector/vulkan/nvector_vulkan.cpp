/*
 * -----------------------------------------------------------------
 * Vulkan NVECTOR implementation using Kompute and Slang (slangc).
 * NOTE: This backend mirrors the CUDA API but executes on Vulkan.
 * -----------------------------------------------------------------
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
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
#define NVEC_VULKAN_LENGTH(x) (NVEC_VULKAN_CONTENT(x)->length)
#define NVEC_VULKAN_MEMHELP(x) (NVEC_VULKAN_CONTENT(x)->mem_helper)
#define NVEC_VULKAN_PRIVATE(x) ((N_PrivateVectorContent_Vulkan)(NVEC_VULKAN_CONTENT(x)->priv))

struct _N_PrivateVectorContent_Vulkan
{
  std::shared_ptr<kp::Manager> manager;
  std::shared_ptr<kp::Tensor> device_tensor;
  std::vector<sunrealtype> host_buffer;
  bool host_dirty{true};
  bool device_dirty{false};
};

typedef struct _N_PrivateVectorContent_Vulkan* N_PrivateVectorContent_Vulkan;

static std::shared_ptr<kp::Manager> GetDefaultManager()
{
  static std::shared_ptr<kp::Manager> mgr = std::make_shared<kp::Manager>();
  return mgr;
}

static void RefreshHostMemoryView(N_Vector v)
{
  SUNMemory mem = NVEC_VULKAN_CONTENT(v)->host_data;
  if (mem == NULL)
  {
    mem = SUNMemoryNewEmpty(v->sunctx);
    NVEC_VULKAN_CONTENT(v)->host_data = mem;
  }
  mem->ptr   = NVEC_VULKAN_PRIVATE(v)->host_buffer.data();
  mem->bytes = NVEC_VULKAN_LENGTH(v) * sizeof(sunrealtype);
  mem->own   = SUNFALSE;
  mem->type  = SUNMEMTYPE_HOST;
}

static void RefreshDeviceMemoryView(N_Vector v)
{
  SUNMemory mem = NVEC_VULKAN_CONTENT(v)->device_data;
  if (mem == NULL)
  {
    mem = SUNMemoryNewEmpty(v->sunctx);
    NVEC_VULKAN_CONTENT(v)->device_data = mem;
  }
  mem->ptr   = (void*)NVEC_VULKAN_PRIVATE(v)->host_buffer.data();
  mem->bytes = NVEC_VULKAN_LENGTH(v) * sizeof(sunrealtype);
  mem->own   = SUNFALSE;
  mem->type  = SUNMEMTYPE_DEVICE;
}

static void MarkHostDirty(N_Vector v)
{
  NVEC_VULKAN_PRIVATE(v)->host_dirty   = true;
  NVEC_VULKAN_PRIVATE(v)->device_dirty = false;
}

static void MarkDeviceDirty(N_Vector v)
{
  NVEC_VULKAN_PRIVATE(v)->device_dirty = true;
  NVEC_VULKAN_PRIVATE(v)->host_dirty   = false;
}

static void EnsureTensor(N_Vector v)
{
  auto priv = NVEC_VULKAN_PRIVATE(v);
  if (!priv->device_tensor)
  {
    priv->device_tensor = priv->manager->tensor(
      priv->host_buffer.data(),
      static_cast<uint32_t>(NVEC_VULKAN_LENGTH(v)),
      sizeof(sunrealtype),
      kp::Memory::dataType<sunrealtype>(),
      kp::Memory::MemoryTypes::eDevice);
  }
}

static void CopyHostToDevice(N_Vector v)
{
  auto priv = NVEC_VULKAN_PRIVATE(v);
  if (!priv->host_dirty) { return; }
  EnsureTensor(v);
  priv->device_tensor->setData(priv->host_buffer);
  auto seq = priv->manager->sequence();
  seq->record<kp::OpSyncDevice>(
    {std::static_pointer_cast<kp::Memory>(priv->device_tensor)});
  seq->eval();
  priv->host_dirty   = false;
  priv->device_dirty = false;
  RefreshDeviceMemoryView(v);
}

static void CopyDeviceToHost(N_Vector v)
{
  auto priv = NVEC_VULKAN_PRIVATE(v);
  if (!priv->device_dirty && !priv->host_dirty) { return; }
  EnsureTensor(v);
  auto seq = priv->manager->sequence();
  seq->record<kp::OpSyncLocal>(
    {std::static_pointer_cast<kp::Memory>(priv->device_tensor)});
  seq->eval();
  priv->host_buffer = priv->device_tensor->vector<sunrealtype>();
  priv->host_dirty   = false;
  priv->device_dirty = false;
  RefreshHostMemoryView(v);
  RefreshDeviceMemoryView(v);
}

// ---------------------------------------------------------------------------
// Slang compilation helpers
// ---------------------------------------------------------------------------

static std::vector<uint32_t> CompileSlangToSpirv(const std::string& source,
                                                 const std::string& entry,
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
    throw std::runtime_error("slangc failed when compiling Vulkan NVECTOR shaders");
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
  static const std::string src = R"(
        // Elementwise operations controlled via push constants.
        struct Params {
            uint op;
            float a;
            float b;
            uint n;
        };

        [[vk::push_constant]]
        ConstantBuffer<Params> params;

        [[vk::binding(0,0)]] RWStructuredBuffer<float> X;
        [[vk::binding(1,0)]] RWStructuredBuffer<float> Y;
        [[vk::binding(2,0)]] RWStructuredBuffer<float> Z;

        [numthreads(LOCAL_SIZE_X, LOCAL_SIZE_Y, LOCAL_SIZE_Z)]
        void main(uint3 dtid : SV_DispatchThreadID)
        {
            uint i = dtid.x;
            if (i >= params.n) return;

            switch (params.op)
            {
            case 0: Z[i] = params.a * X[i] + params.b * Y[i]; break; // linear sum
            case 1: Z[i] = params.a; break; // const
            case 2: Z[i] = X[i] * Y[i]; break; // prod
            case 3: Z[i] = X[i] / Y[i]; break; // div
            case 4: Z[i] = params.a * X[i]; break; // scale
            case 5: Z[i] = abs(X[i]); break; // abs
            case 6: Z[i] = 1.0 / X[i]; break; // inv
            case 7: Z[i] = X[i] + params.a; break; // add const
            case 8: Z[i] = (abs(X[i]) >= params.a) ? 1.0 : 0.0; break; // compare
            default: Z[i] = X[i]; break;
            }
        }
    )";
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

  CopyHostToDevice(x);
  if (y) { CopyHostToDevice(y); }
  EnsureTensor(z);

  auto spirv = GetElementwiseSpirv();
  auto seq   = privZ->manager->sequence();

  // Sync inputs and outputs
  std::vector<std::shared_ptr<kp::Memory>> memObjects;
  memObjects.reserve(y ? 3 : 2);
  memObjects.push_back(privX->device_tensor);
  if (y) memObjects.push_back(privY->device_tensor);
  memObjects.push_back(privZ->device_tensor);
  seq->record<kp::OpSyncDevice>(memObjects);

  struct Push { uint32_t op; float a; float b; uint32_t n; };
  Push push{static_cast<uint32_t>(op), static_cast<float>(a), static_cast<float>(b),
            static_cast<uint32_t>(NVEC_VULKAN_LENGTH(z))};

  std::vector<uint8_t> pushConstants(sizeof(Push));
  std::memcpy(pushConstants.data(), &push, sizeof(Push));

  auto stream_policy = NVEC_VULKAN_CONTENT(z)->stream_exec_policy;
  auto algo          = privZ->manager->algorithm(
                                        memObjects,
                                        spirv,
                                        {stream_policy->gridSize(push.n), 1, 1},
                                        std::vector<uint32_t>{},
                                        pushConstants);

  seq->record<kp::OpAlgoDispatch>(algo);
  seq->record<kp::OpSyncLocal>({std::static_pointer_cast<kp::Memory>(privZ->device_tensor)});
  seq->eval();

  privZ->host_buffer = privZ->device_tensor->vector<sunrealtype>();
  privZ->host_dirty   = false;
  privZ->device_dirty = false;
  RefreshHostMemoryView(z);
  RefreshDeviceMemoryView(z);
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

  NVEC_VULKAN_CONTENT(v)->priv = malloc(sizeof(_N_PrivateVectorContent_Vulkan));
  if (NVEC_VULKAN_CONTENT(v)->priv == NULL)
  {
    N_VDestroy(v);
    return NULL;
  }

  NVEC_VULKAN_CONTENT(v)->length             = 0;
  NVEC_VULKAN_CONTENT(v)->host_data          = NULL;
  NVEC_VULKAN_CONTENT(v)->device_data        = NULL;
  NVEC_VULKAN_CONTENT(v)->mem_helper         = NULL;
  NVEC_VULKAN_CONTENT(v)->own_helper         = SUNFALSE;
  NVEC_VULKAN_CONTENT(v)->stream_exec_policy = new ExecPolicy(256);
  NVEC_VULKAN_CONTENT(v)->reduce_exec_policy = new AtomicReduceExecPolicy(256);

  auto priv        = NVEC_VULKAN_PRIVATE(v);
  priv->manager    = GetDefaultManager();
  priv->device_tensor.reset();
  priv->host_buffer.clear();
  priv->host_dirty   = true;
  priv->device_dirty = false;

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
  v->ops->nvscaleaddmulti = N_VScaleAddMulti_Vulkan;
  v->ops->nvdotprodmulti  = N_VDotProdMulti_Vulkan;

  // vector array operations
  v->ops->nvlinearsumvectorarray         = N_VLinearSumVectorArray_Vulkan;
  v->ops->nvscalevectorarray             = N_VScaleVectorArray_Vulkan;
  v->ops->nvconstvectorarray             = N_VConstVectorArray_Vulkan;
  v->ops->nvscaleaddmultivectorarray     = N_VScaleAddMultiVectorArray_Vulkan;
  v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_Vulkan;
  v->ops->nvwrmsnormvectorarray          = N_VWrmsNormVectorArray_Vulkan;
  v->ops->nvwrmsnormmaskvectorarray      = N_VWrmsNormMaskVectorArray_Vulkan;

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

  NVEC_VULKAN_CONTENT(v)->length = length;
  NVEC_VULKAN_PRIVATE(v)->host_buffer.resize(length, ZERO);
  NVEC_VULKAN_CONTENT(v)->mem_helper = SUNMemoryHelper_Vulkan(sunctx);
  NVEC_VULKAN_CONTENT(v)->own_helper = SUNTRUE;
  RefreshHostMemoryView(v);
  RefreshDeviceMemoryView(v);
  return v;
}

N_Vector N_VNewWithMemHelp_Vulkan(sunindextype length, SUNMemoryHelper helper,
                                  SUNContext sunctx)
{
  N_Vector v = N_VNew_Vulkan(length, sunctx);
  if (v == NULL) { return NULL; }

  NVEC_VULKAN_CONTENT(v)->mem_helper = helper;
  NVEC_VULKAN_CONTENT(v)->own_helper = SUNFALSE;
  RefreshHostMemoryView(v);
  RefreshDeviceMemoryView(v);
  return v;
}

N_Vector N_VMake_Vulkan(sunindextype length, sunrealtype* h_vdata,
                        sunrealtype* d_vdata, SUNContext sunctx)
{
  N_Vector v = N_VNew_Vulkan(length, sunctx);
  if (v == NULL) { return NULL; }

  // Wrap user-provided buffers
  NVEC_VULKAN_PRIVATE(v)->host_buffer.assign(h_vdata, h_vdata + length);
  if (d_vdata != NULL)
  {
    NVEC_VULKAN_PRIVATE(v)->device_tensor =
      NVEC_VULKAN_PRIVATE(v)->manager->tensor(
        d_vdata,
        static_cast<uint32_t>(length),
        sizeof(sunrealtype),
        kp::Memory::dataType<sunrealtype>(),
        kp::Memory::MemoryTypes::eDevice);
    NVEC_VULKAN_PRIVATE(v)->device_dirty = false;
  }
  NVEC_VULKAN_PRIVATE(v)->host_dirty = false;
  RefreshHostMemoryView(v);
  RefreshDeviceMemoryView(v);
  return v;
}

void N_VSetHostArrayPointer_Vulkan(sunrealtype* h_vdata, N_Vector v)
{
  auto priv = NVEC_VULKAN_PRIVATE(v);
  priv->host_buffer.assign(h_vdata, h_vdata + NVEC_VULKAN_LENGTH(v));
  priv->host_dirty   = true;
  priv->device_dirty = false;
  RefreshHostMemoryView(v);
}

void N_VSetDeviceArrayPointer_Vulkan(sunrealtype* d_vdata, N_Vector v)
{
  auto priv = NVEC_VULKAN_PRIVATE(v);
  priv->device_tensor =
    priv->manager->tensor(d_vdata,
                          static_cast<uint32_t>(NVEC_VULKAN_LENGTH(v)),
                          sizeof(sunrealtype),
                          kp::Memory::dataType<sunrealtype>(),
                          kp::Memory::MemoryTypes::eDevice);
  priv->device_dirty = false;
  priv->host_dirty   = true;
  RefreshDeviceMemoryView(v);
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

void N_VCopyToDevice_Vulkan(N_Vector v) { CopyHostToDevice(v); }

void N_VCopyFromDevice_Vulkan(N_Vector v) { CopyDeviceToHost(v); }

N_Vector N_VCloneEmpty_Vulkan(N_Vector w)
{
  N_Vector v = N_VNewEmpty_Vulkan(w->sunctx);
  if (v == NULL) { return NULL; }

  NVEC_VULKAN_CONTENT(v)->length = NVEC_VULKAN_CONTENT(w)->length;
  NVEC_VULKAN_PRIVATE(v)->manager = NVEC_VULKAN_PRIVATE(w)->manager;
  RefreshHostMemoryView(v);
  RefreshDeviceMemoryView(v);
  return v;
}

N_Vector N_VClone_Vulkan(N_Vector w)
{
  N_Vector v = N_VCloneEmpty_Vulkan(w);
  if (v == NULL) { return NULL; }

  NVEC_VULKAN_PRIVATE(v)->host_buffer = NVEC_VULKAN_PRIVATE(w)->host_buffer;
  NVEC_VULKAN_PRIVATE(v)->host_dirty  = true;
  NVEC_VULKAN_PRIVATE(v)->device_dirty = false;
  RefreshHostMemoryView(v);
  RefreshDeviceMemoryView(v);

  return v;
}

void N_VDestroy_Vulkan(N_Vector v)
{
  if (v == NULL) { return; }
  if (v->content)
  {
    auto priv = NVEC_VULKAN_PRIVATE(v);
    priv->device_tensor.reset();
    priv->host_buffer.clear();
    free(priv);

    delete NVEC_VULKAN_CONTENT(v)->stream_exec_policy;
    delete NVEC_VULKAN_CONTENT(v)->reduce_exec_policy;
    if (NVEC_VULKAN_CONTENT(v)->own_helper &&
        NVEC_VULKAN_CONTENT(v)->mem_helper != NULL)
    {
      SUNMemoryHelper_Destroy_Vulkan(NVEC_VULKAN_CONTENT(v)->mem_helper);
    }
    if (NVEC_VULKAN_CONTENT(v)->host_data) { free(NVEC_VULKAN_CONTENT(v)->host_data); }
    if (NVEC_VULKAN_CONTENT(v)->device_data)
    {
      free(NVEC_VULKAN_CONTENT(v)->device_data);
    }
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
  CopyDeviceToHost(x);
  CopyDeviceToHost(y);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  const auto& hy = NVEC_VULKAN_PRIVATE(y)->host_buffer;
  return std::inner_product(hx.begin(), hx.end(), hy.begin(), ZERO);
}

sunrealtype N_VMaxNorm_Vulkan(N_Vector x)
{
  CopyDeviceToHost(x);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  sunrealtype m  = ZERO;
  for (auto v : hx) m = std::max(m, std::abs(v));
  return m;
}

sunrealtype N_VWrmsNorm_Vulkan(N_Vector x, N_Vector w)
{
  CopyDeviceToHost(x);
  CopyDeviceToHost(w);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  const auto& hw = NVEC_VULKAN_PRIVATE(w)->host_buffer;
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
  CopyDeviceToHost(x);
  CopyDeviceToHost(w);
  CopyDeviceToHost(id);
  const auto& hx  = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  const auto& hw  = NVEC_VULKAN_PRIVATE(w)->host_buffer;
  const auto& hid = NVEC_VULKAN_PRIVATE(id)->host_buffer;
  sunrealtype sum = ZERO;
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
  CopyDeviceToHost(x);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  return *std::min_element(hx.begin(), hx.end());
}

sunrealtype N_VWL2Norm_Vulkan(N_Vector x, N_Vector w)
{
  CopyDeviceToHost(x);
  CopyDeviceToHost(w);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  const auto& hw = NVEC_VULKAN_PRIVATE(w)->host_buffer;
  sunrealtype sum = ZERO;
  for (size_t i = 0; i < hx.size(); ++i) sum += hx[i] * hx[i] * hw[i] * hw[i];
  return std::sqrt(sum);
}

sunrealtype N_VL1Norm_Vulkan(N_Vector x)
{
  CopyDeviceToHost(x);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
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
  CopyDeviceToHost(x);
  auto& hz = NVEC_VULKAN_PRIVATE(z)->host_buffer;
  hz.resize(NVEC_VULKAN_LENGTH(x));
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  sunbooleantype result = SUNTRUE;
  for (size_t i = 0; i < hx.size(); ++i)
  {
    if (hx[i] == ZERO)
    {
      hz[i] = ZERO;
      result = SUNFALSE;
    }
    else
    {
      hz[i] = ONE / hx[i];
    }
  }
  RefreshHostMemoryView(z);
  MarkDeviceDirty(z);
  return result;
}

sunbooleantype N_VConstrMask_Vulkan(N_Vector c, N_Vector x, N_Vector m)
{
  CopyDeviceToHost(c);
  CopyDeviceToHost(x);
  auto& hm = NVEC_VULKAN_PRIVATE(m)->host_buffer;
  hm.assign(NVEC_VULKAN_LENGTH(x), ZERO);
  const auto& hc = NVEC_VULKAN_PRIVATE(c)->host_buffer;
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;

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
  RefreshHostMemoryView(m);
  MarkDeviceDirty(m);
  return test;
}

sunrealtype N_VMinQuotient_Vulkan(N_Vector num, N_Vector denom)
{
  CopyDeviceToHost(num);
  CopyDeviceToHost(denom);
  const auto& hn = NVEC_VULKAN_PRIVATE(num)->host_buffer;
  const auto& hd = NVEC_VULKAN_PRIVATE(denom)->host_buffer;
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
  auto& hz = NVEC_VULKAN_PRIVATE(Z)->host_buffer;
  hz.assign(NVEC_VULKAN_LENGTH(Z), ZERO);
  for (int j = 0; j < nvec; j++)
  {
    CopyDeviceToHost(X[j]);
    const auto& hx = NVEC_VULKAN_PRIVATE(X[j])->host_buffer;
    for (size_t i = 0; i < hz.size(); ++i) hz[i] += c[j] * hx[i];
  }
  RefreshHostMemoryView(Z);
  MarkDeviceDirty(Z);
  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMulti_Vulkan(int nvec, sunrealtype* c, N_Vector X,
                                   N_Vector* Y, N_Vector* Z)
{
  if (nvec <= 0) { return SUN_ERR_ARG_OUTOFRANGE; }
  CopyDeviceToHost(X);
  const auto& hx = NVEC_VULKAN_PRIVATE(X)->host_buffer;
  for (int j = 0; j < nvec; j++)
  {
    CopyDeviceToHost(Y[j]);
    auto& hz = NVEC_VULKAN_PRIVATE(Z[j])->host_buffer;
    hz.resize(hx.size());
    const auto& hy = NVEC_VULKAN_PRIVATE(Y[j])->host_buffer;
    for (size_t i = 0; i < hx.size(); ++i) hz[i] = c[j] * hx[i] + hy[i];
    RefreshHostMemoryView(Z[j]);
    MarkDeviceDirty(Z[j]);
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VDotProdMulti_Vulkan(int nvec, N_Vector x, N_Vector* Y,
                                  sunrealtype* dotprods)
{
  CopyDeviceToHost(x);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  for (int j = 0; j < nvec; j++)
  {
    CopyDeviceToHost(Y[j]);
    const auto& hy = NVEC_VULKAN_PRIVATE(Y[j])->host_buffer;
    dotprods[j]     = std::inner_product(hx.begin(), hx.end(), hy.begin(), ZERO);
  }
  return SUN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Vector array operations
// ---------------------------------------------------------------------------

SUNErrCode N_VLinearSumVectorArray_Vulkan(int nvec, sunrealtype a, N_Vector* X,
                                          sunrealtype b, N_Vector* Y, N_Vector* Z)
{
  for (int j = 0; j < nvec; j++) { N_VLinearSum_Vulkan(a, X[j], b, Y[j], Z[j]); }
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

SUNErrCode N_VScaleAddMultiVectorArray_Vulkan(int nvec, int nsum, sunrealtype* a,
                                              N_Vector* X, N_Vector** Y,
                                              N_Vector** Z)
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
  CopyDeviceToHost(x);
  CopyDeviceToHost(w);
  const auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  const auto& hw = NVEC_VULKAN_PRIVATE(w)->host_buffer;
  sunrealtype sum = ZERO;
  for (size_t i = 0; i < hx.size(); ++i) sum += hx[i] * hx[i] * hw[i] * hw[i];
  return sum;
}

sunrealtype N_VWSqrSumMaskLocal_Vulkan(N_Vector x, N_Vector w, N_Vector id)
{
  CopyDeviceToHost(x);
  CopyDeviceToHost(w);
  CopyDeviceToHost(id);
  const auto& hx  = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  const auto& hw  = NVEC_VULKAN_PRIVATE(w)->host_buffer;
  const auto& hid = NVEC_VULKAN_PRIVATE(id)->host_buffer;
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
  CopyDeviceToHost(x);
  std::memcpy(buf, NVEC_VULKAN_PRIVATE(x)->host_buffer.data(),
              NVEC_VULKAN_LENGTH(x) * sizeof(sunrealtype));
  return SUN_SUCCESS;
}

SUNErrCode N_VBufUnpack_Vulkan(N_Vector x, void* buf)
{
  auto& hx = NVEC_VULKAN_PRIVATE(x)->host_buffer;
  hx.resize(NVEC_VULKAN_LENGTH(x));
  std::memcpy(hx.data(), buf, hx.size() * sizeof(sunrealtype));
  RefreshHostMemoryView(x);
  MarkDeviceDirty(x);
  return SUN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Debug print
// ---------------------------------------------------------------------------

void N_VPrint_Vulkan(N_Vector v)
{
  N_VPrintFile_Vulkan(v, stdout);
}

void N_VPrintFile_Vulkan(N_Vector v, FILE* outfile)
{
  CopyDeviceToHost(v);
  const auto& hv = NVEC_VULKAN_PRIVATE(v)->host_buffer;
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
