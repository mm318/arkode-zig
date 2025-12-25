/* -----------------------------------------------------------------
 * Vulkan implementation of the NVECTOR module (Kompute backend).
 * -----------------------------------------------------------------*/

#ifndef _NVECTOR_VULKAN_H
#define _NVECTOR_VULKAN_H

#include <kompute/Kompute.hpp>
#include <stdio.h>
#include <sundials/sundials_config.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_vulkan_policies.hpp>
#include <sunmemory/sunmemory_vulkan.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

struct _N_VectorContent_Vulkan
{
  sunindextype length;
  SUNVulkanExecPolicy* stream_exec_policy;
  SUNVulkanExecPolicy* reduce_exec_policy;
  void* priv; /* 'private' data */
};

typedef struct _N_VectorContent_Vulkan* N_VectorContent_Vulkan;

SUNDIALS_EXPORT N_Vector N_VNewEmpty_Vulkan(SUNContext sunctx);
SUNDIALS_EXPORT N_Vector N_VNew_Vulkan(sunindextype length, SUNContext sunctx);
SUNDIALS_EXPORT N_Vector N_VMake_Vulkan(sunindextype length, sunrealtype* h_vdata,
                                        sunrealtype* d_vdata, SUNContext sunctx);

SUNDIALS_EXPORT void N_VSetHostArrayPointer_Vulkan(sunrealtype* h_vdata,
                                                   N_Vector v);

SUNDIALS_EXPORT SUNErrCode N_VSetKernelExecPolicy_Vulkan(
  N_Vector x, SUNVulkanExecPolicy* stream_exec_policy,
  SUNVulkanExecPolicy* reduce_exec_policy);

SUNDIALS_EXPORT void N_VCopyToDevice_Vulkan(N_Vector v);
SUNDIALS_EXPORT void N_VCopyFromDevice_Vulkan(N_Vector v);

static inline sunindextype N_VGetLength_Vulkan(N_Vector x)
{
  N_VectorContent_Vulkan content = (N_VectorContent_Vulkan)x->content;
  return content->length;
}

static inline N_Vector_ID N_VGetVectorID_Vulkan(N_Vector /*v*/)
{
  return SUNDIALS_NVEC_CUSTOM;
}

sunrealtype* N_VGetHostArrayPointer_Vulkan(N_Vector x);
sunrealtype* N_VGetDeviceArrayPointer_Vulkan(N_Vector x);

SUNDIALS_EXPORT N_Vector N_VCloneEmpty_Vulkan(N_Vector w);
SUNDIALS_EXPORT N_Vector N_VClone_Vulkan(N_Vector w);
SUNDIALS_EXPORT void N_VDestroy_Vulkan(N_Vector v);
SUNDIALS_EXPORT void N_VSpace_Vulkan(N_Vector v, sunindextype* lrw,
                                     sunindextype* liw);

/* standard vector operations */
SUNDIALS_EXPORT void N_VLinearSum_Vulkan(sunrealtype a, N_Vector x,
                                         sunrealtype b, N_Vector y, N_Vector z);
SUNDIALS_EXPORT void N_VConst_Vulkan(sunrealtype c, N_Vector z);
SUNDIALS_EXPORT void N_VProd_Vulkan(N_Vector x, N_Vector y, N_Vector z);
SUNDIALS_EXPORT void N_VDiv_Vulkan(N_Vector x, N_Vector y, N_Vector z);
SUNDIALS_EXPORT void N_VScale_Vulkan(sunrealtype c, N_Vector x, N_Vector z);
SUNDIALS_EXPORT void N_VAbs_Vulkan(N_Vector x, N_Vector z);
SUNDIALS_EXPORT void N_VInv_Vulkan(N_Vector x, N_Vector z);
SUNDIALS_EXPORT void N_VAddConst_Vulkan(N_Vector x, sunrealtype b, N_Vector z);
SUNDIALS_EXPORT sunrealtype N_VDotProd_Vulkan(N_Vector x, N_Vector y);
SUNDIALS_EXPORT sunrealtype N_VMaxNorm_Vulkan(N_Vector x);
SUNDIALS_EXPORT sunrealtype N_VWrmsNorm_Vulkan(N_Vector x, N_Vector w);
SUNDIALS_EXPORT sunrealtype N_VWrmsNormMask_Vulkan(N_Vector x, N_Vector w,
                                                   N_Vector id);
SUNDIALS_EXPORT sunrealtype N_VMin_Vulkan(N_Vector x);
SUNDIALS_EXPORT sunrealtype N_VWL2Norm_Vulkan(N_Vector x, N_Vector w);
SUNDIALS_EXPORT sunrealtype N_VL1Norm_Vulkan(N_Vector x);
SUNDIALS_EXPORT void N_VCompare_Vulkan(sunrealtype c, N_Vector x, N_Vector z);
SUNDIALS_EXPORT sunbooleantype N_VInvTest_Vulkan(N_Vector x, N_Vector z);
SUNDIALS_EXPORT sunbooleantype N_VConstrMask_Vulkan(N_Vector c, N_Vector x,
                                                    N_Vector m);
SUNDIALS_EXPORT sunrealtype N_VMinQuotient_Vulkan(N_Vector num, N_Vector denom);

/* fused vector operations */
SUNDIALS_EXPORT SUNErrCode N_VLinearCombination_Vulkan(int nvec, sunrealtype* c,
                                                       N_Vector* X, N_Vector Z);
SUNDIALS_EXPORT SUNErrCode N_VScaleAddMulti_Vulkan(int nvec, sunrealtype* c,
                                                   N_Vector X, N_Vector* Y,
                                                   N_Vector* Z);
SUNDIALS_EXPORT SUNErrCode N_VDotProdMulti_Vulkan(int nvec, N_Vector x,
                                                  N_Vector* Y,
                                                  sunrealtype* dotprods);

/* vector array operations */
SUNDIALS_EXPORT SUNErrCode N_VLinearSumVectorArray_Vulkan(
  int nvec, sunrealtype a, N_Vector* X, sunrealtype b, N_Vector* Y, N_Vector* Z);
SUNDIALS_EXPORT SUNErrCode N_VScaleVectorArray_Vulkan(int nvec, sunrealtype* c,
                                                      N_Vector* X, N_Vector* Z);
SUNDIALS_EXPORT SUNErrCode N_VConstVectorArray_Vulkan(int nvec, sunrealtype c,
                                                      N_Vector* Z);
SUNDIALS_EXPORT SUNErrCode N_VScaleAddMultiVectorArray_Vulkan(
  int nvec, int nsum, sunrealtype* a, N_Vector* X, N_Vector** Y, N_Vector** Z);
SUNDIALS_EXPORT SUNErrCode N_VLinearCombinationVectorArray_Vulkan(
  int nvec, int nsum, sunrealtype* c, N_Vector** X, N_Vector* Z);
SUNDIALS_EXPORT SUNErrCode N_VWrmsNormVectorArray_Vulkan(int nvec, N_Vector* X,
                                                         N_Vector* W,
                                                         sunrealtype* nrm);
SUNDIALS_EXPORT SUNErrCode N_VWrmsNormMaskVectorArray_Vulkan(
  int nvec, N_Vector* X, N_Vector* W, N_Vector id, sunrealtype* nrm);

/* OPTIONAL local reduction kernels (no parallel communication) */
SUNDIALS_EXPORT sunrealtype N_VWSqrSumLocal_Vulkan(N_Vector x, N_Vector w);
SUNDIALS_EXPORT sunrealtype N_VWSqrSumMaskLocal_Vulkan(N_Vector x, N_Vector w,
                                                       N_Vector id);

/* OPTIONAL XBraid interface operations */
SUNDIALS_EXPORT SUNErrCode N_VBufSize_Vulkan(N_Vector x, sunindextype* size);
SUNDIALS_EXPORT SUNErrCode N_VBufPack_Vulkan(N_Vector x, void* buf);
SUNDIALS_EXPORT SUNErrCode N_VBufUnpack_Vulkan(N_Vector x, void* buf);

/* OPTIONAL operations for debugging */
SUNDIALS_EXPORT void N_VPrint_Vulkan(N_Vector v);
SUNDIALS_EXPORT void N_VPrintFile_Vulkan(N_Vector v, FILE* outfile);

SUNDIALS_EXPORT SUNErrCode N_VEnableFusedOps_Vulkan(N_Vector v,
                                                    sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode N_VEnableLinearCombination_Vulkan(N_Vector v,
                                                             sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode N_VEnableScaleAddMulti_Vulkan(N_Vector v,
                                                         sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode N_VEnableDotProdMulti_Vulkan(N_Vector v,
                                                        sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode N_VEnableLinearSumVectorArray_Vulkan(N_Vector v,
                                                                sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode N_VEnableScaleVectorArray_Vulkan(N_Vector v,
                                                            sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode N_VEnableConstVectorArray_Vulkan(N_Vector v,
                                                            sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode N_VEnableWrmsNormVectorArray_Vulkan(N_Vector v,
                                                               sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode
N_VEnableWrmsNormMaskVectorArray_Vulkan(N_Vector v, sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode
N_VEnableScaleAddMultiVectorArray_Vulkan(N_Vector v, sunbooleantype tf);
SUNDIALS_EXPORT SUNErrCode
N_VEnableLinearCombinationVectorArray_Vulkan(N_Vector v, sunbooleantype tf);

#ifdef __cplusplus
}
#endif

#endif
