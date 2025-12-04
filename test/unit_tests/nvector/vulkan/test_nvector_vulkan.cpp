/* -----------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2025, Lawrence Livermore National Security,
 * University of Maryland Baltimore County, and the SUNDIALS contributors.
 * Copyright (c) 2013-2025, Lawrence Livermore National Security
 * and Southern Methodist University.
 * Copyright (c) 2002-2013, Lawrence Livermore National Security.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the testing routine to check the Vulkan NVECTOR module
 * implementation.
 * -----------------------------------------------------------------*/

#include <nvector/nvector_vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunmemory/sunmemory_vulkan.h>

#include "test_nvector.h"

/* ----------------------------------------------------------------------
 * Main NVector Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  int fails = 0;             /* counter for test failures */
  int retval;                /* function return value     */
  sunindextype length;       /* vector length             */
  N_Vector U, V, W, X, Y, Z; /* test vectors              */
  int print_timing;          /* turn timing on/off        */

  Test_Init(SUN_COMM_NULL);

  /* check input and set vector length */
  if (argc < 3)
  {
    printf("ERROR: TWO (2) Inputs required: vector length, print timing \n");
    Test_Finalize();
    return (-1);
  }

  length = (sunindextype)atol(argv[1]);
  if (length <= 0)
  {
    printf("ERROR: length of vector must be a positive integer \n");
    Test_Finalize();
    return (-1);
  }

  print_timing = atoi(argv[2]);
  SetTiming(print_timing, 0);

  printf("Testing Vulkan N_Vector \n");
  printf("Vector length %ld \n", (long int)length);

  /* Create new vectors */
  W = N_VNewEmpty_Vulkan(sunctx);
  if (W == NULL)
  {
    printf("FAIL: Unable to create a new empty vector \n\n");
    Test_Finalize();
    return (1);
  }

  /* attach length to empty vector so set/get array pointer tests have size */
  N_VectorContent_Vulkan wcontent = (N_VectorContent_Vulkan)W->content;
  wcontent->length                = length;
  if (wcontent->mem_helper == NULL)
  {
    wcontent->mem_helper = SUNMemoryHelper_Vulkan(sunctx);
    wcontent->own_helper = SUNTRUE;
    if (wcontent->mem_helper == NULL)
    {
      N_VDestroy(W);
      printf("FAIL: Unable to create Vulkan memory helper \n\n");
      Test_Finalize();
      return (1);
    }
  }

  X = N_VNew_Vulkan(length, sunctx);
  if (X == NULL)
  {
    N_VDestroy(W);
    printf("FAIL: Unable to create a new vector \n\n");
    Test_Finalize();
    return (1);
  }

  /* Check vector ID */
  fails += Test_N_VGetVectorID(X, SUNDIALS_NVEC_CUSTOM, 0);

  /* Check vector length */
  fails += Test_N_VGetLength(X, 0);

  /* Check vector communicator */
  fails += Test_N_VGetCommunicator(X, SUN_COMM_NULL, 0);

  /* Test clone functions */
  fails += Test_N_VCloneEmpty(X, 0);
  fails += Test_N_VClone(X, length, 0);
  fails += Test_N_VCloneEmptyVectorArray(5, X, 0);
  fails += Test_N_VCloneVectorArray(5, X, length, 0);

  /* Test setting/getting array data */
  fails += Test_N_VSetArrayPointer(W, length, 0);
  fails += Test_N_VGetArrayPointer(X, length, 0);

  /* Clone additional vectors for testing */
  Y = N_VClone(X);
  if (Y == NULL)
  {
    N_VDestroy(W);
    N_VDestroy(X);
    printf("FAIL: Unable to create a new vector \n\n");
    Test_Finalize();
    return (1);
  }

  Z = N_VClone(X);
  if (Z == NULL)
  {
    N_VDestroy(W);
    N_VDestroy(X);
    N_VDestroy(Y);
    printf("FAIL: Unable to create a new vector \n\n");
    Test_Finalize();
    return (1);
  }

  /* Standard vector operation tests */
  printf("\nTesting standard vector operations:\n\n");

  fails += Test_N_VConst(X, length, 0);
  fails += Test_N_VLinearSum(X, Y, Z, length, 0);
  fails += Test_N_VProd(X, Y, Z, length, 0);
  fails += Test_N_VDiv(X, Y, Z, length, 0);
  fails += Test_N_VScale(X, Z, length, 0);
  fails += Test_N_VAbs(X, Z, length, 0);
  fails += Test_N_VInv(X, Z, length, 0);
  fails += Test_N_VAddConst(X, Z, length, 0);
  fails += Test_N_VDotProd(X, Y, length, 0);
  fails += Test_N_VMaxNorm(X, length, 0);
  fails += Test_N_VWrmsNorm(X, Y, length, 0);
  fails += Test_N_VWrmsNormMask(X, Y, Z, length, 0);
  fails += Test_N_VMin(X, length, 0);
  fails += Test_N_VWL2Norm(X, Y, length, 0);
  fails += Test_N_VL1Norm(X, length, 0);
  fails += Test_N_VCompare(X, Z, length, 0);
  fails += Test_N_VInvTest(X, Z, length, 0);
  fails += Test_N_VConstrMask(X, Y, Z, length, 0);
  fails += Test_N_VMinQuotient(X, Y, length, 0);

  /* Fused and vector array operations tests (disabled) */
  printf("\nTesting fused and vector array operations (disabled):\n\n");

  /* create vector and disable all fused and vector array operations */
  U      = N_VNew_Vulkan(length, sunctx);
  retval = N_VEnableFusedOps_Vulkan(U, SUNFALSE);
  if (U == NULL || retval != 0)
  {
    N_VDestroy(W);
    N_VDestroy(X);
    N_VDestroy(Y);
    N_VDestroy(Z);
    printf("FAIL: Unable to create a new vector \n\n");
    Test_Finalize();
    return (1);
  }

  /* fused operations */
  fails += Test_N_VLinearCombination(U, length, 0);
  fails += Test_N_VScaleAddMulti(U, length, 0);
  fails += Test_N_VDotProdMulti(U, length, 0);

  /* vector array operations */
  fails += Test_N_VLinearSumVectorArray(U, length, 0);
  fails += Test_N_VScaleVectorArray(U, length, 0);
  fails += Test_N_VConstVectorArray(U, length, 0);
  fails += Test_N_VWrmsNormVectorArray(U, length, 0);
  fails += Test_N_VWrmsNormMaskVectorArray(U, length, 0);
  fails += Test_N_VScaleAddMultiVectorArray(U, length, 0);
  fails += Test_N_VLinearCombinationVectorArray(U, length, 0);

  /* Fused and vector array operations tests (enabled) */
  printf("\nTesting fused and vector array operations (enabled):\n\n");

  /* create vector and enable all fused and vector array operations */
  V      = N_VNew_Vulkan(length, sunctx);
  retval = N_VEnableFusedOps_Vulkan(V, SUNTRUE);
  if (V == NULL || retval != 0)
  {
    N_VDestroy(W);
    N_VDestroy(X);
    N_VDestroy(Y);
    N_VDestroy(Z);
    N_VDestroy(U);
    printf("FAIL: Unable to create a new vector \n\n");
    Test_Finalize();
    return (1);
  }

  /* fused operations */
  fails += Test_N_VLinearCombination(V, length, 0);
  fails += Test_N_VScaleAddMulti(V, length, 0);
  fails += Test_N_VDotProdMulti(V, length, 0);

  /* vector array operations */
  fails += Test_N_VLinearSumVectorArray(V, length, 0);
  fails += Test_N_VScaleVectorArray(V, length, 0);
  fails += Test_N_VConstVectorArray(V, length, 0);
  fails += Test_N_VWrmsNormVectorArray(V, length, 0);
  fails += Test_N_VWrmsNormMaskVectorArray(V, length, 0);
  fails += Test_N_VScaleAddMultiVectorArray(V, length, 0);
  fails += Test_N_VLinearCombinationVectorArray(V, length, 0);

  /* local reduction operations (partial support) */
  printf("\nTesting local reduction operations (partial support):\n\n");

  if (X->ops->nvwsqrsumlocal)
  {
    fails += Test_N_VWSqrSumLocal(X, Y, length, 0);
  }
  else
  {
    printf("SKIP: N_VWSqrSumLocal not implemented\n");
  }

  if (X->ops->nvwsqrsummasklocal)
  {
    fails += Test_N_VWSqrSumMaskLocal(X, Y, Z, length, 0);
  }
  else
  {
    printf("SKIP: N_VWSqrSumMaskLocal not implemented\n");
  }

  /* XBraid interface operations */
  printf("\nTesting XBraid interface operations:\n\n");

  fails += Test_N_VBufSize(X, length, 0);
  fails += Test_N_VBufPack(X, length, 0);
  fails += Test_N_VBufUnpack(X, length, 0);

  /* Free vectors */
  N_VDestroy(W);
  N_VDestroy(X);
  N_VDestroy(Y);
  N_VDestroy(Z);
  N_VDestroy(U);
  N_VDestroy(V);

  /* Print result */
  if (fails) { printf("FAIL: NVector module failed %i tests \n\n", fails); }
  else { printf("SUCCESS: NVector module passed all tests \n\n"); }

  Test_Finalize();
  return (fails);
}

/* ----------------------------------------------------------------------
 * Implementation specific utility functions for vector tests
 * --------------------------------------------------------------------*/
int check_ans(sunrealtype ans, N_Vector X, sunindextype local_length)
{
  int failure = 0;
  sunindextype i;
  sunrealtype* Xdata;

  Xdata = N_VGetHostArrayPointer_Vulkan(X);
  if (Xdata == NULL) { return 1; }

  /* check vector data */
  for (i = 0; i < local_length; i++) { failure += SUNRCompare(Xdata[i], ans); }

  return (failure > ZERO) ? (1) : (0);
}

sunbooleantype has_data(N_Vector X)
{
  /* check if data array is non-null */
  return (N_VGetHostArrayPointer_Vulkan(X) == NULL) ? SUNFALSE : SUNTRUE;
}

void set_element(N_Vector X, sunindextype i, sunrealtype val)
{
  /* set i-th element of data array */
  set_element_range(X, i, i, val);
}

void set_element_range(N_Vector X, sunindextype is, sunindextype ie,
                       sunrealtype val)
{
  sunindextype i;

  /* set elements [is,ie] of the data array */
  sunrealtype* xd = N_VGetHostArrayPointer_Vulkan(X);
  if (xd == NULL) { return; }

  for (i = is; i <= ie; i++) { xd[i] = val; }

  /* refresh helper state so device data matches host modifications */
  N_VSetHostArrayPointer_Vulkan(xd, X);
  N_VCopyToDevice_Vulkan(X);
}

sunrealtype get_element(N_Vector X, sunindextype i)
{
  /* get i-th element of data array */
  sunrealtype* xd = N_VGetHostArrayPointer_Vulkan(X);
  if (xd == NULL) { return ZERO; }
  return xd[i];
}

double max_time(N_Vector X, double time)
{
  /* not running in parallel, just return input time */
  return (time);
}

void sync_device(N_Vector x)
{
  /* ensure device view matches the current host buffer */
  sunrealtype* hx = N_VGetHostArrayPointer_Vulkan(x);
  if (hx == NULL) { return; }
  N_VSetHostArrayPointer_Vulkan(hx, x);
  N_VCopyToDevice_Vulkan(x);
}
