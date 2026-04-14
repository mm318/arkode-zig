/*
 * Aggregate header for Vulkan-enabled Arkode-focused benchmark and unit-test
 * executables. This mirrors sundials_arkode.h and adds the Vulkan NVECTOR and
 * SUNMemory public headers so Vulkan tests can rely on a single umbrella.
 */

#ifndef SUNDIALS_ARKODE_VULKAN_H
#define SUNDIALS_ARKODE_VULKAN_H

#include <sundials/sundials_config.h>

#include <arkode/arkode.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_bandpre.h>
#include <arkode/arkode_bbdpre.h>
#include <arkode/arkode_butcher.h>
#include <arkode/arkode_butcher_dirk.h>
#include <arkode/arkode_butcher_erk.h>
#include <arkode/arkode_erkstep.h>
#include <arkode/arkode_forcingstep.h>
#include <arkode/arkode_ls.h>
#include <arkode/arkode_lsrkstep.h>
#include <arkode/arkode_mristep.h>
#include <arkode/arkode_splittingstep.h>
#include <arkode/arkode_sprk.h>
#include <arkode/arkode_sprkstep.h>

#include <nvector/nvector_serial.h>
#ifdef SUNDIALS_NVECTOR_PTHREADS
#include <nvector/nvector_pthreads.h>
#endif
#ifdef SUNDIALS_NVECTOR_MANYVECTOR
#include <nvector/nvector_manyvector.h>
#endif
#include <nvector/nvector_vulkan.h>

#include <sunadaptcontroller/sunadaptcontroller_imexgus.h>
#include <sunadaptcontroller/sunadaptcontroller_mrihtol.h>
#include <sunadaptcontroller/sunadaptcontroller_soderlind.h>
#include <sunadjointcheckpointscheme/sunadjointcheckpointscheme_fixed.h>

#include <sundials/sundials_adaptcontroller.h>
#include <sundials/sundials_adjointcheckpointscheme.h>
#include <sundials/sundials_adjointstepper.h>
#include <sundials/sundials_band.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_direct.h>
#include <sundials/sundials_domeigestimator.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_iterative.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_logger.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_memory.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_profiler.h>
#include <sundials/sundials_stepper.h>
#include <sundials/sundials_types.h>

#include <sunlinsol/sunlinsol_band.h>
#include <sunlinsol/sunlinsol_dense.h>
#ifdef SUNDIALS_SUNLINSOL_KLU
#include <sunlinsol/sunlinsol_klu.h>
#endif
#ifdef SUNDIALS_SUNLINSOL_LAPACKBAND
#include <sunlinsol/sunlinsol_lapackband.h>
#endif
#ifdef SUNDIALS_SUNLINSOL_LAPACKDENSE
#include <sunlinsol/sunlinsol_lapackdense.h>
#endif
#include <sunlinsol/sunlinsol_pcg.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include <sunlinsol/sunlinsol_spfgmr.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>
#ifdef SUNDIALS_SUNLINSOL_SUPERLUMT
#include <sunlinsol/sunlinsol_superlumt.h>
#endif

#include <sunmatrix/sunmatrix_band.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include <sunmemory/sunmemory_system.h>
#include <sunmemory/sunmemory_vulkan.h>

#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>

#ifdef __cplusplus
#include <sundials/sundials_base.hpp>
#include <sundials/sundials_context.hpp>
#include <sundials/sundials_convertibleto.hpp>
#include <sundials/sundials_core.hpp>
#include <sundials/sundials_linearsolver.hpp>
#include <sundials/sundials_matrix.hpp>
#include <sundials/sundials_memory.hpp>
#include <sundials/sundials_nonlinearsolver.hpp>
#include <sundials/sundials_nvector.hpp>
#include <sundials/sundials_profiler.hpp>
#include <sundials/sundials_vulkan_policies.hpp>
#include "sundials_reductions.hpp"
#endif

#endif
