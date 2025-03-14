const std = @import("std");
const builtin = @import("builtin");

const RunArgs = []const []const u8;

const c_flags: RunArgs = &.{
    "-std=gnu99",
    "-g",
    "-DNDEBUG",
    "-DSUNDIALS_STATIC_DEFINE",
};

fn sundials_add_library(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    name: []const u8,
    sources: []const []const u8,
    config_header: *std.Build.Step.ConfigHeader,
) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = name,
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.addCSourceFiles(.{ .files = sources, .flags = c_flags });

    lib.addConfigHeader(config_header);
    lib.addIncludePath(b.path("include/"));
    lib.addIncludePath(b.path("src/sundials/"));

    lib.linkLibC();

    return lib;
}

fn sundials_add_executable(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    name: []const u8,
    sources: []const []const u8,
    config_header: *std.Build.Step.ConfigHeader,
    library: *std.Build.Step.Compile,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.addCSourceFiles(.{ .files = sources, .flags = c_flags });

    exe.addConfigHeader(config_header);
    exe.addIncludePath(b.path("include/"));

    exe.linkLibrary(library);

    return exe;
}

fn configHeader(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.ConfigHeader {
    _ = target;

    return b.addConfigHeader(.{
        .style = .{ .cmake = b.path("include/sundials/sundials_config.in") },
        .include_path = "sundials/sundials_config.h",
    }, .{
        .SUNDIALS_DEPRECATED_MSG_MACRO = "__attribute__ ((__deprecated__(msg)))",
        .PACKAGE_VERSION_MAJOR = 7,
        .PACKAGE_VERSION_MINOR = 2,
        .PACKAGE_VERSION_PATCH = 1,
        .PACKAGE_VERSION_LABEL = "",
        .PACKAGE_VERSION = "7.2.1",
        .SUNDIALS_GIT_VERSION = "",
        .SUNDIALS_C_COMPILER_HAS_BUILTIN_EXPECT = 1,
        // .SUNDIALS_C_COMPILER_HAS_ATTRIBUTE_ASSUME = 0,
        // .SUNDIALS_C_COMPILER_HAS_BUILTIN_ASSUME = 0,
        // .SUNDIALS_C_COMPILER_HAS_ASSUME = 0,
        .SUNDIALS_C_COMPILER_HAS_ATTRIBUTE_UNUSED = 1,
        .PRECISION_LEVEL = "#define SUNDIALS_DOUBLE_PRECISION 1",
        .INDEX_TYPE = "#define SUNDIALS_INT64_T 1",
        .SUNDIALS_CINDEX_TYPE = "int64_t",
        .SUNDIALS_HAVE_POSIX_TIMERS = 1,
        // .SUNDIALS_BUILD_PACKAGE_FUSED_KERNELS=0,
        // .SUNDIALS_BUILD_WITH_MONITORING=0,
        // .SUNDIALS_BUILD_WITH_PROFILING=0,
        // .SUNDIALS_ENABLE_ERROR_CHECKS=0,
        .SUNDIALS_LOGGING_LEVEL = 2,
        .CMAKE_C_COMPILER_ID = "zig",
        .CMAKE_C_COMPILER_VERSION = builtin.zig_version_string,
        .CMAKE_C_FLAGS = "",
        .CMAKE_CXX_COMPILER_ID = "",
        .CMAKE_CXX_COMPILER_VERSION = "",
        .CMAKE_CXX_FLAGS = "",
        .CMAKE_FORTRAN_COMPILER_ID = "",
        .CMAKE_FORTRAN_COMPILER_VERSION = "",
        .CMAKE_FORTRAN_FLAGS = "",
        .CMAKE_BUILD_TYPE = @tagName(optimize),
        .JOB_ID = "",
        .JOB_START_TIME = "",
        .SUNDIALS_TPL_LIST = "",
        .SUNDIALS_TPL_LIST_SIZE = "",
        .SPACK_VERSION = "",
        // .SUNDIALS_CALIPER_ENABLED = 0,
        // .SUNDIALS_ADIAK_ENABLED = 0,
        // .SUNDIALS_GINKGO_ENABLED = 0,
        .Ginkgo_VERSION = "",
        // .SUNDIALS_HYPRE_ENABLED = 0,
        .HYPRE_VERSION = "",
        .HYPRE_VERSION_MAJOR = "",
        .HYPRE_VERSION_MINOR = "",
        .HYPRE_VERSION_PATCH = "",
        // .SUNDIALS_KLU_ENABLED = 0,
        .KLU_VERSION = "",
        // .SUNDIALS_KOKKOS_ENABLED = 0,
        .Kokkos_VERSION = "",
        // .SUNDIALS_KOKKOS_KERNELS_ENABLED = 0,
        .KokkosKernels_VERSION = "",
        // .SUNDIALS_BLAS_LAPACK_ENABLED = 0,
        .LAPACK_VERSION = "",
        // .SUNDIALS_MAGMA_ENABLED = 0,
        .MAGMA_VERSION = "",
        .MPI_C_COMPILER = "",
        .MPI_C_VERSION = "",
        .MPI_CXX_COMPILER = "",
        .MPI_CXX_VERSION = "",
        .MPI_FORTRAN_COMPILER = "",
        .MPI_FORTRAN_VERSION = "",
        // .SUNDIALS_ONEMKL_ENABLED = 0,
        .MKL_VERSION = "",
        // .SUNDIALS_OPENMP_ENABLED = 0,
        .OpenMP_VERSION = "",
        // .SUNDIALS_PETSC_ENABLED = 0,
        .PETSC_VERSION = "",
        // .SUNDIALS_PTHREADS_ENABLED = 0,
        .Threads_VERSION = "",
        // .SUNDIALS_RAJA_ENABLED = 0,
        .RAJA_VERSION = "",
        // .SUNDIALS_SUPERLUDIST_ENABLED = 0,
        .SUPERLUDIST_VERSION = "",
        // .SUNDIALS_SUPERLUMT_ENABLED = 0,
        .SUPERLUMT_VERSION = "",
        // .SUNDIALS_TRILLINOS_ENABLED = 0,
        .Trillinos_VERSION = "",
        // .SUNDIALS_XBRAID_ENABLED = 0,
        .XBRAID_VERSION = "",
        // .SUNDIALS_RAJA_BACKENDS_CUDA = 0,
        // .SUNDIALS_RAJA_BACKENDS_HIP = 0,
        // .SUNDIALS_RAJA_BACKENDS_SYCL = 0,
        // .SUNDIALS_GINKGO_BACKENDS_CUDA = 0,
        // .SUNDIALS_GINKGO_BACKENDS_HIP = 0,
        // .SUNDIALS_GINKGO_BACKENDS_OMP = 0,
        // .SUNDIALS_GINKGO_BACKENDS_REF = 0,
        // .SUNDIALS_GINKGO_BACKENDS_SYCL = 0,
        // .SUNDIALS_MAGMA_BACKENDS_CUDA = 0,
        // .SUNDIALS_MAGMA_BACKENDS_HIP = 0,
        .SUNDIALS_MPI_ENABLED = 0,
        // .SUNDIALS_ONEMKL_USE_GETRF_LOOP = 0,
        // .SUNDIALS_ONEMKL_USE_GETRS_LOOP = 0,
        .SUPERLUMT_THREAD_TYPE = "PTHREAD",
        // .SUNDIALS_TRILINOS_HAVE_MPI = 0,
        // .SUNDIALS_CUDA_ENABLED = 0,
        .CMAKE_CUDA_COMPILER_VERSION = "",
        .CMAKE_CUDA_COMPILER = "",
        .CMAKE_CUDA_ARCHITECTURES = "",
        // .SUNDIALS_HIP_ENABLED = 0,
        .HIP_VERSION = "",
        .AMDGPU_TARGETS = "",
        // .SUNDIALS_SYCL_2020_UNSUPPORTED = 0,
        .SUNDIALS_CONFIGH_BUILDS =
        \\#define SUNDIALS_ARKODE 1
        \\#define SUNDIALS_NVECTOR_SERIAL 1
        \\#define SUNDIALS_NVECTOR_MANYVECTOR 1
        \\#define SUNDIALS_SUNMATRIX_BAND 1
        \\#define SUNDIALS_SUNMATRIX_DENSE 1
        \\#define SUNDIALS_SUNMATRIX_SPARSE 1
        \\#define SUNDIALS_SUNLINSOL_BAND 1
        \\#define SUNDIALS_SUNLINSOL_DENSE 1
        \\#define SUNDIALS_SUNLINSOL_PCG 1
        \\#define SUNDIALS_SUNLINSOL_SPBCGS 1
        \\#define SUNDIALS_SUNLINSOL_SPFGMR 1
        \\#define SUNDIALS_SUNLINSOL_SPGMR 1
        \\#define SUNDIALS_SUNLINSOL_SPTFQMR 1
        \\#define SUNDIALS_SUNNONLINSOL_NEWTON 1
        \\#define SUNDIALS_SUNNONLINSOL_FIXEDPOINT 1
        ,
    });
}

const SundialsComponent = struct {
    name: []const u8,
    src_files: []const []const u8,
};

pub fn build(b: *std.Build) !void {
    const sundials_libs = [_]SundialsComponent{
        .{
            .name = "sundials_core",
            .src_files = &.{
                "src/sundials/sundials_adaptcontroller.c",
                "src/sundials/sundials_band.c",
                "src/sundials/sundials_context.c",
                "src/sundials/sundials_dense.c",
                "src/sundials/sundials_direct.c",
                "src/sundials/sundials_errors.c",
                "src/sundials/sundials_futils.c",
                "src/sundials/sundials_hashmap.c",
                "src/sundials/sundials_iterative.c",
                "src/sundials/sundials_linearsolver.c",
                "src/sundials/sundials_logger.c",
                "src/sundials/sundials_math.c",
                "src/sundials/sundials_matrix.c",
                "src/sundials/sundials_memory.c",
                "src/sundials/sundials_nonlinearsolver.c",
                "src/sundials/sundials_nvector_senswrapper.c",
                "src/sundials/sundials_nvector.c",
                "src/sundials/sundials_stepper.c",
                "src/sundials/sundials_profiler.c",
                "src/sundials/sundials_version.c",
            },
        },
        .{
            .name = "sundials_sunmemsys",
            .src_files = &.{"src/sunmemory/system/sundials_system_memory.c"},
        },
        .{
            .name = "sundials_nvecmanyvector",
            .src_files = &.{"src/nvector/manyvector/nvector_manyvector.c"},
        },
        .{
            .name = "sundials_nvecserial",
            .src_files = &.{"src/nvector/serial/nvector_serial.c"},
        },
        .{
            .name = "sundials_sunmatrixband",
            .src_files = &.{"src/sunmatrix/band/sunmatrix_band.c"},
        },
        .{
            .name = "sundials_sunmatrixsparse",
            .src_files = &.{"src/sunmatrix/sparse/sunmatrix_sparse.c"},
        },
        .{
            .name = "sundials_sunmatrixdense",
            .src_files = &.{"src/sunmatrix/dense/sunmatrix_dense.c"},
        },
        .{
            .name = "sundials_sunlinsolpcg",
            .src_files = &.{"src/sunlinsol/pcg/sunlinsol_pcg.c"},
        },
        .{
            .name = "sundials_sunlinsolspgmr",
            .src_files = &.{"src/sunlinsol/spgmr/sunlinsol_spgmr.c"},
        },
        .{
            .name = "sundials_sunlinsolsptfqmr",
            .src_files = &.{"src/sunlinsol/sptfqmr/sunlinsol_sptfqmr.c"},
        },
        .{
            .name = "sundials_sunlinsolspfgmr",
            .src_files = &.{"src/sunlinsol/spfgmr/sunlinsol_spfgmr.c"},
        },
        .{
            .name = "sundials_sunlinsolband",
            .src_files = &.{"src/sunlinsol/band/sunlinsol_band.c"},
        },
        .{
            .name = "sundials_sunlinsolspbcgs",
            .src_files = &.{"src/sunlinsol/spbcgs/sunlinsol_spbcgs.c"},
        },
        .{
            .name = "sundials_sunlinsoldense",
            .src_files = &.{"src/sunlinsol/dense/sunlinsol_dense.c"},
        },
        .{
            .name = "sunnonlinsol_fixedpoint",
            .src_files = &.{"src/sunnonlinsol/fixedpoint/sunnonlinsol_fixedpoint.c"},
        },
        .{
            .name = "sunnonlinsol_newton",
            .src_files = &.{"src/sunnonlinsol/newton/sunnonlinsol_newton.c"},
        },

        .{
            .name = "sundials_sunadaptcontrollersoderlind",
            .src_files = &.{"src/sunadaptcontroller/soderlind/sunadaptcontroller_soderlind.c"},
        },
        .{
            .name = "sundials_sunadaptcontrollermrihtol",
            .src_files = &.{"src/sunadaptcontroller/mrihtol/sunadaptcontroller_mrihtol.c"},
        },
        .{
            .name = "sundials_sunadaptcontrollerimexgus",
            .src_files = &.{"src/sunadaptcontroller/imexgus/sunadaptcontroller_imexgus.c"},
        },
    };

    const arkode_lib = SundialsComponent{
        .name = "arkode",
        .src_files = &.{
            "src/arkode/arkode_adapt.c",
            "src/arkode/arkode_arkstep_io.c",
            "src/arkode/arkode_arkstep_nls.c",
            "src/arkode/arkode_arkstep.c",
            "src/arkode/arkode_bandpre.c",
            "src/arkode/arkode_bbdpre.c",
            "src/arkode/arkode_butcher_dirk.c",
            "src/arkode/arkode_butcher_erk.c",
            "src/arkode/arkode_butcher.c",
            "src/arkode/arkode_erkstep_io.c",
            "src/arkode/arkode_erkstep.c",
            "src/arkode/arkode_forcingstep.c",
            "src/arkode/arkode_interp.c",
            "src/arkode/arkode_io.c",
            "src/arkode/arkode_ls.c",
            "src/arkode/arkode_lsrkstep_io.c",
            "src/arkode/arkode_lsrkstep.c",
            "src/arkode/arkode_mri_tables.c",
            "src/arkode/arkode_mristep_controller.c",
            "src/arkode/arkode_mristep_io.c",
            "src/arkode/arkode_mristep_nls.c",
            "src/arkode/arkode_mristep.c",
            "src/arkode/arkode_relaxation.c",
            "src/arkode/arkode_root.c",
            "src/arkode/arkode_splittingstep_coefficients.c",
            "src/arkode/arkode_splittingstep.c",
            "src/arkode/arkode_sprkstep_io.c",
            "src/arkode/arkode_sprkstep.c",
            "src/arkode/arkode_sprk.c",
            "src/arkode/arkode_sunstepper.c",
            "src/arkode/arkode_user_controller.c",
            "src/arkode/arkode.c",
        },
    };

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const config_header = configHeader(b, target, optimize);

    var libs = std.ArrayList(*std.Build.Step.Compile).init(b.allocator);
    defer libs.deinit();
    for (sundials_libs) |sundials_lib| {
        const lib = sundials_add_library(
            b,
            target,
            optimize,
            sundials_lib.name,
            sundials_lib.src_files,
            config_header,
        );
        try libs.append(lib);
    }

    const arkode = sundials_add_library(
        b,
        target,
        optimize,
        arkode_lib.name,
        arkode_lib.src_files,
        config_header,
    );
    for (libs.items) |sundials_lib| {
        arkode.linkLibrary(sundials_lib);
    }
    b.installArtifact(arkode);

    build_examples(b, arkode, target, optimize, config_header);
}

fn build_examples(
    b: *std.Build,
    arkode: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    config_header: *std.Build.Step.ConfigHeader,
) void {
    const SundialsRunTarget = struct {
        build_info: SundialsComponent,
        run_infos: []const RunArgs,
    };

    const arkode_examples = [_]SundialsRunTarget{
        .{
            .build_info = .{
                .name = "ark_brusselator1D_manyvec",
                .src_files = &.{"examples/arkode/C_manyvector/ark_brusselator1D_manyvec.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_analytic",
                .src_files = &.{"examples/arkode/C_serial/ark_analytic.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_advection_diffusion_reaction_splitting",
                .src_files = &.{"examples/arkode/C_serial/ark_advection_diffusion_reaction_splitting.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_analytic_lsrk",
                .src_files = &.{"examples/arkode/C_serial/ark_analytic_lsrk.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_analytic_lsrk_varjac",
                .src_files = &.{"examples/arkode/C_serial/ark_analytic_lsrk_varjac.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_analytic_mels",
                .src_files = &.{"examples/arkode/C_serial/ark_analytic_mels.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_analytic_nonlin",
                .src_files = &.{"examples/arkode/C_serial/ark_analytic_nonlin.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_analytic_partitioned",
                .src_files = &.{"examples/arkode/C_serial/ark_analytic_partitioned.c"},
            },
            .run_infos = &.{
                &.{"forcing"},
                &.{"splitting"},
                &.{ "splitting", "ARKODE_SPLITTING_BEST_2_2_2" },
                &.{ "splitting", "ARKODE_SPLITTING_RUTH_3_3_2" },
                &.{ "splitting", "ARKODE_SPLITTING_YOSHIDA_8_6_2" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_analytic_ssprk",
                .src_files = &.{"examples/arkode/C_serial/ark_analytic_ssprk.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_brusselator_1D_mri",
                .src_files = &.{"examples/arkode/C_serial/ark_brusselator_1D_mri.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_brusselator_fp",
                .src_files = &.{"examples/arkode/C_serial/ark_brusselator_fp.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_brusselator_mri",
                .src_files = &.{"examples/arkode/C_serial/ark_brusselator_mri.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_brusselator",
                .src_files = &.{"examples/arkode/C_serial/ark_brusselator.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_brusselator1D_imexmri",
                .src_files = &.{"examples/arkode/C_serial/ark_brusselator1D_imexmri.c"},
            },
            .run_infos = &.{
                &.{ "0", "0.001" },
                &.{ "2", "0.001" },
                &.{ "3", "0.001" },
                &.{ "4", "0.001" },
                &.{ "5", "0.001" },
                &.{ "6", "0.001" },
                &.{ "7", "0.001" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_brusselator1D",
                .src_files = &.{"examples/arkode/C_serial/ark_brusselator1D.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_conserved_exp_entropy_ark",
                .src_files = &.{"examples/arkode/C_serial/ark_conserved_exp_entropy_ark.c"},
            },
            .run_infos = &.{
                &.{ "1", "0" },
                &.{ "1", "1" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_conserved_exp_entropy_erk",
                .src_files = &.{"examples/arkode/C_serial/ark_conserved_exp_entropy_erk.c"},
            },
            .run_infos = &.{
                &.{"1"},
            },
        },
        .{
            .build_info = .{
                .name = "ark_damped_harmonic_symplectic",
                .src_files = &.{"examples/arkode/C_serial/ark_damped_harmonic_symplectic.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_dissipated_exp_entropy",
                .src_files = &.{"examples/arkode/C_serial/ark_dissipated_exp_entropy.c"},
            },
            .run_infos = &.{
                &.{ "1", "0" },
                &.{ "1", "1" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_harmonic_symplectic",
                .src_files = &.{"examples/arkode/C_serial/ark_harmonic_symplectic.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_heat1D_adapt",
                .src_files = &.{"examples/arkode/C_serial/ark_heat1D_adapt.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_heat1D",
                .src_files = &.{"examples/arkode/C_serial/ark_heat1D.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_kepler",
                .src_files = &.{"examples/arkode/C_serial/ark_kepler.c"},
            },
            .run_infos = &.{
                &.{ "--stepper", "ERK", "--step-mode", "adapt" },
                &.{ "--stepper", "ERK", "--step-mode", "fixed", "--count-orbits" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--count-orbits", "--use-compensated-sums" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_EULER_1_1", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_LEAPFROG_2_2", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_MCLACHLAN_2_2", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_MCLACHLAN_3_3", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_MCLACHLAN_4_4", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_MCLACHLAN_5_6", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_PSEUDO_LEAPFROG_2_2", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_RUTH_3_3", "--tf", "50", "--check-order", "--nout", "1" },
                &.{ "--stepper", "SPRK", "--step-mode", "fixed", "--method", "ARKODE_SPRK_YOSHIDA_6_8", "--tf", "50", "--check-order", "--nout", "1" },
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_kpr_mri",
                .src_files = &.{"examples/arkode/C_serial/ark_kpr_mri.c"},
            },
            .run_infos = &.{
                &.{ "0", "1", "0.005" },
                &.{ "1", "0", "0.01" },
                &.{ "1", "1", "0.002" },
                &.{ "2", "4", "0.002" },
                &.{ "3", "2", "0.001" },
                &.{ "4", "3", "0.001" },
                &.{ "5", "4", "0.001" },
                &.{ "6", "5", "0.001" },
                &.{ "7", "2", "0.002" },
                &.{ "8", "3", "0.001", "-100", "100", "0.5", "1" },
                &.{ "9", "3", "0.001", "-100", "100", "0.5", "1" },
                &.{ "10", "4", "0.001", "-100", "100", "0.5", "1" },
                &.{ "11", "2", "0.001" },
                &.{ "12", "3", "0.005" },
                &.{ "13", "4", "0.01" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_KrylovDemo_prec",
                .src_files = &.{"examples/arkode/C_serial/ark_KrylovDemo_prec.c"},
            },
            .run_infos = &.{
                &.{},
                &.{"1"},
                &.{"2"},
            },
        },
        .{
            .build_info = .{
                .name = "ark_onewaycouple_mri",
                .src_files = &.{"examples/arkode/C_serial/ark_onewaycouple_mri.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_reaction_diffusion_mri",
                .src_files = &.{"examples/arkode/C_serial/ark_reaction_diffusion_mri.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_robertson_constraints",
                .src_files = &.{"examples/arkode/C_serial/ark_robertson_constraints.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_robertson_root",
                .src_files = &.{"examples/arkode/C_serial/ark_robertson_root.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_robertson",
                .src_files = &.{"examples/arkode/C_serial/ark_robertson.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_twowaycouple_mri",
                .src_files = &.{"examples/arkode/C_serial/ark_twowaycouple_mri.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
    };

    const run_examples = b.step("examples", "Run the examples");
    for (arkode_examples) |arkode_example| {
        const exe = sundials_add_executable(
            b,
            target,
            optimize,
            arkode_example.build_info.name,
            arkode_example.build_info.src_files,
            config_header,
            arkode,
        );
        b.installArtifact(exe);

        for (arkode_example.run_infos) |run_info| {
            const run_example = b.addRunArtifact(exe);
            run_example.addArgs(run_info);
            run_example.setCwd(.{ .cwd_relative = b.getInstallPath(.prefix, "") });
            run_examples.dependOn(&run_example.step);
        }
    }
}
