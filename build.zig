const std = @import("std");
const builtin = @import("builtin");

const RunArgs = []const []const u8;

const common_flags: RunArgs = &.{
    // "-g",
    "-DNDEBUG",
    "-DSUNDIALS_STATIC_DEFINE",
};

const c_flags: RunArgs = &.{
    "-std=gnu99",
};

const cpp_flags: RunArgs = &.{
    "-std=c++17",
};

fn is_c(path: []const u8) bool {
    const extension = std.fs.path.extension(path);
    return std.ascii.eqlIgnoreCase(extension, ".c");
}

var kompute_dep: *std.Build.Dependency = undefined;

const SundialsFeatures = struct {
    with_klu: bool,
    with_superlumt: bool,
    with_lapack: bool,
};

fn sundials_add_compile_options(
    b: *std.Build,
    target: *std.Build.Step.Compile,
    sources: []const []const u8,
    config_header: *std.Build.Step.ConfigHeader,
) void {
    var has_c = false;
    var has_cpp = false;
    for (sources) |source| {
        if (is_c(source)) {
            target.addCSourceFile(.{ .file = b.path(source), .flags = common_flags ++ c_flags });
            has_c = true;
        } else {
            target.addCSourceFile(.{ .file = b.path(source), .flags = common_flags ++ cpp_flags });
            has_cpp = true;
        }
    }

    target.addConfigHeader(config_header);
    target.addIncludePath(b.path("include/"));
    target.addIncludePath(b.path("src/sundials/"));

    if (has_c) {
        target.linkLibC();
    }
    if (has_cpp) {
        target.linkLibCpp();
    }
}

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

    sundials_add_compile_options(b, lib, sources, config_header);
    if (std.mem.indexOf(u8, name, "vulkan")) |_| {
        const kompute_lib = kompute_dep.artifact("kompute");
        lib.linkLibrary(kompute_lib);
    }
    if (std.mem.indexOf(u8, name, "pthread")) |_| {
        lib.linkSystemLibrary("pthread");
    }

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

    sundials_add_compile_options(b, exe, sources, config_header);
    exe.linkLibrary(library);

    return exe;
}

fn appendConfigDefine(allocator: std.mem.Allocator, builder: *std.ArrayList(u8), macro: []const u8) void {
    builder.writer(allocator).print("#define {s} 1\n", .{macro}) catch @panic("OOM");
}

fn configHeader(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    features: SundialsFeatures,
) !*std.Build.Step.ConfigHeader {
    _ = target;

    var config_builds = std.ArrayList(u8).initCapacity(b.allocator, 1024) catch @panic("OOM");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_ARKODE");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_NVECTOR_SERIAL");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_NVECTOR_PTHREADS");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_NVECTOR_MANYVECTOR");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_NVECTOR_VULKAN");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNMATRIX_BAND");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNMATRIX_DENSE");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNMATRIX_SPARSE");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_BAND");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_DENSE");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_PCG");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_SPBCGS");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_SPFGMR");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_SPGMR");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_SPTFQMR");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNNONLINSOL_NEWTON");
    appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNNONLINSOL_FIXEDPOINT");
    if (features.with_klu) {
        appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_KLU");
    }
    if (features.with_lapack) {
        appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_LAPACKBAND");
        appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_LAPACKDENSE");
    }
    if (features.with_superlumt) {
        appendConfigDefine(b.allocator, &config_builds, "SUNDIALS_SUNLINSOL_SUPERLUMT");
    }

    return b.addConfigHeader(.{
        .style = .{ .cmake = b.path("include/sundials/sundials_config.in") },
        .include_path = "sundials/sundials_config.h",
    }, .{
        .SUNDIALS_DEPRECATED_MSG_MACRO = "__attribute__ ((__deprecated__(msg)))",
        .PACKAGE_VERSION_MAJOR = 7,
        .PACKAGE_VERSION_MINOR = 2,
        .PACKAGE_VERSION_PATCH = 1,
        .PACKAGE_VERSION_LABEL = "",
        .PACKAGE_VERSION = "7.5.0",
        .SUNDIALS_GIT_VERSION = "",
        .SUNDIALS_C_COMPILER_HAS_BUILTIN_EXPECT = 1,
        // .SUNDIALS_C_COMPILER_HAS_ATTRIBUTE_ASSUME = 0,
        // .SUNDIALS_C_COMPILER_HAS_BUILTIN_ASSUME = 0,
        // .SUNDIALS_C_COMPILER_HAS_ASSUME = 0,
        .SUNDIALS_C_COMPILER_HAS_ATTRIBUTE_UNUSED = 1,
        .PRECISION_LEVEL = "#define SUNDIALS_DOUBLE_PRECISION 1",
        .INDEX_TYPE = "#define SUNDIALS_INT64_T 1",
        .SUNDIALS_CINDEX_TYPE = "int64_t",
        .SUNDIALS_COUNTER_TYPE = "long int",
        .SUNDIALS_HAVE_POSIX_TIMERS = 1,
        // .SUNDIALS_BUILD_PACKAGE_FUSED_KERNELS=0,
        // .SUNDIALS_BUILD_WITH_MONITORING=0,
        // .SUNDIALS_BUILD_WITH_PROFILING=0,
        // .SUNDIALS_ENABLE_ERROR_CHECKS=0,
        .SUNDIALS_LOGGING_LEVEL = 2,
        .CMAKE_C_COMPILER_ID = "zig",
        .CMAKE_C_COMPILER_VERSION = builtin.zig_version_string,
        .CMAKE_C_FLAGS = "",
        .CMAKE_CXX_COMPILER_ID = "zig",
        .CMAKE_CXX_COMPILER_VERSION = builtin.zig_version_string,
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
        .SUNDIALS_PTHREADS_ENABLED = 1,
        .Threads_VERSION = "0",
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
        .SUNDIALS_CONFIGH_BUILDS = config_builds.toOwnedSlice(b.allocator) catch @panic("OOM"),
    });
}

const SundialsComponent = struct {
    name: []const u8,
    src_files: []const []const u8,
};

pub fn build(b: *std.Build) !void {
    const with_klu = b.option(bool, "with_klu", "Build KLU linear solver components and tests (requires SuiteSparse KLU)") orelse false;
    const with_superlumt = b.option(bool, "with_superlumt", "Build SuperLU_MT linear solver components and tests") orelse false;
    const with_lapack = b.option(bool, "with_lapack", "Build LAPACK-based linear solver components and tests") orelse false;

    const features = SundialsFeatures{
        .with_klu = with_klu,
        .with_superlumt = with_superlumt,
        .with_lapack = with_lapack,
    };

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    kompute_dep = b.dependency("kompute", .{
        .target = target,
        .optimize = optimize,
    });

    const config_header = configHeader(b, target, optimize, features) catch @panic("OOM");

    var sundials_components = std.ArrayList(SundialsComponent).initCapacity(b.allocator, 64) catch @panic("OOM");
    defer sundials_components.deinit(b.allocator);

    sundials_components.appendSlice(b.allocator, &.{
        .{
            .name = "sundials_core",
            .src_files = &.{
                "src/sundials/sundatanode/sundatanode_inmem.c",
                "src/sundials/sundials_adaptcontroller.c",
                "src/sundials/sundials_adjointcheckpointscheme.c",
                "src/sundials/sundials_adjointstepper.c",
                "src/sundials/sundials_band.c",
                "src/sundials/sundials_cli.c",
                "src/sundials/sundials_context.c",
                "src/sundials/sundials_datanode.c",
                "src/sundials/sundials_dense.c",
                "src/sundials/sundials_direct.c",
                "src/sundials/sundials_domeigestimator.c",
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
                "src/sundials/sundials_nvector.c",
                "src/sundials/sundials_nvector_senswrapper.c",
                "src/sundials/sundials_profiler.c",
                "src/sundials/sundials_stepper.c",
                "src/sundials/sundials_version.c",
            },
        },
        .{
            .name = "sundials_sunmemsys",
            .src_files = &.{"src/sunmemory/system/sundials_system_memory.c"},
        },
        .{
            .name = "sundials_sunmemvulkan",
            .src_files = &.{"src/sunmemory/vulkan/sundials_vulkan_memory.cpp"},
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
            .name = "sundials_nvecpthreads",
            .src_files = &.{"src/nvector/pthreads/nvector_pthreads.c"},
        },
        .{
            .name = "sundials_nvecvulkan",
            .src_files = &.{"src/nvector/vulkan/nvector_vulkan.cpp"},
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
        .{
            .name = "sundials_adjointcheckpointscheme_fixed",
            .src_files = &.{"src/sunadjointcheckpointscheme/fixed/sunadjointcheckpointscheme_fixed.c"},
        },
        .{
            .name = "sundials_sundomeigestpower",
            .src_files = &.{"src/sundomeigest/power/sundomeigest_power.c"},
        },
    }) catch @panic("OOM");

    if (features.with_klu) {
        sundials_components.append(b.allocator, .{
            .name = "sundials_sunlinsolklu",
            .src_files = &.{"src/sunlinsol/klu/sunlinsol_klu.c"},
        }) catch @panic("OOM");
    }
    if (features.with_lapack) {
        sundials_components.append(b.allocator, .{
            .name = "sundials_sunlinsollapackband",
            .src_files = &.{"src/sunlinsol/lapackband/sunlinsol_lapackband.c"},
        }) catch @panic("OOM");
        sundials_components.append(b.allocator, .{
            .name = "sundials_sunlinsollapackdense",
            .src_files = &.{"src/sunlinsol/lapackdense/sunlinsol_lapackdense.c"},
        }) catch @panic("OOM");
        sundials_components.append(b.allocator, .{
            .name = "sundials_sundomeigestarnoldi",
            .src_files = &.{"src/sundomeigest/arnoldi/sundomeigest_arnoldi.c"},
        }) catch @panic("OOM");
    }
    if (features.with_superlumt) {
        sundials_components.append(b.allocator, .{
            .name = "sundials_sunlinsolsuperlumt",
            .src_files = &.{"src/sunlinsol/superlumt/sunlinsol_superlumt.c"},
        }) catch @panic("OOM");
    }

    var sundials_libs = std.ArrayList(*std.Build.Step.Compile).initCapacity(
        b.allocator,
        sundials_components.items.len,
    ) catch @panic("OOM");
    defer sundials_libs.deinit(b.allocator);

    for (sundials_components.items) |sundials_lib| {
        const lib = sundials_add_library(
            b,
            target,
            optimize,
            sundials_lib.name,
            sundials_lib.src_files,
            config_header,
        );
        sundials_libs.append(b.allocator, lib) catch @panic("OOM");
    }

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
            "src/arkode/arkode_cli.c",
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
    const arkode = sundials_add_library(
        b,
        target,
        optimize,
        arkode_lib.name,
        arkode_lib.src_files,
        config_header,
    );
    for (sundials_libs.items) |sundials_lib| {
        arkode.linkLibrary(sundials_lib);
    }
    arkode.installHeadersDirectory(b.path("include"), "", .{});
    arkode.installHeader(config_header.getOutput(), "sundials/sundials_config.h");
    b.installArtifact(arkode);

    build_examples(b, target, optimize, features, config_header, arkode);
    build_unit_tests(b, target, optimize, features, config_header, arkode);
    build_benchmarks(b, target, optimize, config_header, arkode);
}

const SundialsRunTarget = struct {
    build_info: SundialsComponent,
    run_infos: []const RunArgs,
    has_main: bool = true,
};

fn build_benchmarks(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    config_header: *std.Build.Step.ConfigHeader,
    arkode: *std.Build.Step.Compile,
) void {
    const benchmarks = [_]SundialsRunTarget{
        .{
            .build_info = .{
                .name = "nvector_serial_benchmark",
                .src_files = &.{
                    "benchmarks/nvector/test_nvector_performance.c",
                    "benchmarks/nvector/serial/test_nvector_performance_serial.c",
                },
            },
            .run_infos = &.{
                &.{ "1000000", "8", "4", "10", "64", "1" },
            },
        },
        .{
            .build_info = .{
                .name = "nvector_pthreads_benchmark",
                .src_files = &.{
                    "benchmarks/nvector/test_nvector_performance.c",
                    "benchmarks/nvector/pthreads/test_nvector_performance_pthreads.c",
                },
            },
            .run_infos = &.{
                &.{ "1000000", "8", "4", "10", "64", "1", "8" },
            },
        },
    };

    const run_benchmarks = b.step("benchmarks", "Run the benchmarks");
    for (benchmarks) |benchmark| {
        const exe = sundials_add_executable(
            b,
            target,
            optimize,
            benchmark.build_info.name,
            benchmark.build_info.src_files,
            config_header,
            arkode,
        );
        exe.addIncludePath(b.path("benchmarks/nvector"));
        b.installArtifact(exe);

        for (benchmark.run_infos) |run_info| {
            const run_benchmark = b.addRunArtifact(exe);
            run_benchmark.addArgs(run_info);
            run_benchmark.setCwd(.{ .cwd_relative = b.getInstallPath(.prefix, "") });
            run_benchmarks.dependOn(&run_benchmark.step);
        }
    }
}

fn build_examples(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    features: SundialsFeatures,
    config_header: *std.Build.Step.ConfigHeader,
    arkode: *std.Build.Step.Compile,
) void {
    var arkode_examples = std.ArrayList(SundialsRunTarget).initCapacity(b.allocator, 64) catch @panic("OOM");
    defer arkode_examples.deinit(b.allocator);

    arkode_examples.appendSlice(b.allocator, &.{
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
    }) catch @panic("OOM");

    arkode_examples.append(b.allocator, .{
        .build_info = .{
            .name = "ark_analytic_lsrk_domeigest",
            .src_files = &.{"examples/arkode/C_serial/ark_analytic_lsrk_domeigest.c"},
        },
        .run_infos = &.{&.{}},
    }) catch @panic("OOM");

    arkode_examples.append(b.allocator, .{
        .build_info = .{
            .name = "ark_lotka_volterra_ASA",
            .src_files = &.{"examples/arkode/C_serial/ark_lotka_volterra_ASA.c"},
        },
        .run_infos = &.{&.{}},
    }) catch @panic("OOM");

    if (features.with_klu) {
        arkode_examples.append(b.allocator, .{
            .build_info = .{
                .name = "ark_brusselator1D_klu",
                .src_files = &.{"examples/arkode/C_klu/ark_brusselator1D_klu.c"},
            },
            .run_infos = &.{&.{}},
        }) catch @panic("OOM");
    }

    if (features.with_superlumt) {
        arkode_examples.append(b.allocator, .{
            .build_info = .{
                .name = "ark_brusselator1D_FEM_slu",
                .src_files = &.{"examples/arkode/C_superlu-mt/ark_brusselator1D_FEM_slu.c"},
            },
            .run_infos = &.{&.{}},
        }) catch @panic("OOM");
    }

    // C++ serial examples
    arkode_examples.appendSlice(b.allocator, &.{
        .{
            .build_info = .{
                .name = "ark_analytic_sys",
                .src_files = &.{"examples/arkode/CXX_serial/ark_analytic_sys.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_heat2D",
                .src_files = &.{"examples/arkode/CXX_serial/ark_heat2D.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_heat2D_lsrk",
                .src_files = &.{"examples/arkode/CXX_serial/ark_heat2D_lsrk.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_kpr_Mt",
                .src_files = &.{"examples/arkode/CXX_serial/ark_kpr_Mt.cpp"},
            },
            .run_infos = &.{
                &.{ "0", "5" },
                &.{ "1", "4" },
                &.{ "2", "8", "0", "-10" },
                &.{ "0", "4", "1" },
                &.{ "0", "-4" },
                &.{ "1", "-5" },
                &.{ "2", "-5", "0", "-10" },
                &.{ "1", "-3", "0", "-10", "0" },
                &.{ "0", "3", "0", "-10", "0" },
                &.{ "2", "4", "0", "-10", "0" },
                &.{ "0", "4", "0", "-10", "1", "10", "1" },
                &.{ "0", "4", "0", "-10", "0", "10", "1" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_kpr_nestedmri",
                .src_files = &.{"examples/arkode/CXX_serial/ark_kpr_nestedmri.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_pendulum",
                .src_files = &.{"examples/arkode/CXX_serial/ark_pendulum.cpp"},
            },
            .run_infos = &.{&.{}},
        },
    }) catch @panic("OOM");

    if (features.with_lapack) {
        arkode_examples.append(b.allocator, .{
            .build_info = .{
                .name = "ark_heat2D_lsrk_domeigest",
                .src_files = &.{"examples/arkode/CXX_lapack/ark_heat2D_lsrk_domeigest.cpp"},
            },
            .run_infos = &.{&.{}},
        }) catch @panic("OOM");
    }

    arkode_examples.append(b.allocator, .{
        .build_info = .{
            .name = "ark_sod_lsrk",
            .src_files = &.{"examples/arkode/CXX_manyvector/ark_sod_lsrk.cpp"},
        },
        .run_infos = &.{&.{}},
    }) catch @panic("OOM");

    const run_examples = b.step("examples", "Run the examples");
    for (arkode_examples.items) |arkode_example| {
        const exe = sundials_add_executable(
            b,
            target,
            optimize,
            arkode_example.build_info.name,
            arkode_example.build_info.src_files,
            config_header,
            arkode,
        );
        exe.addIncludePath(b.path("examples/utilities"));
        b.installArtifact(exe);

        for (arkode_example.run_infos) |run_info| {
            const run_example = b.addRunArtifact(exe);
            run_example.addArgs(run_info);
            run_example.setCwd(.{ .cwd_relative = b.getInstallPath(.prefix, "") });
            run_examples.dependOn(&run_example.step);
        }
    }
}

fn build_unit_tests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    features: SundialsFeatures,
    config_header: *std.Build.Step.ConfigHeader,
    arkode: *std.Build.Step.Compile,
) void {
    var unit_tests = std.ArrayList(SundialsRunTarget).initCapacity(b.allocator, 64) catch @panic("OOM");
    defer unit_tests.deinit(b.allocator);

    unit_tests.appendSlice(b.allocator, &.{
        .{
            .build_info = .{
                .name = "test_nvector_manyvector",
                .src_files = &.{
                    "test/unit_tests/nvector/manyvector/test_nvector_manyvector.c",
                    "test/unit_tests/nvector/test_nvector.c",
                },
            },
            .run_infos = &.{
                &.{ "1000", "100", "0" },
                &.{ "100", "1000", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_nvector_serial",
                .src_files = &.{
                    "test/unit_tests/nvector/serial/test_nvector_serial.c",
                    "test/unit_tests/nvector/test_nvector.c",
                },
            },
            .run_infos = &.{
                &.{ "1000", "0" },
                &.{ "10000", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_nvector_pthreads",
                .src_files = &.{
                    "test/unit_tests/nvector/pthreads/test_nvector_pthreads.c",
                    "test/unit_tests/nvector/test_nvector.c",
                },
            },
            .run_infos = &.{
                &.{ "1000", "1", "0" },
                &.{ "1000", "2", "0" },
                &.{ "1000", "4", "0" },
                &.{ "10000", "1", "0" },
                &.{ "10000", "2", "0" },
                &.{ "10000", "4", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunnonlinsol_newton",
                .src_files = &.{"test/unit_tests/sunnonlinsol/newton/test_sunnonlinsol_newton.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "test_sunnonlinsol_fixedpoint",
                .src_files = &.{"test/unit_tests/sunnonlinsol/fixedpoint/test_sunnonlinsol_fixedpoint.c"},
            },
            .run_infos = &.{
                &.{},
                &.{"2"},
                &.{ "2", "0.5" },
            },
        },
        // .{
        //     // needs to be built with SUNDIALS_BUILD_WITH_PROFILING
        //     .build_info = .{
        //         .name = "test_profiling",
        //         .src_files = &.{"test/unit_tests/profiling/test_profiling.cpp"},
        //     },
        //     .run_infos = &.{
        //         &.{},
        //     },
        // },
        .{
            .build_info = .{
                .name = "test_sunmemory_sys",
                .src_files = &.{"test/unit_tests/sunmemory/sys/test_sunmemory_sys.cpp"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "test_sunmatrix_dense",
                .src_files = &.{
                    "test/unit_tests/sunmatrix/dense/test_sunmatrix_dense.c",
                    "test/unit_tests/sunmatrix/test_sunmatrix.c",
                },
            },
            .run_infos = &.{
                &.{ "100", "100", "0" },
                &.{ "200", "1000", "0" },
                &.{ "2000", "100", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunmatrix_band",
                .src_files = &.{
                    "test/unit_tests/sunmatrix/band/test_sunmatrix_band.c",
                    "test/unit_tests/sunmatrix/test_sunmatrix.c",
                },
            },
            .run_infos = &.{
                &.{ "10", "2", "3", "0" },
                &.{ "300", "7", "4", "0" },
                &.{ "1000", "8", "8", "0" },
                &.{ "5000", "3", "20", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunmatrix_sparse",
                .src_files = &.{
                    "test/unit_tests/sunmatrix/sparse/test_sunmatrix_sparse.c",
                    "test/unit_tests/sunmatrix/test_sunmatrix.c",
                    "test/unit_tests/sunmatrix/dreadrb.c",
                },
            },
            .run_infos = &.{
                &.{ "400", "400", "0", "0" },
                &.{ "450", "450", "1", "0" },
                &.{ "200", "1000", "0", "0" },
                &.{ "6000", "350", "0", "0" },
                &.{ "500", "5000", "1", "0" },
                &.{ "4000", "800", "1", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunlinsol_dense",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/dense/test_sunlinsol_dense.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "10", "0" },
                &.{ "100", "0" },
                &.{ "500", "0" },
                &.{ "1000", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunlinsol_band",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/band/test_sunlinsol_band.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "10", "2", "3", "0" },
                &.{ "300", "7", "4", "0" },
                &.{ "1000", "8", "8", "0" },
                &.{ "5000", "3", "100", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunlinsol_spfgmr_serial",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/spfgmr/serial/test_sunlinsol_spfgmr_serial.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "100", "1", "100", "1e-14", "0" },
                &.{ "100", "2", "100", "1e-14", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunlinsol_spbcgs_serial",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/spbcgs/serial/test_sunlinsol_spbcgs_serial.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "100", "1", "100", "1e-16", "0" },
                &.{ "100", "2", "100", "1e-16", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunlinsol_pcg_serial",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/pcg/serial/test_sunlinsol_pcg_serial.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "100", "500", "1e-16", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunlinsol_sptfqmr_serial",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/sptfqmr/serial/test_sunlinsol_sptfqmr_serial.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "100", "1", "100", "1e-16", "0" },
                &.{ "100", "2", "100", "1e-16", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_sunlinsol_spgmr_serial",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/spgmr/serial/test_sunlinsol_spgmr_serial.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "100", "1", "1", "100", "1e-14", "0" },
                &.{ "100", "2", "1", "100", "1e-14", "0" },
                &.{ "100", "1", "2", "100", "1e-14", "0" },
                &.{ "100", "2", "2", "100", "1e-14", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_arkstepsetforcing",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_arkstepsetforcing.c"},
            },
            .run_infos = &.{
                &.{ "1", "0" },
                &.{ "1", "1" },
                &.{ "1", "2" },
                &.{ "1", "3" },
                &.{ "1", "4" },
                &.{ "1", "5" },
                &.{ "1", "3", "2.0", "10.0" },
                &.{ "1", "3", "2.0", "10.0", "2.0", "8.0" },
                &.{ "1", "3", "2.0", "10.0", "1.0", "5.0" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_getuserdata",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_getuserdata.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_tstop",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_tstop.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_reset",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_reset.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_splittingstep_coefficients",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_splittingstep_coefficients.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_interp",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_interp.c"},
            },
            .run_infos = &.{
                &.{"-100"},
                &.{"-10000"},
                &.{"-1000000"},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_forcingstep",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_forcingstep.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_mass",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_mass.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_innerstepper",
                .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_innerstepper.c"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "test_arkode_error_handling",
                .src_files = &.{"test/unit_tests/arkode/gtest/test_arkode_error_handling.cpp"},
            },
            .run_infos = &.{
                &.{},
            },
            .has_main = false,
        },
        .{
            .build_info = .{
                .name = "test_logging_arkode_forcingstep",
                .src_files = &.{"test/unit_tests/logging/test_logging_arkode_forcingstep.cpp"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "test_logging_arkode_splittingstep",
                .src_files = &.{"test/unit_tests/logging/test_logging_arkode_splittingstep.cpp"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "test_logging_arkode_arkstep",
                .src_files = &.{"test/unit_tests/logging/test_logging_arkode_arkstep.cpp"},
            },
            .run_infos = &.{
                &.{"0"},
                &.{ "1", "1", "1" },
                &.{ "1", "1", "0" },
                &.{ "1", "0" },
                &.{ "2", "1", "1" },
                &.{ "2", "1", "0" },
                &.{ "2", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_logging_arkode_lsrkstep",
                .src_files = &.{"test/unit_tests/logging/test_logging_arkode_lsrkstep.cpp"},
            },
            .run_infos = &.{
                &.{"0"},
                &.{"1"},
                &.{"2"},
                &.{"3"},
                &.{"4"},
                &.{"5"},
            },
        },
        .{
            .build_info = .{
                .name = "test_logging_arkode_mristep",
                .src_files = &.{"test/unit_tests/logging/test_logging_arkode_mristep.cpp"},
            },
            .run_infos = &.{
                &.{"0"},
                &.{ "1", "1", "1" },
                &.{ "1", "1", "0" },
                &.{ "1", "0" },
                &.{ "2", "1", "1" },
                &.{ "2", "1", "0" },
                &.{ "2", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "test_logging_arkode_erkstep",
                .src_files = &.{"test/unit_tests/logging/test_logging_arkode_erkstep.cpp"},
            },
            .run_infos = &.{
                &.{},
            },
        },
        .{
            .build_info = .{
                .name = "test_logging_arkode_sprkstep",
                .src_files = &.{"test/unit_tests/logging/test_logging_arkode_sprkstep.cpp"},
            },
            .run_infos = &.{
                &.{"0"},
                &.{"1"},
            },
        },
        // .{
        //     // needs to be built with SUNDIALS_ENABLE_ERROR_CHECKS
        //     .build_info = .{
        //         .name = "test_sundials_errors",
        //         .src_files = &.{"test/unit_tests/sundials/test_sundials_errors.cpp"},
        //     },
        //     .run_infos = &.{
        //         &.{},
        //     },
        // },
        .{
            .build_info = .{
                .name = "test_reduction_operators",
                .src_files = &.{"test/unit_tests/sundials/reductions/test_reduction_operators.cpp"},
            },
            .run_infos = &.{
                &.{},
            },
        },
    }) catch @panic("OOM");

    unit_tests.append(b.allocator, .{
        .build_info = .{
            .name = "ark_test_adapt",
            .src_files = &.{"test/unit_tests/arkode/C_serial/ark_test_adapt.c"},
        },
        .run_infos = &.{&.{}},
    }) catch @panic("OOM");

    // ARKODE C++ serial unit tests (non-adjoint)
    unit_tests.appendSlice(b.allocator, &[_]SundialsRunTarget{
        .{
            .build_info = .{
                .name = "ark_test_accumerror_brusselator",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_accumerror_brusselator.cpp"},
            },
            .run_infos = &.{
                &.{ "20", "3", "1" },
                &.{ "20", "-4", "0" },
                &.{ "20", "5", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_accumerror_kpr",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_accumerror_kpr.cpp"},
            },
            .run_infos = &.{
                &.{ "20", "2", "0" },
                &.{ "20", "3", "1" },
                &.{ "20", "-4", "1" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_analytic_sys_mri",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_analytic_sys_mri.cpp"},
            },
            .run_infos = &.{
                &.{"0"},
                &.{"1"},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_brusselator_mriadapt",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_brusselator_mriadapt.cpp"},
            },
            .run_infos = &.{
                &.{ "--rtol", "0.000004", "--scontrol", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_butcher",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_butcher.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_test_dahlquist_ark",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_dahlquist_ark.cpp"},
            },
            .run_infos = &.{
                &.{ "0", "-1", "0" },
                &.{ "0", "0", "0" },
                &.{ "0", "0", "1" },
                &.{ "0", "1", "0" },
                &.{ "0", "1", "1" },
                &.{ "1", "-1", "0" },
                &.{ "1", "0", "0" },
                &.{ "1", "0", "1" },
                &.{ "1", "1", "0" },
                &.{ "1", "1", "1" },
                &.{ "2", "-1", "0" },
                &.{ "2", "0", "0" },
                &.{ "2", "0", "1" },
                &.{ "2", "1", "0" },
                &.{ "2", "1", "1" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_dahlquist_erk",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_dahlquist_erk.cpp"},
            },
            .run_infos = &.{
                &.{"-1"},
                &.{"0"},
                &.{"1"},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_dahlquist_mri",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_dahlquist_mri.cpp"},
            },
            .run_infos = &.{
                &.{"-1"},
                &.{"0"},
                &.{"1"},
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_getjac",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_getjac.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_test_getjac_mri",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_getjac_mri.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_test_kpr_mriadapt",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_kpr_mriadapt.cpp"},
            },
            .run_infos = &.{
                &.{ "--hs", "0.002", "--rtol", "0.000004", "--scontrol", "0" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_slowerror_brusselator",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_slowerror_brusselator.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_test_slowerror_kpr",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_slowerror_kpr.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_test_slowerror_polynomial",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_slowerror_polynomial.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_test_splittingstep",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_splittingstep.cpp"},
            },
            .run_infos = &.{&.{}},
        },
        .{
            .build_info = .{
                .name = "ark_test_adjoint_ark",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_adjoint_ark.cpp"},
            },
            .run_infos = &.{
                &.{ "--check-freq", "1" },
                &.{ "--check-freq", "2" },
                &.{ "--check-freq", "5" },
                &.{ "--check-freq", "1", "--dont-keep" },
                &.{ "--check-freq", "2", "--dont-keep" },
                &.{ "--check-freq", "5", "--dont-keep" },
            },
        },
        .{
            .build_info = .{
                .name = "ark_test_adjoint_erk",
                .src_files = &.{"test/unit_tests/arkode/CXX_serial/ark_test_adjoint_erk.cpp"},
            },
            .run_infos = &.{
                &.{ "--check-freq", "1" },
                &.{ "--check-freq", "2" },
                &.{ "--check-freq", "5" },
                &.{ "--check-freq", "1", "--dont-keep" },
                &.{ "--check-freq", "2", "--dont-keep" },
                &.{ "--check-freq", "5", "--dont-keep" },
            },
        },
    }) catch @panic("OOM");

    if (features.with_klu) {
        unit_tests.append(b.allocator, .{
            .build_info = .{
                .name = "test_sunlinsol_klu",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/klu/test_sunlinsol_klu.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "300", "0", "0" },
                &.{ "300", "1", "0" },
                &.{ "1000", "0", "0" },
                &.{ "1000", "1", "0" },
            },
        }) catch @panic("OOM");
    }

    if (features.with_lapack) {
        unit_tests.append(b.allocator, .{
            .build_info = .{
                .name = "test_sunlinsol_lapackband",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/lapackband/test_sunlinsol_lapackband.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "10", "2", "3", "0", "0" },
                &.{ "300", "7", "4", "0", "0" },
                &.{ "1000", "8", "8", "0", "0" },
                &.{ "5000", "3", "100", "0", "0" },
            },
        }) catch @panic("OOM");
        unit_tests.append(b.allocator, .{
            .build_info = .{
                .name = "test_sunlinsol_lapackdense",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/lapackdense/test_sunlinsol_lapackdense.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "10", "0", "0" },
                &.{ "100", "0", "0" },
                &.{ "500", "0", "0" },
                &.{ "1000", "0", "0" },
            },
        }) catch @panic("OOM");
    }

    if (features.with_superlumt) {
        unit_tests.append(b.allocator, .{
            .build_info = .{
                .name = "test_sunlinsol_superlumt",
                .src_files = &.{
                    "test/unit_tests/sunlinsol/superlumt/test_sunlinsol_superlumt.c",
                    "test/unit_tests/sunlinsol/test_sunlinsol.c",
                },
            },
            .run_infos = &.{
                &.{ "300", "0", "1", "0" },
                &.{ "300", "1", "1", "0" },
                &.{ "1000", "0", "3", "0" },
                &.{ "1000", "1", "3", "0" },
            },
        }) catch @panic("OOM");
    }

    const googletest_dep = b.dependency("googletest", .{
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.step("test", "Run the unit tests");
    for (unit_tests.items) |unit_test| {
        const exe = sundials_add_executable(
            b,
            target,
            optimize,
            unit_test.build_info.name,
            unit_test.build_info.src_files,
            config_header,
            arkode,
        );

        // adding include paths for utils needed by sundials unit test
        const main_src_file = unit_test.build_info.src_files[0];
        exe.addIncludePath(b.path("src/"));
        var iter = std.fs.path.componentIterator(main_src_file) catch @panic("OOM");
        _ = iter.next();
        var path_component = iter.next() orelse std.debug.panic("{s} is not a unit test?", .{main_src_file});
        exe.addIncludePath(b.path(path_component.path)); // adding "./test/unit_tests/" directory
        path_component = iter.next() orelse std.debug.panic("{s} is not a unit test?", .{main_src_file});
        exe.addIncludePath(b.path(path_component.path)); // adding "./test/unit_tests/<test_group>/" directory

        // adding include paths for gmock needed by sundials unit test
        if (unit_test.has_main) {
            exe.addIncludePath(googletest_dep.artifact("gmock").getEmittedIncludeTree());
            exe.linkLibrary(googletest_dep.artifact("gmock"));
        } else {
            exe.addIncludePath(googletest_dep.artifact("gmock_main").getEmittedIncludeTree());
            exe.linkLibrary(googletest_dep.artifact("gmock_main"));
        }

        b.installArtifact(exe);

        for (unit_test.run_infos) |run_info| {
            const run_unit_test = b.addRunArtifact(exe);
            run_unit_test.addArgs(run_info);
            run_unit_test.setCwd(.{ .cwd_relative = b.getInstallPath(.prefix, "") });
            run_unit_tests.dependOn(&run_unit_test.step);
        }
    }
}
