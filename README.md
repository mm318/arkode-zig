# ARKODE
### Version 6.2.1 (Dec 2024)

**Daniel R. Reynolds,
  Department of Mathematics, SMU**

**David J. Gardner, Carol S. Woodward, and Cody J. Balos,
  Center for Applied Scientific Computing, LLNL**

ARKODE is a package for the solution of stiff, nonstiff, and multirate ordinary
differential equation (ODE) systems (initial value problems) given in linearly
implicit the form

  $$M(t) \\, y' = f_1(t,y) + f_2(t,y), \quad y(t_0) = y_0$$

The integration methods implemented in ARKODE include explicit and implicit
Runge-Kutta methods, implicit-explicit (IMEX) additive Runge-Kutta methods, and
multirate infinitesimal (MRI) methods.

ARKODE is part of a the SUNDIALS Suite of Nonlinear and Differential/Algebraic
equation Solvers which consists of ARKODE, CVODE, CVODES, IDA, IDAS, and KINSOL.
It is written in ANSI standard C and can be used in a variety of computing
environments including serial, shared memory, distributed memory, and
accelerator-based (e.g., GPU) systems. This flexibility is obtained from a
modular design that leverages the shared vector, matrix, linear solver, and
nonlinear solver APIs used across SUNDIALS packages.

## Installation ##

Add the dependency in your `build.zig.zon` by running the following command:

```
zig fetch --save=arkode_zig https://github.com/mm318/arkode-zig/archive/refs/heads/main.tar.gz
```

Then, in your `build.zig`:

```
const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const arkode_dep = b.dependency("arkode_zig", .{
        .target = target,
        .optimize = optimize,
    });
    const arkode_artifact = arkode_dep.artifact("arkode");

    const your_exe = b.addExecutable(.{
        .target = target,
        .optimize = optimize,
        // your other options...
    });
    your_exe.addIncludePath(arkode_artifact.getEmittedIncludeTree());
    your_exe.linkLibrary(arkode_artifact);
}
```

## Contributing ##

Bug fixes or minor changes are preferred via a pull request to the
[SUNDIALS GitHub repository](https://github.com/LLNL/sundials). For more
information on contributing see the [CONTRIBUTING](./CONTRIBUTING.md) file.

## Citing ##

See the [online documentation](https://sundials.readthedocs.io/en/latest/index.html#citing)
or [CITATIONS](./CITATIONS.md) file for information on how to cite SUNDIALS in
any publications reporting work done using SUNDIALS packages.

## Authors ##

The SUNDIALS library has been developed over many years by a number of
contributors. The current SUNDIALS team consists of Cody J. Balos,
David J. Gardner, Alan C. Hindmarsh, Daniel R. Reynolds, Steven B. Roberts, and
Carol S. Woodward. We thank Radu Serban for significant and critical past
contributions.

Other contributors to SUNDIALS include: James Almgren-Bell, Lawrence E. Banks,
Peter N. Brown, George Byrne, Rujeko Chinomona, Scott D. Cohen, Aaron Collier,
Keith E. Grant, Steven L. Lee, Shelby L. Lockhart, John Loffeld, Daniel McGreer,
Yu Pan, Slaven Peles, Cosmin Petra, H. Hunter Schwartz, Jean M. Sexton,
Dan Shumaker, Steve G. Smith, Shahbaj Sohal, Allan G. Taylor,
Hilari C. Tiedeman, Chris White, Ting Yan, and Ulrike M. Yang.

## Acknowledgements ##

This material is based on work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research, Scientific
Discovery through Advanced Computing (SciDAC) program via the Frameworks,
Algorithms, and Scalable Technologies for Mathematics (FASTMath) Institute under
DOE awards DE-AC52-07NA27344 and DE-SC-0021354.

This material is also based on work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research,
Next-Generation Scientific Software Technologies program under contract
DE-AC52-07NA27344.  Additional support is also provided by SciDAC
partnerships with the U.S. Department of Energy’s FES, NP, BES, OE, and BER
offices as well as the LLNL Institutional Scientific Capability Portfolio.


## License ##

SUNDIALS is released under the BSD 3-clause license. See the [LICENSE](./LICENSE)
and [NOTICE](./NOTICE) files for details. All new contributions must be made
under the BSD 3-clause license.

**Please Note** If you are using SUNDIALS with any third party libraries linked
in (e.g., LAPACK, KLU, SuperLU_MT, PETSc, *hypre*, etc.), be sure to review the
respective license of the package as that license may have more restrictive
terms than the SUNDIALS license.

```
SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-667205  (ARKODE)
UCRL-CODE-155951  (CVODE)
UCRL-CODE-155950  (CVODES)
UCRL-CODE-155952  (IDA)
UCRL-CODE-237203  (IDAS)
LLNL-CODE-665877  (KINSOL)
```
