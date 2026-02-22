const std = @import("std");
const nvector_complex = @import("nvector_complex.zig");

const c = nvector_complex.c;
const Complex = nvector_complex.Complex;

fn check_ans(val: Complex, tol: f64, N: usize, sunvec_x: c.N_Vector) i32 {
    const x = nvector_complex.N_VGetCVec(sunvec_x);
    for (0..N) |i| {
        if (@abs(x.data[i].re - val.re) > tol or @abs(x.data[i].im - val.im) > tol) {
            return 1;
        }
    }
    return 0;
}

pub fn main() !void {
    var fails: i32 = 0;
    const N: usize = 1000;

    var sunctx: c.SUNContext = null;
    if (c.SUNContext_Create(c.SUN_COMM_NULL, &sunctx) != 0) {
        std.debug.print("ERROR: SUNContext_Create failed\n", .{});
        return;
    }
    defer _ = c.SUNContext_Free(&sunctx);

    var Udata: [1000]Complex = undefined;
    for (0..N) |i| {
        Udata[i] = Complex.init(0, 0);
    }

    const sU = try nvector_complex.N_VMake_Complex(@intCast(N), &Udata, sunctx);
    const sV = try nvector_complex.N_VNew_Complex(@intCast(N), sunctx);
    const sW = try nvector_complex.N_VNew_Complex(@intCast(N), sunctx);
    const sX = try nvector_complex.N_VNew_Complex(@intCast(N), sunctx);
    const sY = try nvector_complex.N_VNew_Complex(@intCast(N), sunctx);
    const sZ = nvector_complex.N_VClone_Complex(sU) orelse return error.CloneFailed;

    defer {
        nvector_complex.N_VDestroy_Complex(sU);
        nvector_complex.N_VDestroy_Complex(sV);
        nvector_complex.N_VDestroy_Complex(sW);
        nvector_complex.N_VDestroy_Complex(sX);
        nvector_complex.N_VDestroy_Complex(sY);
        nvector_complex.N_VDestroy_Complex(sZ);
    }

    const X = nvector_complex.N_VGetCVec(sX);
    const Y = nvector_complex.N_VGetCVec(sY);
    const Z = nvector_complex.N_VGetCVec(sZ);

    // ! check vector ID
    if (nvector_complex.N_VGetVectorID_Complex(sU) != c.SUNDIALS_NVEC_CUSTOM) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
        // print *, '    Unrecognized vector type', FN_VGetVectorID(sU)
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! check vector length
    if (nvector_complex.N_VGetLength_Complex(sV) != N) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
        // print *, '    ', FN_VGetLength(sV), ' /= ', N
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VConst
    for (0..N) |i| {
        Udata[i] = Complex.init(0, 0);
    }
    c.N_VConst(1.0, sU);
    if (check_ans(Complex.init(1.0, 0.0), 1.e-14, N, sU) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VLinearSum
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-2.0, 2.0);
    }
    c.N_VLinearSum(1.0, sX, 1.0, sY, sY);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sY) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(2.0, -2.0);
    }
    c.N_VLinearSum(-1.0, sX, 1.0, sY, sY);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sY) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, -2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-2.0, 2.0);
    }
    c.N_VLinearSum(0.50, sX, 1.0, sY, sY);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sY) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, -2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-1.0, 1.0);
    }
    c.N_VLinearSum(1.0, sX, 1.0, sY, sX);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sX) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(2.0, -2.0);
    }
    c.N_VLinearSum(1.0, sX, -1.0, sY, sX);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sX) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, -2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-0.50, 0.50);
    }
    c.N_VLinearSum(1.0, sX, 2.0, sY, sX);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sX) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(-2.0, 2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(1.0, sX, 1.0, sY, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, -2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(1.0, sX, -1.0, sY, sZ);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, -2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(-1.0, sX, 1.0, sY, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, -2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-0.50, 0.50);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(1.0, sX, 2.0, sY, sZ);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(0.50, -0.50);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-2.0, 2.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(2.0, sX, 1.0, sY, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(-2.0, 2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-0.50, 0.50);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(-1.0, sX, 2.0, sY, sZ);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(0.50, -0.50);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(2.0, -2.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(2.0, sX, -1.0, sY, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-0.50, 0.50);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(2.0, sX, 2.0, sY, sZ);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(0.50, -0.50);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(2.0, sX, -2.0, sY, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-2.0, 2.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VLinearSum(2.0, sX, 0.50, sY, sZ);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VProd
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, 0.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-0.50, 0.00);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VProd(sX, sY, sZ);
    if (check_ans(Complex.init(-1.0, 0.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(0.0, 0.50);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(-2.00, 0.00);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VProd(sX, sY, sZ);
    if (check_ans(Complex.init(0.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, 2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(1.00, -2.00);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VProd(sX, sY, sZ);
    if (check_ans(Complex.init(5.0, 0.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VDiv
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, 0.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(2.0, 0.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VDiv(sX, sY, sZ);
    if (check_ans(Complex.init(0.50, 0.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(0.0, 1.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(2.0, 0.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VDiv(sX, sY, sZ);
    if (check_ans(Complex.init(0.0, 0.50), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(4.0, 2.0);
    }
    for (0..N) |i| {
        Y.data[i] = Complex.init(1.0, -1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VDiv(sX, sY, sZ);
    if (check_ans(Complex.init(1.0, 3.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VScale
    for (0..N) |i| {
        X.data[i] = Complex.init(0.50, -0.50);
    }
    c.N_VScale(2.0, sX, sX);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sX) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(-1.0, 1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VScale(1.0, sX, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(-1.0, 1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VScale(-1.0, sX, sZ);
    if (check_ans(Complex.init(1.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(-0.50, 0.50);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VScale(2.0, sX, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VAbs
    for (0..N) |i| {
        X.data[i] = Complex.init(-1.0, 0.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VAbs(sX, sZ);
    if (check_ans(Complex.init(1.0, 0.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, -0.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VAbs(sX, sZ);
    if (check_ans(Complex.init(1.0, 0.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(3.0, -4.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VAbs(sX, sZ);
    if (check_ans(Complex.init(5.0, 0.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VInv
    for (0..N) |i| {
        X.data[i] = Complex.init(2.0, 0.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VInv(sX, sZ);
    if (check_ans(Complex.init(0.50, 0.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    for (0..N) |i| {
        X.data[i] = Complex.init(0.0, 1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VInv(sX, sZ);
    if (check_ans(Complex.init(0.0, -1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    // ! test FN_VAddConst
    for (0..N) |i| {
        X.data[i] = Complex.init(1.0, 1.0);
    }
    for (0..N) |i| {
        Z.data[i] = Complex.init(0.0, 0.0);
    }
    c.N_VAddConst(sX, -2.0, sZ);
    if (check_ans(Complex.init(-1.0, 1.0), 1.e-14, N, sZ) != 0) {
        fails += 1;
        std.debug.print(">>> FAILED test\n", .{});
    } else {
        std.debug.print("PASSED test\n", .{});
    }
    //
    if (fails > 0) {
        std.debug.print("FAILURES: {}\n", .{fails});
        return error.TestFailed;
    } else {
        std.debug.print("ALL TESTS PASSED\n", .{});
    }
}
