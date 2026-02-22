const std = @import("std");

pub const c = @import("arkode-c");

pub const Complex = std.math.Complex(f64);

pub const CVec = struct {
    own_data: bool,
    len: i64,
    data: [*]Complex,
};

pub fn N_VNew_Complex(n: c.sunindextype, sunctx: c.SUNContext) !c.N_Vector {
    const v = c.N_VNewEmpty(sunctx) orelse return error.OutOfMemory;

    var content = try std.heap.c_allocator.create(CVec);
    content.own_data = true;
    content.len = n;
    content.data = (try std.heap.c_allocator.alloc(Complex, @intCast(n))).ptr;

    v.*.content = content;

    init_ops(v);
    return v;
}

pub fn N_VMake_Complex(n: c.sunindextype, data: []Complex, sunctx: c.SUNContext) !c.N_Vector {
    const v = c.N_VNewEmpty(sunctx) orelse return error.OutOfMemory;

    var content = try std.heap.c_allocator.create(CVec);
    content.own_data = false;
    content.len = n;
    content.data = data.ptr;

    v.*.content = content;

    init_ops(v);
    return v;
}

pub fn N_VGetCVec(v: c.N_Vector) *CVec {
    return @ptrCast(@alignCast(v.*.content));
}

fn init_ops(v: c.N_Vector) void {
    const ops = v.*.ops;
    ops.*.nvgetvectorid = N_VGetVectorID_Complex;
    ops.*.nvdestroy = N_VDestroy_Complex;
    ops.*.nvgetlength = N_VGetLength_Complex;
    ops.*.nvconst = N_VConst_Complex;
    ops.*.nvclone = N_VClone_Complex;
    ops.*.nvspace = N_VSpace_Complex;
    ops.*.nvlinearsum = N_VLinearSum_Complex;
    ops.*.nvprod = N_VProd_Complex;
    ops.*.nvdiv = N_VDiv_Complex;
    ops.*.nvscale = N_VScale_Complex;
    ops.*.nvabs = N_VAbs_Complex;
    ops.*.nvinv = N_VInv_Complex;
    ops.*.nvaddconst = N_VAddConst_Complex;
    ops.*.nvdotprod = N_VDotProd_Complex;
    ops.*.nvmaxnorm = N_VMaxNorm_Complex;
    ops.*.nvwrmsnorm = N_VWRMSNorm_Complex;
    ops.*.nvwrmsnormmask = N_VWRMSNormMask_Complex;
    ops.*.nvmin = N_VMin_Complex;
    ops.*.nvwl2norm = N_VWL2Norm_Complex;
    ops.*.nvl1norm = N_VL1Norm_Complex;
    ops.*.nvinvtest = N_VInvTest_Complex;
    ops.*.nvmaxnormlocal = N_VMaxNorm_Complex;
    ops.*.nvdotprodlocal = N_VDotProd_Complex;
    ops.*.nvminlocal = N_VMin_Complex;
    ops.*.nvl1normlocal = N_VL1Norm_Complex;
    ops.*.nvinvtestlocal = N_VInvTest_Complex;
    // SUNDIALS optional ones that Fortran bound
    ops.*.nvwsqrsumlocal = N_VWSqrSum_Complex;
    ops.*.nvwsqrsummasklocal = N_VWSqrSumMask_Complex;
}

pub export fn N_VGetVectorID_Complex(v: c.N_Vector) c.N_Vector_ID {
    _ = v;
    return c.SUNDIALS_NVEC_CUSTOM;
}

pub export fn N_VDestroy_Complex(v: c.N_Vector) void {
    if (v == null) return;
    const content = N_VGetCVec(v);
    if (content.own_data) {
        std.heap.c_allocator.free(content.data[0..@intCast(content.len)]);
    }
    std.heap.c_allocator.destroy(content);
    v.*.content = null;
    c.N_VFreeEmpty(v);
}

pub export fn N_VGetLength_Complex(v: c.N_Vector) c.sunindextype {
    return N_VGetCVec(v).len;
}

pub export fn N_VConst_Complex(const_val: c.sunrealtype, v: c.N_Vector) void {
    const x = N_VGetCVec(v);
    const val = Complex.init(const_val, 0.0);
    const n: usize = @intCast(x.len);
    for (0..n) |i| {
        x.data[i] = val;
    }
}

pub export fn N_VClone_Complex(w: c.N_Vector) c.N_Vector {
    const x = N_VGetCVec(w);
    const v = c.N_VNewEmpty(w.*.sunctx) orelse return null;
    _ = c.N_VCopyOps(w, v);

    var content = std.heap.c_allocator.create(CVec) catch return null;
    content.own_data = true;
    content.len = x.len;
    content.data = (std.heap.c_allocator.alloc(Complex, @intCast(x.len)) catch return null).ptr;

    v.*.content = content;
    return v;
}

pub export fn N_VSpace_Complex(v: c.N_Vector, lrw: [*c]c.sunindextype, liw: [*c]c.sunindextype) void {
    const x = N_VGetCVec(v);
    if (lrw != null) lrw[0] = 2 * x.len;
    if (liw != null) liw[0] = 3;
}

pub export fn N_VLinearSum_Complex(a: c.sunrealtype, x_vec: c.N_Vector, b: c.sunrealtype, y_vec: c.N_Vector, z_vec: c.N_Vector) void {
    const x = N_VGetCVec(x_vec);
    const y = N_VGetCVec(y_vec);
    const z = N_VGetCVec(z_vec);
    const n: usize = @intCast(x.len);
    const a_c = Complex.init(a, 0);
    const b_c = Complex.init(b, 0);
    for (0..n) |i| {
        z.data[i] = x.data[i].mul(a_c).add(y.data[i].mul(b_c));
    }
}

pub export fn N_VProd_Complex(x_vec: c.N_Vector, y_vec: c.N_Vector, z_vec: c.N_Vector) void {
    const x = N_VGetCVec(x_vec);
    const y = N_VGetCVec(y_vec);
    const z = N_VGetCVec(z_vec);
    const n: usize = @intCast(x.len);
    for (0..n) |i| {
        z.data[i] = x.data[i].mul(y.data[i]);
    }
}

pub export fn N_VDiv_Complex(x_vec: c.N_Vector, y_vec: c.N_Vector, z_vec: c.N_Vector) void {
    const x = N_VGetCVec(x_vec);
    const y = N_VGetCVec(y_vec);
    const z = N_VGetCVec(z_vec);
    const n: usize = @intCast(x.len);
    for (0..n) |i| {
        z.data[i] = x.data[i].div(y.data[i]);
    }
}

pub export fn N_VScale_Complex(c_val: c.sunrealtype, x_vec: c.N_Vector, z_vec: c.N_Vector) void {
    const x = N_VGetCVec(x_vec);
    const z = N_VGetCVec(z_vec);
    const c_c = Complex.init(c_val, 0);
    const n: usize = @intCast(x.len);
    for (0..n) |i| {
        z.data[i] = x.data[i].mul(c_c);
    }
}

pub export fn N_VAbs_Complex(x_vec: c.N_Vector, z_vec: c.N_Vector) void {
    const x = N_VGetCVec(x_vec);
    const z = N_VGetCVec(z_vec);
    const n: usize = @intCast(x.len);
    for (0..n) |i| {
        z.data[i] = Complex.init(x.data[i].magnitude(), 0.0);
    }
}

pub export fn N_VInv_Complex(x_vec: c.N_Vector, z_vec: c.N_Vector) void {
    const x = N_VGetCVec(x_vec);
    const z = N_VGetCVec(z_vec);
    const n: usize = @intCast(x.len);
    const one = Complex.init(1.0, 0.0);
    for (0..n) |i| {
        z.data[i] = one.div(x.data[i]);
    }
}

pub export fn N_VAddConst_Complex(x_vec: c.N_Vector, b: c.sunrealtype, z_vec: c.N_Vector) void {
    const x = N_VGetCVec(x_vec);
    const z = N_VGetCVec(z_vec);
    const n: usize = @intCast(x.len);
    const b_c = Complex.init(b, 0.0);
    for (0..n) |i| {
        z.data[i] = x.data[i].add(b_c);
    }
}

pub export fn N_VDotProd_Complex(x_vec: c.N_Vector, y_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const y = N_VGetCVec(y_vec);
    const n: usize = @intCast(x.len);
    var sum: f64 = 0.0;
    for (0..n) |i| {
        sum += x.data[i].re * y.data[i].re + x.data[i].im * y.data[i].im;
    }
    return sum;
}

pub export fn N_VMaxNorm_Complex(x_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const n: usize = @intCast(x.len);
    var max_val: f64 = 0.0;
    for (0..n) |i| {
        const mag = x.data[i].magnitude();
        if (mag > max_val) max_val = mag;
    }
    return max_val;
}

pub export fn N_VWSqrSum_Complex(x_vec: c.N_Vector, w_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const w = N_VGetCVec(w_vec);
    const n: usize = @intCast(x.len);
    var sum: f64 = 0.0;
    for (0..n) |i| {
        const x_mag = x.data[i].magnitude();
        const w_mag = w.data[i].magnitude();
        sum += (x_mag * w_mag) * (x_mag * w_mag);
    }
    return sum;
}

pub export fn N_VWSqrSumMask_Complex(x_vec: c.N_Vector, w_vec: c.N_Vector, id_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const w = N_VGetCVec(w_vec);
    const id = N_VGetCVec(id_vec);
    const n: usize = @intCast(x.len);
    var sum: f64 = 0.0;
    for (0..n) |i| {
        if (id.data[i].re > 0.0) {
            const x_mag = x.data[i].magnitude();
            const w_mag = w.data[i].magnitude();
            sum += (x_mag * w_mag) * (x_mag * w_mag);
        }
    }
    return sum;
}

pub export fn N_VWRMSNorm_Complex(x_vec: c.N_Vector, w_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const sqrsum = N_VWSqrSum_Complex(x_vec, w_vec);
    return @sqrt(sqrsum / @as(f64, @floatFromInt(x.len)));
}

pub export fn N_VWRMSNormMask_Complex(x_vec: c.N_Vector, w_vec: c.N_Vector, id_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const sqrsum = N_VWSqrSumMask_Complex(x_vec, w_vec, id_vec);
    return @sqrt(sqrsum / @as(f64, @floatFromInt(x.len)));
}

pub export fn N_VMin_Complex(x_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const n: usize = @intCast(x.len);
    if (n == 0) return 0.0;
    var min_val: f64 = x.data[0].re;
    for (1..n) |i| {
        if (x.data[i].re < min_val) min_val = x.data[i].re;
    }
    return min_val;
}

pub export fn N_VWL2Norm_Complex(x_vec: c.N_Vector, w_vec: c.N_Vector) c.sunrealtype {
    return @sqrt(N_VWSqrSum_Complex(x_vec, w_vec));
}

pub export fn N_VL1Norm_Complex(x_vec: c.N_Vector) c.sunrealtype {
    const x = N_VGetCVec(x_vec);
    const n: usize = @intCast(x.len);
    var sum: f64 = 0.0;
    for (0..n) |i| {
        sum += x.data[i].magnitude();
    }
    return sum;
}

pub export fn N_VInvTest_Complex(x_vec: c.N_Vector, z_vec: c.N_Vector) c_int {
    const x = N_VGetCVec(x_vec);
    const z = N_VGetCVec(z_vec);
    const n: usize = @intCast(x.len);
    var no_zero_found: c_int = 1;
    const one = Complex.init(1.0, 0.0);
    for (0..n) |i| {
        if (x.data[i].re == 0.0 and x.data[i].im == 0.0) {
            no_zero_found = 0;
        } else {
            z.data[i] = one.div(x.data[i]);
        }
    }
    return no_zero_found;
}
