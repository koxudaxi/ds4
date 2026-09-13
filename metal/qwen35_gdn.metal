// Qwen3.5-MoE / Ornith 1.5 35B gated-delta-net (linear attention) decode
// kernels, derived from metal/glm53_kda.metal.
//
// The four Ornith differences from the GLM-5.3 KDA kernels:
//   1. fused attn_qkv: one depthwise causal conv over conv_dim = 8192
//      channels (2*n_k*d_k q/k plus n_v*d_v v), not three projections;
//   2. 2:1 value-head broadcast (n_k = 16 key heads, n_v = 32 value heads):
//      every key head feeds two value heads, vh % n_k;
//   3. decay g = exp(a_log * softplus(alpha + dt_bias)) with a_log the already
//      folded -exp(A_log);
//   4. RMSNormGated * silu(z) as the output gate, from a separate attn_gate
//      (z) projection.
//
// Decode only: the host runs the projection matmuls (dense) and passes the
// already-projected qkv/z/alpha/beta rows in, exactly as the GLM-5.3 KDA
// kernel does for q/k/v.

struct qwen35_gdn_args {
    uint n_rows;
    float norm_eps;
};

struct qwen35_gdn_out_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

#define QWEN35_GDN_DK       128u
#define QWEN35_GDN_DV       128u
#define QWEN35_GDN_NK       16u
#define QWEN35_GDN_NV       32u
#define QWEN35_GDN_CONV     4u
#define QWEN35_GDN_CONVDIM  8192u
#define QWEN35_GDN_DINNER   4096u

static inline float qwen35_gdn_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float qwen35_gdn_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

/*
 * Depthwise causal conv (kernel 4) over the fused attn_qkv projection,
 * followed by silu, plus the shift of the previous kconv-1 raw rows.  One
 * thread owns one channel and walks every row; each channel's conv state is
 * private, so no cross-thread synchronisation is needed.
 */
kernel void kernel_qwen35_gdn_conv(
        constant qwen35_gdn_args &args,
        device const float *qkv,
        device const float *conv1d,
        device float       *convout,
        device float       *conv_state,
        uint gid [[thread_position_in_grid]]) {
    const uint ch = gid;
    if (ch >= QWEN35_GDN_CONVDIM) return;
    for (uint r = 0; r < args.n_rows; r++) {
        const ulong base = (ulong)r * QWEN35_GDN_CONVDIM;
        float acc = 0.0f;
        for (uint kk = 0; kk + 1u < QWEN35_GDN_CONV; kk++) {
            acc = fma(conv_state[(ulong)kk * QWEN35_GDN_CONVDIM + ch],
                      conv1d[(ulong)ch * QWEN35_GDN_CONV + kk], acc);
        }
        const float x = qkv[base + ch];
        acc = fma(x, conv1d[(ulong)ch * QWEN35_GDN_CONV +
                            (QWEN35_GDN_CONV - 1u)], acc);
        convout[base + ch] = qwen35_gdn_silu(acc);
        for (uint kk = 0; kk + 1u < QWEN35_GDN_CONV; kk++) {
            conv_state[(ulong)kk * QWEN35_GDN_CONVDIM + ch] =
                conv_state[(ulong)(kk + 1u) * QWEN35_GDN_CONVDIM + ch];
        }
        conv_state[(ulong)(QWEN35_GDN_CONV - 2u) * QWEN35_GDN_CONVDIM + ch] = x;
    }
}

/*
 * One threadgroup owns one value head and walks every row: per-head L2 norm of
 * q/k for the broadcast key head, the exp-softplus decay, the delta-rule
 * recurrence over the [d_v, d_k] state, and the RMSNormGated * silu(z) output
 * gate into attn_out.  Four simdgroups update four value rows concurrently;
 * every lane owns four adjacent key columns.
 */
kernel void kernel_qwen35_gdn_core(
        constant qwen35_gdn_args &args,
        device const float *convout,
        device const float *z,
        device const float *alpha,
        device const float *beta,
        device const float *a_log,
        device const float *dt_bias,
        device const float *norm,
        device float       *attn_out,
        device float       *state,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint vh [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint DK = QWEN35_GDN_DK;
    constexpr uint DV = QWEN35_GDN_DV;
    constexpr uint NK = QWEN35_GDN_NK;
    constexpr uint NV = QWEN35_GDN_NV;
    constexpr uint CD = QWEN35_GDN_CONVDIM;
    constexpr uint DI = QWEN35_GDN_DINNER;
    if (vh >= NV) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + DK;
    threadgroup float *so = sk + DK;
    threadgroup float *reduce_q = so + DV;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *reduce_o = reduce_k + 4u;

    const uint kh = vh % NK;
    const uint q_offset = kh * DK;
    const uint k_offset = NK * DK + kh * DK;
    const uint v_offset = 2u * NK * DK + vh * DV;
    const uint k0 = lane * 4u;
    device float *state_head = state + (ulong)vh * DV * DK;

    const float a = a_log[vh];
    const float dtb = dt_bias[vh];

    for (uint r = 0; r < args.n_rows; r++) {
        const ulong base = (ulong)r * CD;
        if (tid < DK) {
            sq[tid] = convout[base + q_offset + tid];
            sk[tid] = convout[base + k_offset + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        if (tid < DK) {
            sq[tid] *= rsqrt(q_total + 1.0e-6f) * 0.08838834764831845f;
            sk[tid] *= rsqrt(k_total + 1.0e-6f);
        }

        const float gate = a * qwen35_gdn_softplus(
            alpha[(ulong)r * NV + vh] + dtb);
        const float g = exp(gate);
        const float b = 1.0f / (1.0f + exp(-beta[(ulong)r * NV + vh]));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float4 q4 = *((threadgroup float4 *)(sq + k0));
        const float4 k4 = *((threadgroup float4 *)(sk + k0));

        for (uint value = sg; value < DV; value += 4u) {
            device float4 *hptr = (device float4 *)(
                state_head + (ulong)value * DK + k0);
            float4 h = (*hptr) * g;
            const float hk = simd_sum(dot(h, k4));
            const float delta = (convout[base + v_offset + value] - hk) * b;
            h = fma(k4, float4(delta), h);
            *hptr = h;
            const float hq = simd_sum(dot(h, q4));
            if (lane == 0u) so[value] = hq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup |
                            mem_flags::mem_device);

        float o_sumsq = simd_sum(so[tid] * so[tid]);
        if (lane == 0u) reduce_o[sg] = o_sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
        o_total = simd_sum(o_total);
        const float o_scale = rsqrt(o_total / (float)DV + args.norm_eps);
        if (tid < DV) {
            const ulong index = (ulong)r * DI + vh * DV + tid;
            attn_out[index] = so[tid] * o_scale * norm[tid] *
                              qwen35_gdn_silu(z[index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup |
                            mem_flags::mem_device);
    }
}

/* ssm_out: the output projection from d_inner back to n_embd. */
kernel void kernel_qwen35_gdn_out_proj(
        constant qwen35_gdn_out_args &args,
        device const float *weight,
        device const float *x,
        device float       *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint o = tgpig.y * 128u + tid;
    if (row >= args.n_rows || o >= args.out_dim) return;
    const ulong wbase = (ulong)o * args.in_dim;
    const ulong xbase = (ulong)row * args.in_dim;
    float acc = 0.0f;
    for (uint i = 0; i < args.in_dim; i++) {
        acc = fma(weight[wbase + i], x[xbase + i], acc);
    }
    out[(ulong)row * args.out_dim + o] = acc;
}
