// Qwen3.5-MoE / Ornith 1.5 35B full-attention support kernels.
//
// The CPU full-attention core (ds4.c, qwen35_attn_core) is the golden
// reference:
//   * the joint query+gate projection is [n_head][2*head_dim] with the query
//     in the first head_dim of each head and the sigmoid output gate in the
//     second (interleaved per head);
//   * per-head weighted RMSNorm normalises only the query half, at a stride
//     of 2*head_dim, so a contiguous weighted row norm is wrong;
//   * K is contiguous per KV head and uses attn_k_norm;
//   * partial RoPE is Neox half-pairing over the FIRST n_rot dims of each
//     head -- pair (p, p + n_rot/2), rotated by
//     pos * freq_base^(-2p/n_rot) -- which is what llama.cpp's
//     ggml_rope_multi produces for qwen35moe's text-only position.
//
// These kernels are in-place and operate on explicit strides so the same
// launch serves both the interleaved query (stride 2*head_dim) and the
// contiguous K (stride head_dim).

struct qwen35_head_norm_args {
    int32_t  n;       /* normalised width (head_dim) */
    int32_t  n4;      /* n / 4 */
    uint64_t stride;  /* bytes between head rows */
    float    eps;
};

// Weighted RMSNorm over the first `n` floats of each strided head row.  The
// reduction and scale mirror kernel_rms_norm_fuse_impl (and the CPU
// rms_norm_weight): mean = sum/n, scale = 1/sqrt(mean + eps), y = x*scale*w.
kernel void kernel_qwen35_head_rms_norm_f32_4(
        constant qwen35_head_norm_args & args,
        device const char * weight,
        device       char * x,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const uint row = tgpig.x;
    device const float4 * w = (device const float4 *) weight;
    device       float4 * y = (device       float4 *) (x + (uint64_t)row * args.stride);

    float sumf = 0.0f;
    for (int i = tpitg.x; i < args.n4; i += ntg.x) {
        const float4 v = y[i];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f / sqrt(sumf / float(args.n) + args.eps);
    for (int i = tpitg.x; i < args.n4; i += ntg.x) {
        y[i] = (y[i] * scale) * w[i];
    }
}

struct qwen35_rope_front_args {
    int32_t  n_head;
    int32_t  head_dim;
    int32_t  n_rot;
    int32_t  n_half;
    uint64_t head_stride;  /* bytes between heads within a token */
    uint64_t row_stride;   /* bytes between tokens */
    int32_t  pos0;
    int32_t  pos_step;
    float    freq_base;
};

// Partial Neox half-pair RoPE over the first n_rot dims of each head, in
// place.  One thread rotates one pair; pair p becomes
// (x[p]*c - x[p+half]*s, x[p]*s + x[p+half]*c) with
// theta = (pos0 + t*pos_step) * freq_base^(-2p/n_rot).  Dims [n_rot, head_dim)
// are never read or written.
kernel void kernel_qwen35_rope_neox_front_f32(
        constant qwen35_rope_front_args & args,
        device       char * x,
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const int h = tgpig[0];
    const int t = tgpig[1];
    if (h >= args.n_head || tid >= args.n_half) {
        return;
    }

    device float * head = (device float *)
        (x + (uint64_t)t * args.row_stride + (uint64_t)h * args.head_stride);

    const float pos = (float)(args.pos0 + t * args.pos_step);
    const float theta = pos * pow(args.freq_base, -2.0f * (float)tid / (float)args.n_rot);
    const float c = cos(theta);
    const float s = sin(theta);
    const int j0 = tid;
    const int j1 = tid + args.n_half;
    const float x0 = head[j0];
    const float x1 = head[j1];

    head[j0] = x0 * c - x1 * s;
    head[j1] = x0 * s + x1 * c;
}

// Qwen3.5-MoE router.  Matches ds4.c's qwen35_moe_route exactly: a max-
// subtracted softmax over ALL experts, top-k selection by probability (ties
// keep the lower expert index), then renormalisation of the selected k
// weights.  This is a distinct convention from the DeepSeek (sigmoid) and GLM
// (softplus) routers.
//
// One threadgroup per token.  The softmax numerator and every selection round
// are computed in parallel across the threadgroup with a max/argmax tree
// reduction, replacing the old lane-0 O(n*k^2) selection sort that stalled the
// whole GPU at one threadgroup and dominated decode.  The reduction orders by
// (value descending, index ascending), so a tie keeps the lower expert index
// exactly as the CPU selection sort does.  Only the final k weight writes and
// their denominator sum stay on lane 0, in the same order as before, so the
// arithmetic is bit-identical to the committed kernel.
struct qwen35_route_args {
    int32_t n_expert;
    int32_t n_expert_used;
    float   expert_weight_scale;
    int32_t pad0;
};

// Value/index pair reduction in favour of the larger value; on an exact tie the
// lower index wins.  `width` must be the launched threadgroup size (a power of
// two), so every thread, including inactive ones, reaches the barriers.
static inline int qwen35_route_reduce(
        threadgroup float * red_val,
        threadgroup int   * red_idx,
        uint tid, uint width) {
    for (uint step = width >> 1; step > 0u; step >>= 1) {
        if (tid < step) {
            const float v0 = red_val[tid];
            const float v1 = red_val[tid + step];
            const int   i0 = red_idx[tid];
            const int   i1 = red_idx[tid + step];
            if (v1 > v0 || (v1 == v0 && i1 < i0)) {
                red_val[tid] = v1;
                red_idx[tid] = i1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return red_idx[0];
}

kernel void kernel_qwen35_moe_route(
        constant qwen35_route_args & args,
        device const float * logits,
        device int32_t     * selected,
        device float       * weights,
        threadgroup float  * scratch [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint tid = tpitg.x;
    const int  n   = args.n_expert;
    const int  k   = args.n_expert_used;
    const uint width = n > 256 ? 512u : 256u;

    // scratch layout: [width] exp values, [width] reduction values,
    //                 [width] reduction indices (reinterpreted as int32).
    threadgroup float *vals    = scratch;
    threadgroup float *red_val = scratch + width;
    threadgroup int   *red_idx = (threadgroup int *)(scratch + 2u * width);

    device const float *row_logits   = logits   + (uint64_t)row * n;
    device int32_t     *row_selected = selected + (uint64_t)row * k;
    device float       *row_weights  = weights  + (uint64_t)row * k;

    // Global max over the expert row (exact: max is order-independent).
    const bool active = tid < (uint) n;
    red_val[tid] = active ? row_logits[tid] : -INFINITY;
    red_idx[tid] = (int) tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (void) qwen35_route_reduce(red_val, red_idx, tid, width);
    const float m = red_val[0];

    // Max-subtracted softmax numerator, matching the CPU's float32 exp().
    vals[tid] = active ? exp(row_logits[tid] - m) : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-k by repeated parallel argmax.  A winner is struck from `vals` so it
    // cannot repeat; the struck -INFINITY is below every real exp (the max
    // expert is exactly 1.0), and ties resolve to the lower index.
    int   sel[32];
    float selval[32];
    for (int t = 0; t < k; t++) {
        red_val[tid] = vals[tid];
        red_idx[tid] = (int) tid;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int best = qwen35_route_reduce(red_val, red_idx, tid, width);
        if (tid == 0) {
            sel[t] = best;
            selval[t] = red_val[0];
            vals[best] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float wsum = 0.0f;
        for (int t = 0; t < k; t++) wsum += selval[t];
        if (!(wsum > 0.0f)) wsum = 1.0f;
        for (int t = 0; t < k; t++) {
            row_selected[t] = sel[t];
            row_weights[t] = selval[t] / wsum * args.expert_weight_scale;
        }
    }
}
