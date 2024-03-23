

using namespace cooperative_groups;

__device__ int getSign(float value) {
    return __float_as_int(value) < 0 ? 0x80000000 : 0x00000000;
}

__device__ void reduceBlock(float* sdata, int* sign_sdata, const thread_block& cgb) {
    const unsigned int tid = cgb.thread_rank();
    thread_block_tile<32> tile32 = tiled_partition<32>(cgb);

    float abs_tid = abs(sdata[tid]);
    int sign_tid = sign_sdata[tid];

    // Perform reduction within each warp using shuffle instructions
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        float abs_other = tile32.shfl_down(abs_tid, offset);
        int sign_other = tile32.shfl_down(sign_tid, offset);

        if (abs_other > abs_tid || (abs_other == abs_tid && sign_other == 0x80000000)) {
            abs_tid = abs_other;
            sign_tid = sign_other;
        }
    }

    // Collect the maximum value from each warp
    if (tile32.thread_rank() == 0) {
        sdata[tid / tile32.size()] = __int_as_float((__float_as_int(abs_tid) & 0x7FFFFFFF) | sign_tid);
    }
    cgb.sync();

    // Perform final reduction across warps
    if (tid < blockDim.x / tile32.size()) {
        for (int offset = (blockDim.x / tile32.size()) / 2; offset > 0; offset /= 2) {
            float other = sdata[tid + offset];
            float abs_other = abs(other);
            int sign_other = getSign(other);

            if (abs_other > abs_tid || (abs_other == abs_tid && sign_other == 0x80000000)) {
                abs_tid = abs_other;
                sign_tid = sign_other;
            }
        }

        sdata[tid] = __int_as_float((__float_as_int(abs_tid) & 0x7FFFFFFF) | sign_tid);
    }
    cgb.sync();
}

__global__ void reduceSinglePassMultiBlockCG(const float *g_idata, float *g_odata, unsigned int n) {
    // Handle to thread block group
    thread_block block = this_thread_block();
    grid_group grid = this_grid();

    extern float __shared__ sdata[];
    extern int __shared__ sign_sdata[];

    sdata[block.thread_rank()] = 0;
    sign_sdata[block.thread_rank()] = 0;

    float abs_sdata = 0;
    int sign_sdata_local = 0;

    for (int i = grid.thread_rank(); i < n; i += grid.size()) {
        float abs_g_idata = abs(g_idata[i]);

        if (abs_g_idata > abs_sdata || (abs_g_idata == abs_sdata && getSign(g_idata[i]) == 0x80000000)) {
            abs_sdata = abs_g_idata;
            sign_sdata_local = getSign(g_idata[i]);
        }
    }

    sdata[block.thread_rank()] = __int_as_float((__float_as_int(abs_sdata) & 0x7FFFFFFF) | sign_sdata_local);
    block.sync();

    // Reduce each block (called once per block)
    reduceBlock(sdata, sign_sdata, block);

    // Write out the result to global memory
    if (block.thread_rank() == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
    grid.sync();

    if (grid.thread_rank() == 0) {
        float max_value = g_odata[0];
        float abs_max = abs(max_value);
        int sign_max = getSign(max_value);

        for (int block = 1; block < gridDim.x; block++) {
            float value = g_odata[block];
            float abs_value = abs(value);
            int sign_value = getSign(value);

            if (abs_value > abs_max || (abs_value == abs_max && sign_value == 0x80000000)) {
                max_value = value;
                abs_max = abs_value;
                sign_max = sign_value;
            }
        }

        g_odata[0] = max_value;
    }
}
