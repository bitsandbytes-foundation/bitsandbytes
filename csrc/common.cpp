#include <common.h>
#include <float.h>

void quantize_block(const quantize_block_args& args) {
    // 1. find absmax in block
    // 2. divide input value by absmax to normalize into [-1.0, 1.0]
    // 3. do binary search to find the closest value
    // 4. check minimal distance
    // 5. store index

    // 1. find absmax in block
    float absmax_block = -FLT_MAX;
    for (long long i = args.block_idx; i < args.block_end; i++)
        absmax_block = fmax(absmax_block, fabs(args.A[i]));

    args.absmax[args.block_idx / args.blocksize] = absmax_block;

    for (long long i = args.block_idx; i < args.block_end; i++) {
        // 2. divide input value by absmax to normalize into [-1.0, 1.0]
        // 3. do binary search to find the closest value
        float normed_value = args.A[i] / absmax_block;
        long long idx = args.bin_searcher->scalar(normed_value);

        // 4. check minimal distance
        // The binary search returns always the value to the left, which might not be the closest value
        if (idx < 255) {
            float dist_left = fabs(normed_value - (args.code[idx]));
            float dist_right = fabs(normed_value - (args.code[idx + 1]));
            if (dist_right < dist_left) { idx += 1; }
        }

        // 5. store index
        args.out[i] = (unsigned char) idx;
    }
}
