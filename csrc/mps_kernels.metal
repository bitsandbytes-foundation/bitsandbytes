#include <metal_stdlib>
using namespace metal;

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

template<bool STOCHASTIC>
static unsigned char quantize_scalar(
  float rand,
  device float* code,
  float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = code[pivot];
    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = code[pivot];
    }

    if(upper_pivot == 255)
        upper = code[upper_pivot];
    if(lower_pivot == 0)
        lower = code[lower_pivot];

    if(!STOCHASTIC)
    {
      if(x > val)
      {
        float midpoint = (upper+val)*0.5f;
        if(x > midpoint)
        {
          return upper_pivot;
        }
        else
          return pivot;
      }
      else
      {
        float midpoint = (lower+val)*0.5f;
        if(x < midpoint)
          return lower_pivot;
        else
          return pivot;
      }
    }
    else
    {
      if(x > val)
      {
        float dist_to_upper = fabs(upper-x);
        float dist_full = upper-val;
        if(rand >= dist_to_upper/dist_full) return upper_pivot;
        else return pivot;
      }
      else
      {
        float dist_to_lower = fabs(lower-x);
        float dist_full = val-lower;
        if(rand >= dist_to_lower/dist_full) return lower_pivot;
        else return pivot;
      }
    }
}

kernel void quantize(device float* code [[buffer(0)]],
                      device float* A [[buffer(1)]],
                      device uchar* out [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
  const uint n_full = (NUM_BLOCK * (n / NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
  uint valid_items = (id / NUM_BLOCK + 1 == (n + NUM_BLOCK - 1) / NUM_BLOCK) ? n - (id / NUM_BLOCK * NUM_BLOCK) : NUM_BLOCK;
  const uint base_idx = (id / NUM_BLOCK * NUM_BLOCK);

  float vals[NUM];
  uchar qvals[NUM];

  for (uint i = base_idx; i < n_full; i += ((n + NUM_BLOCK - 1) / NUM_BLOCK) * NUM_BLOCK) {
    valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < valid_items; j++) {
      vals[j] = A[i + j];
    }

    for (uint j = 0; j < valid_items; j++) {
      qvals[j] = quantize_scalar<false>(0.0f, code, vals[j]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < valid_items; j++) {
      out[i + j] = qvals[j];
    }
  }
}
