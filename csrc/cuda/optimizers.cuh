template <typename T, int OPTIMIZER>
void optimizer32bit(
    T* g, T* p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm, float beta1, float beta2,
    float beta3, float alpha, float eps, float weight_decay, int step, float lr, const float gnorm_scale,
    bool skip_zeros, int n
);

template <typename T, int OPTIMIZER>
void optimizerStatic8bitBlockwise(
    T* p, T* g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3, float alpha,
    float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1, float* absmax2,
    float weight_decay, const float gnorm_scale, bool skip_zeros, int n
);
