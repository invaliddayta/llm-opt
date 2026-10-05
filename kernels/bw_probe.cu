// Measure achievable DRAM read bandwidth on this GPU (the roofline for weight streaming).
#include <cstdio>
#include <cuda_runtime.h>

__global__ void read_kernel(const int4 * __restrict__ p, size_t n, int4 * out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int4 v = __ldg(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0x12345678) out[0] = acc;
}

int main() {
    const size_t bytes = (size_t) 1 << 30;
    int4 * p; int4 * out;
    cudaMalloc(&p, bytes); cudaMalloc(&out, 16);
    cudaMemset(p, 1, bytes);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    for (int bpsm : {2, 4, 8, 16}) {
        for (int it = 0; it < 2; ++it) {
            cudaEventRecord(a);
            for (int r = 0; r < 5; ++r) read_kernel<<<sms * bpsm, 256>>>(p, bytes / 16, out);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            if (it == 1) printf("blocks/SM %2d: %.1f GB/s\n", bpsm, 5.0 * (double) bytes / (ms * 1e6));
        }
    }
    return 0;
}
