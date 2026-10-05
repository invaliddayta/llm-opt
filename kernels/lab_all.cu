// Lab driver for mmsq-kernels.cuh: correctness vs CPU reference and bandwidth for IQ4_XS / Q4_K / Q5_K / Q6_K.
// usage: lab_all [check|bench] [type...]
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../llama.cpp/ggml/src/ggml-cuda/mmsq-kernels.cuh"

#ifndef KW_H
#define KW_H 8
#endif
#ifndef ST_H
#define ST_H 2
#endif
#ifndef MINB_H
#define MINB_H 2
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

using mmsq::qtype;
static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }
static uint16_t f2h(float f) { __half x = __float2half(f); uint16_t h; memcpy(&h, &x, 2); return h; }

static void scale_min(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static void dequant(qtype t, const uint8_t * b, float * y) {  // one super-block -> 256 floats
    if (t == qtype::iq4_xs) {
        const float d = h2f(*(const uint16_t *) b);
        const uint16_t sh = *(const uint16_t *) (b + 2);
        const uint8_t * sl = b + 4, * qs = b + 8;
        for (int j = 0; j < 8; ++j) {
            const int ls = ((sl[j / 2] >> (4 * (j % 2))) & 0xF) | (((sh >> (2 * j)) & 3) << 4);
            for (int i = 0; i < 16; ++i) {
                y[32 * j + i] = d * (ls - 32) * kv[qs[16 * j + i] & 0xF];
                y[32 * j + 16 + i] = d * (ls - 32) * kv[qs[16 * j + i] >> 4];
            }
        }
    } else if (t == qtype::q4_K || t == qtype::q5_K) {
        const float d = h2f(*(const uint16_t *) b), dmin = h2f(*(const uint16_t *) (b + 2));
        const uint8_t * sc = b + 4;
        const uint8_t * qh = b + 16;
        const uint8_t * ql = t == qtype::q5_K ? b + 48 : b + 16;
        for (int j = 0; j < 8; ++j) {
            int s, m; scale_min(j, sc, s, m);
            for (int l = 0; l < 32; ++l) {
                int q = (ql[32 * (j / 2) + l] >> (4 * (j & 1))) & 0xF;
                if (t == qtype::q5_K) q |= ((qh[l] >> j) & 1) << 4;
                y[32 * j + l] = d * s * q - dmin * m;
            }
        }
    } else {
        const uint8_t * ql = b, * qh = b + 128;
        const int8_t * sc = (const int8_t *) (b + 192);
        const float d = h2f(*(const uint16_t *) (b + 208));
        for (int n = 0; n < 2; ++n) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                y[128 * n + l + 0]  = d * sc[8 * n + is + 0] * (((ql[64 * n + l] & 0xF)      | (((qh[32 * n + l] >> 0) & 3) << 4)) - 32);
                y[128 * n + l + 32] = d * sc[8 * n + is + 2] * (((ql[64 * n + l + 32] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4)) - 32);
                y[128 * n + l + 64] = d * sc[8 * n + is + 4] * (((ql[64 * n + l] >> 4)       | (((qh[32 * n + l] >> 4) & 3) << 4)) - 32);
                y[128 * n + l + 96] = d * sc[8 * n + is + 6] * (((ql[64 * n + l + 32] >> 4)  | (((qh[32 * n + l] >> 6) & 3) << 4)) - 32);
            }
        }
    }
}

static int blk_bytes(qtype t) {
    switch (t) { case qtype::iq4_xs: return 136; case qtype::q4_K: return 144; case qtype::q5_K: return 176; default: return 210; }
}

template <qtype T, int NT, bool baseline = false>
static void launch_t(const uint8_t * W, int64_t row_bytes, const uint8_t * XF, float * out, int M, int K, int N, int64_t sdc, float * part, int * cnt, int splits, int sbps, cudaStream_t stream) {
    constexpr int kw = baseline ? 2 : KW_H, stages = baseline ? 2 : ST_H, minb = baseline ? 8 : MINB_H;
    constexpr int smem = mmsq::cfg<T, kw, stages>::SMEM;
    if constexpr (smem > 48 * 1024) CK(cudaFuncSetAttribute(mmsq::mul_mat<T, NT, kw, minb, stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    mmsq::mul_mat<T, NT, kw, minb, stages><<<dim3((M + 15) / 16, splits), kw * 32, smem, stream>>>(W, row_bytes, XF, out, M, K, N, sdc, part, cnt, sbps);
    CK(cudaGetLastError());
}

template <qtype T, bool baseline = false>
static void launch(const uint8_t * W, int64_t row_bytes, const uint8_t * XF, float * out, int M, int K, int N, int64_t sdc, float * ssplit, int * cnt, int splits, int sbps, cudaStream_t stream) {
    if (N <= 8) launch_t<T, 1, baseline>(W, row_bytes, XF, out, M, K, N, sdc, ssplit, cnt, splits, sbps, stream);
    else        launch_t<T, 2, baseline>(W, row_bytes, XF, out, M, K, N, sdc, ssplit, cnt, splits, sbps, stream);
}

int main(int argc, char ** argv) {
    const bool check = argc > 1 && std::string(argv[1]) == "check";
    std::vector<qtype> types;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        types.push_back(a == "iq4_xs" ? qtype::iq4_xs : a == "q4_K" ? qtype::q4_K : a == "q5_K" ? qtype::q5_K : qtype::q6_K);
    }
    if (types.empty()) types = {qtype::iq4_xs, qtype::q4_K, qtype::q5_K, qtype::q6_K};
    if (getenv("EXACT_BASELINE") && (KW_H != 2 || std::any_of(types.begin(), types.end(), [](qtype t) {
            return t != qtype::iq4_xs && t != qtype::q5_K;
        }))) {
        std::fprintf(stderr, "EXACT_BASELINE requires KW=2 and IQ4_XS/Q5_K only\n");
        return 2;
    }
    const char * tn[] = {"iq4_xs", "q4_K", "q5_K", "q6_K"};
    struct Shape { int M, K; };
    std::vector<Shape> shapes = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {6144, 5120}, {5120, 6144}, {1024, 5120}};
    if (getenv("WEAK_SHAPES")) shapes = {{6144, 5120}, {1024, 5120}};
    if (getenv("SHAPES")) {
        shapes.clear();
        for (char * t = strtok(getenv("SHAPES"), ","); t; t = strtok(nullptr, ",")) {
            int m, k;
            if (sscanf(t, "%dx%d", &m, &k) != 2 || m <= 0 || k <= 0 || k % 256 != 0) {
                std::fprintf(stderr, "bad SHAPES entry '%s' (want MxK, M,K > 0, K %% 256 == 0)\n", t);
                return 2;
            }
            shapes.push_back({m, k});
        }
        if (shapes.empty()) { std::fprintf(stderr, "SHAPES is empty\n"); return 2; }
    }
    std::vector<int> Ns = {1, 2, 4, 8, 12, 16};
    if (getenv("NS")) {
        Ns.clear();
        for (char * t = strtok(getenv("NS"), ","); t; t = strtok(nullptr, ",")) {
            const int n = atoi(t);
            if (n < 1 || n > 16) { std::fprintf(stderr, "bad NS entry '%s' (want 1..16)\n", t); return 2; }
            Ns.push_back(n);
        }
        if (Ns.empty()) { std::fprintf(stderr, "NS is empty\n"); return 2; }
    }
    const int rounds = getenv("ROUNDS") ? atoi(getenv("ROUNDS")) : 40;
    if (rounds < 1) { std::fprintf(stderr, "ROUNDS must be >= 1\n"); return 2; }
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    std::mt19937 rng(1);
    int fails = 0;
    cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int * dcnt; CK(cudaMalloc(&dcnt, 65536 * 4)); CK(cudaMemsetAsync(dcnt, 0, 65536 * 4, stream));

    for (qtype T : types) {
        double agg_b[17] = {0}, agg_us[17] = {0};
        const int bb = blk_bytes(T);
        for (auto sh : shapes) {
            const int M = sh.M, K = sh.K, nsb = K / 256;
            const int64_t row_bytes = (int64_t) nsb * bb;
            std::vector<uint8_t> hw((size_t) M * row_bytes);
            for (auto & x : hw) x = (uint8_t) rng();
            for (int64_t r = 0; r < M; ++r) for (int s = 0; s < nsb; ++s) {
                uint8_t * b = hw.data() + r * row_bytes + (int64_t) s * bb;
                const float d = std::uniform_real_distribution<float>(0.0005f, 0.004f)(rng);
                if (T == qtype::q6_K) { *(uint16_t *) (b + 208) = f2h(d); }
                else { *(uint16_t *) b = f2h(d); }
                if (T == qtype::q4_K || T == qtype::q5_K) *(uint16_t *) (b + 2) = f2h(d * 2);
            }
            const int copies = getenv("COLD_WEIGHTS") ? std::max<size_t>(1, (32 * 1024 * 1024 + hw.size() - 1) / hw.size()) : 1;
            uint8_t * dW; CK(cudaMalloc(&dW, hw.size() * copies));
            for (int copy = 0; copy < copies; ++copy) CK(cudaMemcpyAsync(dW + copy * hw.size(), hw.data(), hw.size(), cudaMemcpyHostToDevice, stream));
            CK(cudaStreamSynchronize(stream));

            for (int N : Ns) {
                std::vector<float> hx((size_t) N * K);
                for (auto & v : hx) v = std::normal_distribution<float>(0, 1)(rng);
                float * dx; CK(cudaMalloc(&dx, hx.size() * 4));
                CK(cudaMemcpyAsync(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice, stream));
                const int nt = (N + 7) / 8;
                uint8_t * dXF; CK(cudaMalloc(&dXF, (size_t) nt * nsb * mmsq::FSB));
                mmsq::quantize_x<<<dim3(nsb, nt), 32, 0, stream>>>(dx, dXF, K, N, K);
                CK(cudaGetLastError());
                CK(cudaStreamSynchronize(stream));

                const int ctas_m = (M + 15) / 16;
                int splits = 1;
                const int split_target = getenv("SPLIT_TARGET") ? atoi(getenv("SPLIT_TARGET")) : 4;
                while (ctas_m * splits < split_target * sms && nsb / (splits * 2) >= KW_H) splits *= 2;
                const int sbps = ((nsb + splits - 1) / splits + KW_H - 1) / KW_H * KW_H;
                splits = (nsb + sbps - 1) / sbps;
                float * dpart; CK(cudaMalloc(&dpart, (size_t) splits * N * M * 4));
                float * dD; CK(cudaMalloc(&dD, (size_t) N * M * 4));
                int weight_copy = 0;
                auto run = [&]() {
                    const uint8_t * weights = dW + (weight_copy++ % copies) * hw.size();
                    switch (T) {
                        case qtype::iq4_xs: launch<qtype::iq4_xs>(weights, row_bytes, dXF, dD, M, K, N, M, dpart, dcnt, splits, sbps, stream); break;
                        case qtype::q4_K:   launch<qtype::q4_K>  (weights, row_bytes, dXF, dD, M, K, N, M, dpart, dcnt, splits, sbps, stream); break;
                        case qtype::q5_K:   launch<qtype::q5_K>  (weights, row_bytes, dXF, dD, M, K, N, M, dpart, dcnt, splits, sbps, stream); break;
                        case qtype::q6_K:   launch<qtype::q6_K>  (weights, row_bytes, dXF, dD, M, K, N, M, dpart, dcnt, splits, sbps, stream); break;
                    }
                };
                run();
                CK(cudaDeviceSynchronize());
                if (check) {
                    // reference against the unquantized f32 activations (so it also covers the activation quantizer)
                    std::vector<float> hd((size_t) N * M);
                    CK(cudaMemcpy(hd.data(), dD, hd.size() * 4, cudaMemcpyDeviceToHost));
                    std::vector<float> w((size_t) K);
                    double err2 = 0, ref2 = 0;
                    for (int m = 0; m < M; m += M / 97 + 1) {
                        for (int s = 0; s < nsb; ++s) dequant(T, hw.data() + (int64_t) m * row_bytes + (int64_t) s * bb, w.data() + 256 * s);
                        for (int n = 0; n < N; ++n) {
                            double ref = 0;
                            for (int k = 0; k < K; ++k) ref += (double) w[k] * hx[(size_t) n * K + k];
                            const double e = ref - hd[(size_t) n * M + m];
                            err2 += e * e; ref2 += ref * ref;
                        }
                    }
                    const double nmse = err2 / ref2;
                    const bool ok = nmse < 1e-3;   // int8 activations with per-256 scales
                    fails += !ok;
                    printf("%-7s M=%5d K=%5d N=%2d splits=%d NMSE %.2e %s\n", tn[(int) T], M, K, N, splits, nmse, ok ? "OK" : "FAIL");
                    if (getenv("EXACT_BASELINE")) {
                        float * reference;
                        CK(cudaMalloc(&reference, hd.size() * sizeof(float)));
                        if (T == qtype::iq4_xs) launch<qtype::iq4_xs, true>(dW, row_bytes, dXF, reference, M, K, N, M, dpart, dcnt, splits, sbps, stream);
                        else                   launch<qtype::q5_K, true>(dW, row_bytes, dXF, reference, M, K, N, M, dpart, dcnt, splits, sbps, stream);
                        CK(cudaStreamSynchronize(stream));
                        std::vector<float> baseline(hd.size());
                        CK(cudaMemcpy(baseline.data(), reference, hd.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        size_t differences = 0;
                        for (size_t i = 0; i < hd.size(); ++i) differences += std::memcmp(&hd[i], &baseline[i], sizeof(float)) != 0;
                        fails += differences != 0;
                        printf("EXACT_BASELINE %-7s M=%5d K=%5d N=%2d compared=%zu differences=%zu %s\n",
                                tn[(int) T], M, K, N, hd.size(), differences, differences ? "FAIL" : "OK");
                        CK(cudaFree(reference));
                    }
                } else {
                    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                    const int reps = copies * 8;
                    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    for (int r = 0; r < reps; ++r) run();
                    cudaGraph_t graph;
                    CK(cudaStreamEndCapture(stream, &graph));
                    cudaGraphExec_t exec;
                    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                    for (int r = 0; r < 5; ++r) CK(cudaGraphLaunch(exec, stream));
                    // min over rounds: other GPU work only adds time
                    std::vector<double> t(rounds);
                    for (int r = 0; r < rounds; ++r) {
                        CK(cudaEventRecord(a, stream));
                        CK(cudaGraphLaunch(exec, stream));
                        CK(cudaEventRecord(b, stream)); CK(cudaEventSynchronize(b));
                        float ms; CK(cudaEventElapsedTime(&ms, a, b));
                        t[r] = 1000.0 * ms / reps;
                    }
                    std::sort(t.begin(), t.end());
                    const double us = t[0], med = t[rounds / 2];
                    agg_b[N] += (double) hw.size(); agg_us[N] += us;
                    if (!getenv("QUIET")) printf("%-7s M=%5d K=%5d N=%2d splits=%d %8.1f us %6.1f GB/s (median %6.1f GB/s)\n", tn[(int) T], M, K, N, splits, us, hw.size() / (us * 1e3), hw.size() / (med * 1e3));
                    CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
                    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
                }
                cudaFree(dx); cudaFree(dXF); cudaFree(dpart); cudaFree(dD);
            }
            cudaFree(dW);
        }
        if (!check) {
            printf("AGG %-7s KW=%d MINB=%d ST=%d :", tn[(int) T], KW_H, MINB_H, ST_H);
            for (int N : Ns) printf(" N%d=%.0f", N, agg_b[N] / (agg_us[N] * 1e3));
            printf("\n");
        }
    }
    if (check) printf("%s (%d failures)\n", fails ? "FAILED" : "ALL OK", fails);
    CK(cudaFree(dcnt)); CK(cudaStreamDestroy(stream));
    return fails != 0;
}
