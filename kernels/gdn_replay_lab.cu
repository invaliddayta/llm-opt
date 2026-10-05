// Standalone replay of the exact application GDN kernel. No rollback integration.
#include "../llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(call) do { const auto e = (call); if (e != cudaSuccess) { std::fprintf(stderr, "%d %s\n", __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

constexpr int D = 128, H = 48, QH = 16, MAX_T = 16, LAYERS = 48;
constexpr size_t STATE = (size_t) D * D * H;

struct buffers {
    float * q, * k, * v, * g, * beta, * initial, * checkpoint, * snapshots, * latest, * replayed, * attention, * alternative;
    buffers() {
        const size_t q_size = (size_t) D * QH * MAX_T * LAYERS;
        const size_t v_size = (size_t) D * H * MAX_T * LAYERS;
        for (auto ptr : {&q, &k}) CK(cudaMalloc(ptr, q_size * sizeof(float)));
        for (auto ptr : {&v, &attention, &alternative}) CK(cudaMalloc(ptr, v_size * sizeof(float)));
        for (auto ptr : {&g, &beta}) CK(cudaMalloc(ptr, (size_t) H * MAX_T * LAYERS * sizeof(float)));
        for (auto ptr : {&initial, &checkpoint, &latest, &replayed}) CK(cudaMalloc(ptr, STATE * LAYERS * sizeof(float)));
        CK(cudaMalloc(&snapshots, STATE * MAX_T * LAYERS * sizeof(float)));
        std::mt19937 random(331);
        auto fill = [&](float * ptr, size_t n, float low, float high) {
            std::vector<float> values(n);
            for (auto & value : values) value = std::uniform_real_distribution<float>(low, high)(random);
            CK(cudaMemcpy(ptr, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        };
        fill(q, q_size, -0.125f, 0.125f); fill(k, q_size, -0.125f, 0.125f);
        fill(v, v_size, -1, 1); fill(g, (size_t) H * MAX_T * LAYERS, -0.2f, 0);
        fill(beta, (size_t) H * MAX_T * LAYERS, 0, 1); fill(initial, STATE * LAYERS, -0.1f, 0.1f);
    }
    ~buffers() {
        for (auto ptr : {q, k, v, g, beta, initial, checkpoint, snapshots, latest, replayed, attention, alternative}) CK(cudaFree(ptr));
    }
};

template<bool snapshots> static void launch(buffers & b, int layer, int tokens, const float * initial,
        float * state, float * output, cudaStream_t stream) {
    const size_t q_offset = (size_t) layer * D * QH * MAX_T;
    const size_t v_offset = (size_t) layer * D * H * MAX_T;
    const size_t gate_offset = (size_t) layer * H * MAX_T;
    gated_delta_net_cuda<D, false, snapshots><<<dim3(H, 1, D / 4), dim3(32, 4), 0, stream>>>(
            b.q + q_offset, b.k + q_offset, b.v + v_offset, b.g + gate_offset, b.beta + gate_offset,
            initial, output, state, H, tokens, 1,
            D, D * QH, D * QH * MAX_T, D, D * H, D * H * MAX_T,
            1, H, H * MAX_T, init_fastdiv_values(QH), init_fastdiv_values(1), 1.0f / std::sqrt(float(D)), STATE, MAX_T);
    CK(cudaGetLastError());
}

template<class F> static double timed(F run, cudaStream_t stream) {
    run(); CK(cudaStreamSynchronize(stream));
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    run();
    cudaGraph_t graph; CK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec; CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(exec, stream));
    cudaEvent_t begin, end; CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    CK(cudaEventRecord(begin, stream));
    for (int i = 0; i < 40; ++i) CK(cudaGraphLaunch(exec, stream));
    CK(cudaEventRecord(end, stream)); CK(cudaEventSynchronize(end));
    float ms; CK(cudaEventElapsedTime(&ms, begin, end));
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end)); CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
    return ms * 1000 / 40;
}

int main() {
    cudaDeviceProp props; CK(cudaGetDeviceProperties(&props, 0));
    if (props.warpSize != 32 || props.major != 8 || props.minor != 6) return 2;
    buffers b;
    cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<float> expected(STATE), actual(STATE);
    int prefixes = 0;
    for (int tokens : {1, 2, 4, 8, 16}) {
        launch<true>(b, 0, tokens, b.initial, b.snapshots, b.attention, stream);
        launch<false>(b, 0, tokens, b.initial, b.latest, b.alternative, stream);
        CK(cudaStreamSynchronize(stream));
        std::vector<float> a((size_t) D * H * tokens), c(a.size());
        CK(cudaMemcpy(a.data(), b.attention, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c.data(), b.alternative, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(a.data(), c.data(), a.size() * sizeof(float))) return 1;
        for (int accept = 0; accept <= tokens; ++accept) {
            if (accept == 0) CK(cudaMemcpyAsync(b.replayed, b.initial, STATE * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            else launch<false>(b, 0, accept, b.initial, b.replayed, b.alternative, stream);
            CK(cudaStreamSynchronize(stream));
            const auto * reference = accept == 0 ? b.initial : b.snapshots + (tokens - accept) * STATE;
            CK(cudaMemcpy(expected.data(), reference, STATE * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(actual.data(), b.replayed, STATE * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(expected.data(), actual.data(), STATE * sizeof(float))) {
                std::fprintf(stderr, "state mismatch tokens=%d accepted=%d\n", tokens, accept); return 1;
            }
            ++prefixes;
        }
    }
    auto baseline = [&]() {
        for (int l = 0; l < LAYERS; ++l) launch<true>(b, l, 8, b.initial + l * STATE,
                b.snapshots + l * MAX_T * STATE, b.attention + (size_t) l * D * H * MAX_T, stream);
    };
    const double baseline_us = timed(baseline, stream);
    for (int accept : {0, 1, 2, 3, 4, 8}) {
        auto replay = [&]() {
            for (int l = 0; l < LAYERS; ++l) {
                CK(cudaMemcpyAsync(b.checkpoint + l * STATE, b.initial + l * STATE, STATE * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                launch<false>(b, l, 8, b.initial + l * STATE, b.latest + l * STATE,
                        b.attention + (size_t) l * D * H * MAX_T, stream);
            }
            if (accept != 8) for (int l = 0; l < LAYERS; ++l) {
                if (accept == 0) CK(cudaMemcpyAsync(b.replayed + l * STATE, b.checkpoint + l * STATE, STATE * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                else launch<false>(b, l, accept, b.checkpoint + l * STATE, b.replayed + l * STATE,
                        b.alternative + (size_t) l * D * H * MAX_T, stream);
            }
        };
        const double replay_us = timed(replay, stream);
        std::printf("LAB ONLY layers=48 tokens=8 accepted=%d baseline_snapshots_us=%.2f checkpoint_verify_replay_us=%.2f\n", accept, baseline_us, replay_us);
    }
    CK(cudaStreamDestroy(stream));
    std::printf("PASS exact attention and %d accepted-prefix recurrent states; uses unchanged application kernel\n", prefixes);
    std::puts("No convolution/branching/cache integration. Synthetic timings do not establish serving throughput.");
}
