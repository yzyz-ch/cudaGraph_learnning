// 入门：相同的小 kernel 工作量，比较普通提交与整图提交的重复执行时间。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 -o build/03_launch_vs_graph 03_launch_vs_graph.cu
// 运行: ./build/03_launch_vs_graph

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(error));                                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.f;
}

// 普通路径和捕获路径共用这个简单流程，保证每轮工作一致。
void launch_sequence(float* d, int n, int steps, cudaStream_t stream) {
    for (int k = 0; k < steps; ++k) {
        add_one<<<(n + 255) / 256, 256, 0, stream>>>(d, n);
    }
}

int main() {
    const int n = 256;
    const int steps = 1;    // 一轮包含 20 个 kernel，每个都给所有元素加 1。
    const int repeats = 1000;  // 重复 1000 轮，最终每个元素应为 20000。
    const int warmup = 10;     // 预热：先执行几轮，排除首次执行的准备成本。
    float* d = nullptr;
    float output[n];
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_sequence(d, n, steps, stream);
    CHECK(cudaGetLastError());
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    double us_per_round[2];
    bool ok = true;
    for (int mode = 0; mode < 2; ++mode) {
        const bool use_graph = (mode == 1);
        const char* name = use_graph ? "graph" : "normal";
        CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
        for (int i = 0; i < warmup; ++i) {
            if (use_graph) {
                CHECK(cudaGraphLaunch(exec, stream));
            } else {
                launch_sequence(d, n, steps, stream);
            }
        }
        CHECK(cudaGetLastError());
        CHECK(cudaStreamSynchronize(stream));

        // 两种方式都从零开始；清零和等待清零完成不计时。
        CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
        CHECK(cudaStreamSynchronize(stream));
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) {
            if (use_graph) {
                CHECK(cudaGraphLaunch(exec, stream));
            } else {
                launch_sequence(d, n, steps, stream);
            }
        }
        CHECK(cudaGetLastError());
        // 只在整批结束时同步；计时包含 CPU 提交及等待 GPU 完成。
        CHECK(cudaStreamSynchronize(stream));
        const auto stop = std::chrono::steady_clock::now();
        const double total_us = std::chrono::duration<double, std::micro>(stop - start).count();
        us_per_round[mode] = total_us / repeats;

        // 结果复制与校验不在计时区间内。
        CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool mode_ok = true;
        for (int i = 0; i < n; ++i) {
            if (output[i] != static_cast<float>(steps * repeats)) mode_ok = false;
        }
        ok = ok && mode_ok;
        printf("%s: %.3f us/round, %s (expected each element: %d)\n",
               name, us_per_round[mode], mode_ok ? "PASS" : "FAIL", steps * repeats);
    }
    printf("speedup (normal / graph): %.2fx\n", us_per_round[0] / us_per_round[1]);
    printf("Timing excludes graph construction, allocation and data copies.\n");

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return ok ? 0 : 1;
}
