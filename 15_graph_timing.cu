// 15：分别观察构图、实例化、首次执行、节点更新和预热后重放的耗时。
// CPU 时钟测经过的时间，CUDA event 测 GPU 时间线上的区间。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 15_graph_timing.cu -o build/15_graph_timing
// 运行: ./build/15_graph_timing
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        return 1;                                                             \
    }                                                                         \
} while (0)

using Clock = std::chrono::steady_clock;
double microseconds(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::micro>(stop - start).count();
}
__global__ void add_value(float* x, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += value;
}

int main() {
    int n = 256;
    const int steps = 20, repeats = 1000, updates = 1000;
    float *d, output[256];
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t gpu_start, gpu_end;
    CHECK(cudaEventCreate(&gpu_start));
    CHECK(cudaEventCreate(&gpu_end));
    // 提前初始化 CUDA 和 kernel；“首次执行”指这个 exec 的首次执行。
    CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
    add_value<<<1, n, 0, stream>>>(d, n, 1.f);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream));

    float value = 1.f;
    void* args[] = {&d, &n, &value};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(add_value);
    params.gridDim = dim3(1);
    params.blockDim = dim3(n);
    params.kernelParams = args;
    cudaGraph_t graph;
    cudaGraphNode_t nodes[steps];
    const auto build_start = Clock::now();
    CHECK(cudaGraphCreate(&graph, 0));
    for (int i = 0; i < steps; ++i) {
        CHECK(cudaGraphAddKernelNode(&nodes[i], graph, i ? &nodes[i - 1] : nullptr, i ? 1 : 0, &params));
    }
    const auto build_end = Clock::now();
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    const auto instantiate_end = Clock::now();

    CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
    CHECK(cudaStreamSynchronize(stream));
    const auto first_start = Clock::now();
    CHECK(cudaGraphLaunch(exec, stream));
    const auto first_submitted = Clock::now();
    CHECK(cudaStreamSynchronize(stream));
    const auto first_end = Clock::now();
    CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (float x : output) if (x != steps) ok = false;

    // 再预热 10 轮，之后才测更新和重复执行。
    for (int i = 0; i < 10; ++i) CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    const auto update_start = Clock::now();
    for (int i = 0; i < updates; ++i) {
        value = 1.f + (i % 2);  // 在 1、2 之间切换，只更新第一个节点。
        CHECK(cudaGraphExecKernelNodeSetParams(exec, nodes[0], &params));
    }
    const auto update_end = Clock::now();
    // 更新后的配置也先执行几轮，再测稳定重放；这些轮次不计时。
    for (int i = 0; i < 10; ++i) CHECK(cudaGraphLaunch(exec, stream));
    // updates=1000 时最后 value=2；其余 19 个节点始终加 1。
    CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
    CHECK(cudaStreamSynchronize(stream));
    const auto replay_start = Clock::now();
    CHECK(cudaEventRecord(gpu_start, stream));
    for (int i = 0; i < repeats; ++i) CHECK(cudaGraphLaunch(exec, stream));
    const auto replay_submitted = Clock::now();
    CHECK(cudaEventRecord(gpu_end, stream));
    CHECK(cudaEventSynchronize(gpu_end));
    const auto replay_end = Clock::now();
    float gpu_ms = 0.f;
    CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end));
    CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    const float expected = (steps - 1 + value) * repeats;
    for (float x : output) if (x != expected) ok = false;

    printf("构图: %.3f us\n", microseconds(build_start, build_end));
    printf("实例化: %.3f us\n", microseconds(build_end, instantiate_end));
    printf("首次提交: %.3f us，首次到完成: %.3f us\n",
           microseconds(first_start, first_submitted), microseconds(first_start, first_end));
    printf("单节点更新平均: %.3f us/次（%d 次）\n", microseconds(update_start, update_end) / updates, updates);
    printf("重放 %d 轮，提交阶段: %.3f ms，到全部完成: %.3f ms\n", repeats,
           microseconds(replay_start, replay_submitted) / 1000., microseconds(replay_start, replay_end) / 1000.);
    printf("GPU event 区间: %.3f ms\n", gpu_ms);
    printf("最终每个元素应为 %.0f: %s\n", expected, ok ? "PASS" : "FAIL");

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaEventDestroy(gpu_start));
    CHECK(cudaEventDestroy(gpu_end));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return ok ? 0 : 1;
}
