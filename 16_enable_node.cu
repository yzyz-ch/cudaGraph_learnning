// 16：图内预先放入“乘 2 -> 加 10”，运行时决定是否跳过“加 10”。
// 开关状态：启用 -> 禁用 -> 再启用；一直使用同一个 exec。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 16_enable_node.cu -o build/16_enable_node
// 运行: ./build/16_enable_node
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        return 1;                                                             \
    }                                                                         \
} while (0)

__global__ void scale(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.f;
}
__global__ void add_ten(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 10.f;
}

int main() {
    int n = 4;
    const float input[4] = {-2.f, -1.f, 0.f, 3.f};
    const unsigned int enabled[] = {1, 0, 1};
    float output[4], *d;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    void* args[] = {&d, &n};
    cudaKernelNodeParams params{};
    params.gridDim = dim3(1);
    params.blockDim = dim3(n);
    params.kernelParams = args;
    cudaGraphNode_t first, optional;
    params.func = reinterpret_cast<void*>(scale);
    CHECK(cudaGraphAddKernelNode(&first, graph, nullptr, 0, &params));
    params.func = reinterpret_cast<void*>(add_ten);
    CHECK(cudaGraphAddKernelNode(&optional, graph, &first, 1, &params));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    bool all_ok = true;
    for (int round = 0; round < 3; ++round) {
        CHECK(cudaGraphNodeSetEnabled(exec, optional, enabled[round]));
        unsigned int actual = 0;
        CHECK(cudaGraphNodeGetEnabled(exec, optional, &actual));
        CHECK(cudaMemcpy(d, input, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = (actual == enabled[round]);
        printf("第 %d 次执行，加 10 节点%s:", round + 1, actual ? "启用" : "禁用");
        for (int i = 0; i < n; ++i) {
            printf(" %.0f", output[i]);
            if (output[i] != input[i] * 2.f + (enabled[round] ? 10.f : 0.f)) ok = false;
        }
        printf("  %s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return all_ok ? 0 : 1;
}
