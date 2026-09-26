// 05：把“加 1”改成“加 10”，只更新一个 kernel 节点。
// 先看三轮输出：原参数 -> 只改 CPU 变量 -> 调用更新 API。
// 本节手动添加一个节点，直接保留 node（节点句柄，即操作的标识）。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 05_update_kernel_node.cu -o build/05_update_kernel_node
// 运行: ./build/05_update_kernel_node

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

__global__ void add_value(float* x, int n, float delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += delta;
}

int main() {
    int n = 4;
    const float input[4] = {1.f, 2.f, 3.f, 4.f};
    float output[4];
    float delta = 1.f;
    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // params 对应普通 launch 的“函数、grid、block、参数”。
    // args 放的是 CPU 上各参数变量的地址；顺序与 add_value 的形参一致。
    // &d 让 CUDA 读取指针 d 的值，并不是让 GPU 去使用 &d 这个 CPU 地址。
    void* args[] = {&d, &n, &delta};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(add_value);
    params.gridDim = dim3(1);
    params.blockDim = dim3(n);
    params.kernelParams = args;

    cudaGraph_t graph;
    cudaGraphNode_t node;
    CHECK(cudaGraphCreate(&graph, 0));
    // nullptr, 0：这个节点没有前置依赖。添加时 CUDA 会复制参数值。
    CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    const char* labels[] = {"original", "CPU variable only", "exec updated"};
    bool all_ok = true;
    for (int round = 0; round < 3; ++round) {
        if (round == 1) delta = 10.f;  // 仅修改 CPU 变量，exec 中仍然是 1。
        if (round == 2) {
            // 此时才把 args 指向的最新参数值交给 exec，影响之后的重放。
            CHECK(cudaGraphExecKernelNodeSetParams(exec, node, &params));
        }
        // 每轮输入相同，排除累积计算对观察的干扰。
        CHECK(cudaMemcpy(d, input, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        const float expected_delta = round < 2 ? 1.f : delta;
        bool ok = true;
        printf("%s (CPU delta=%.0f):", labels[round], delta);
        for (int i = 0; i < n; ++i) {
            printf(" %.0f", output[i]);
            if (output[i] != input[i] + expected_delta) ok = false;
        }
        printf("  %s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    // 保留原 graph 到更新完成，因为上面的 node 来自它。
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return all_ok ? 0 : 1;
}
