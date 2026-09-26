// 09：图可以分成两条分支，再汇合，不一定是一条直线。
// A: a = x * 2  ──┐
//                 ├──> C: result = a + b = 3*x + 1
// B: b = x + 1  ──┘
// A、B 只读同一份输入，写不同输出；C 必须等两者都完成。
// 没有 A->B 依赖意味着允许并行，但不保证实际同时运行。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 09_graph_dependencies.cu -o build/09_graph_dependencies
// 运行: ./build/09_graph_dependencies

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

// affine：乘一个系数，再加一个常数；这里用同一个函数表达两条简单分支。
__global__ void affine(const float* x, float* y, int n, float factor, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * factor + offset;
}

__global__ void add_arrays(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) result[i] = a[i] + b[i];
}

int main() {
    int n = 4;
    const float input[4] = {1.f, 2.f, 3.f, 4.f};
    float output[4];
    float *d_x, *d_a, *d_b, *d_result;
    CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK(cudaMalloc(&d_result, n * sizeof(float)));
    CHECK(cudaMemcpy(d_x, input, n * sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));

    float two = 2.f, one = 1.f, zero = 0.f;
    void* args_a[] = {&d_x, &d_a, &n, &two, &zero};
    void* args_b[] = {&d_x, &d_b, &n, &one, &one};
    void* args_c[] = {&d_a, &d_b, &d_result, &n};
    cudaKernelNodeParams params{};
    params.gridDim = dim3(1);
    params.blockDim = dim3(n);
    params.func = reinterpret_cast<void*>(affine);
    cudaGraphNode_t a, b, c;

    // A、B 都没有前置节点；“先调用 Add A，再调用 Add B”不等于 A->B。
    params.kernelParams = args_a;
    CHECK(cudaGraphAddKernelNode(&a, graph, nullptr, 0, &params));
    params.kernelParams = args_b;
    CHECK(cudaGraphAddKernelNode(&b, graph, nullptr, 0, &params));

    // 依赖数组明确告诉 CUDA：必须完成 A 和 B，才能执行 C。
    cudaGraphNode_t dependencies[] = {a, b};
    params.func = reinterpret_cast<void*>(add_arrays);
    params.kernelParams = args_c;
    CHECK(cudaGraphAddKernelNode(&c, graph, dependencies, 2, &params));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    // 图提交到一条流，不代表图内所有节点都被强制串行；内部看依赖关系。
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(output, d_result, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    printf("A=2*x, B=x+1, C=A+B:");
    for (int i = 0; i < n; ++i) {
        printf(" %.0f", output[i]);
        if (output[i] != input[i] * 3.f + 1.f) ok = false;
    }
    printf("  %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_result));
    return ok ? 0 : 1;
}
