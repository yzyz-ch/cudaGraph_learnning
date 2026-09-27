// 10：在 09 的分支图上，把 B 的 kernel 从 ReLU 换成求立方。
// 第一次：A = 2*x，B = max(x, 0)，C = A+B。
// 第二次：A = 2*x，B = x*x*x，    C = A+B。
// 只构图、实例化一次；更新 B 的函数后，再运行同一个 exec。
// B 前后都是 Kernel 节点，节点类型和 A/B -> C 的依赖都没变。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 10_update_kernel_function.cu -o build/10_update_kernel_function
// 运行: ./build/10_update_kernel_function

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

__global__ void scale(const float* x, float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = x[i] * 2.f;
}

__global__ void relu(const float* x, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = x[i] > 0.f ? x[i] : 0.f;
}

// 新 kernel：读取原始输入 x，计算立方，仍然写入 b。
// 与 relu 的参数顺序和类型一致，所以可沿用 B 原来的参数列表。
__global__ void cube(const float* x, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = x[i] * x[i] * x[i];
}

__global__ void add_arrays(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) result[i] = a[i] + b[i];
}

int main() {
    int n = 4;
    const float input[4] = {-2.f, -1.f, 0.f, 3.f};
    float output_a[4], output_b[4], output[4];
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

    void* args_a[] = {&d_x, &d_a, &n};
    void* args_b[] = {&d_x, &d_b, &n};
    void* args_c[] = {&d_a, &d_b, &d_result, &n};
    cudaKernelNodeParams params{};
    params.gridDim = dim3(1);
    params.blockDim = dim3(n);
    cudaGraphNode_t a, b, c;

    params.func = reinterpret_cast<void*>(scale);
    params.kernelParams = args_a;
    CHECK(cudaGraphAddKernelNode(&a, graph, nullptr, 0, &params));

    params.func = reinterpret_cast<void*>(relu);
    params.kernelParams = args_b;
    // 单独保存 B 的完整参数描述，之后配置 C 时不会把它覆盖。
    cudaKernelNodeParams b_params = params;
    CHECK(cudaGraphAddKernelNode(&b, graph, nullptr, 0, &b_params));

    cudaGraphNode_t dependencies[] = {a, b};
    params.func = reinterpret_cast<void*>(add_arrays);
    params.kernelParams = args_c;
    CHECK(cudaGraphAddKernelNode(&c, graph, dependencies, 2, &params));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    bool all_ok = true;
    for (int round = 0; round < 2; ++round) {
        if (round == 1) {
            // 第一次已同步完成。现在只把 exec 中 B 的函数换成 cube。
            // 只改 b_params.func 还不够，下一行 API 才真正更新 exec。
            b_params.func = reinterpret_cast<void*>(cube);
            CHECK(cudaGraphExecKernelNodeSetParams(exec, b, &b_params));
        }
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(output_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(output, d_result, n * sizeof(float), cudaMemcpyDeviceToHost));

        printf("第 %d 次执行：B = %s\n", round + 1, round == 0 ? "ReLU(x)" : "x*x*x");
        printf(" x   A=2*x            B    C=A+B\n");
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            const float x = input[i];
            const float expected_a = x * 2.f;
            const float expected_b = round == 0 ? (x > 0.f ? x : 0.f) : x * x * x;
            printf("%2.0f %7.0f %12.0f %8.0f\n", x, output_a[i], output_b[i], output[i]);
            if (output_a[i] != expected_a || output_b[i] != expected_b ||
                output[i] != expected_a + expected_b) ok = false;
        }
        printf("%s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    // node b 来自 graph，因此保留原 graph 到节点更新完成之后。
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_result));
    return all_ok ? 0 : 1;
}
