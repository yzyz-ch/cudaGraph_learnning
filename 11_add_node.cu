// 11：给已有 graph 添加节点 D，并建立 C -> D 的依赖。
// 原来：A=2*x ────┐          添加后：A=2*x ────┐
//                ├-> C=A+B                  ├-> C=A+B -> D：结果加 10
//       B=ReLU(x) ┘                 B=ReLU(x) ┘
// 第一次运行原 exec；第二次只给 graph 加节点，仍运行旧 exec；
// 第三次重新实例化，用新 exec 执行包含 D 的流程。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 11_add_node.cu -o build/11_add_node
// 运行: ./build/11_add_node

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

__global__ void add_arrays(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) result[i] = a[i] + b[i];
}

// D：在 C 写出的结果上加一个数。
__global__ void add_value(float* result, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) result[i] += value;
}

int main() {
    int n = 4;
    float added_value = 10.f;
    const float input[4] = {-2.f, -1.f, 0.f, 3.f};
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

    void* args_a[] = {&d_x, &d_a, &n};
    void* args_b[] = {&d_x, &d_b, &n};
    void* args_c[] = {&d_a, &d_b, &d_result, &n};
    void* args_d[] = {&d_result, &n, &added_value};
    cudaKernelNodeParams params{};
    params.gridDim = dim3(1);
    params.blockDim = dim3(n);
    cudaGraphNode_t a, b, c, d;

    // 先建立与 09 相同的三个节点。
    params.func = reinterpret_cast<void*>(scale);
    params.kernelParams = args_a;
    CHECK(cudaGraphAddKernelNode(&a, graph, nullptr, 0, &params));
    params.func = reinterpret_cast<void*>(relu);
    params.kernelParams = args_b;
    CHECK(cudaGraphAddKernelNode(&b, graph, nullptr, 0, &params));
    cudaGraphNode_t dependencies[] = {a, b};
    params.func = reinterpret_cast<void*>(add_arrays);
    params.kernelParams = args_c;
    CHECK(cudaGraphAddKernelNode(&c, graph, dependencies, 2, &params));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    bool all_ok = true;
    for (int round = 0; round < 3; ++round) {
        if (round == 1) {
            params.func = reinterpret_cast<void*>(add_value);
            params.kernelParams = args_d;
            // &d 接收新节点的标识；&c, 1 表示 D 依赖 C 这一个节点。
            // C 已依赖 A、B，所以 D 不需要再单独依赖 A、B。
            CHECK(cudaGraphAddKernelNode(&d, graph, &c, 1, &params));
            // 这里只改了 graph，旧 exec 不会自动多出 D。
        }
        if (round == 2) {
            // 节点数改变，直接重新实例化。上一轮已经同步完成。
            cudaGraphExec_t replacement;
            CHECK(cudaGraphInstantiate(&replacement, graph, 0));
            CHECK(cudaGraphExecDestroy(exec));
            exec = replacement;
        }
        // 每轮 C 都覆盖写入结果，不会累积上一轮的值。
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output, d_result, n * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        printf("第 %d 次执行:", round + 1);
        for (int i = 0; i < n; ++i) {
            float expected = 2.f * input[i] + (input[i] > 0.f ? input[i] : 0.f);
            if (round == 2) expected += added_value;
            printf(" %.0f", output[i]);
            if (output[i] != expected) ok = false;
        }
        printf("  %s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_result));
    return all_ok ? 0 : 1;
}
