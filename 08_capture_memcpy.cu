// 08：把“输入复制 -> 乘 2 -> 加 1 -> 输出复制”一起录入图。
// 每轮只改同一个 CPU 输入缓冲区的内容，再重放。地址和复制大小不变。
// pinned memory（页锁定内存）：由 cudaMallocHost 分配，适合异步 CPU/GPU 复制。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 08_capture_memcpy.cu -o build/08_capture_memcpy
// 运行: ./build/08_capture_memcpy

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

__global__ void scale(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.f;
}

__global__ void bias(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.f;
}

int main() {
    const int n = 4;
    const size_t bytes = n * sizeof(float);
    float* input = nullptr;
    float* output = nullptr;
    float* d = nullptr;
    CHECK(cudaMallocHost(&input, bytes));   // CPU 上可直接读写。
    CHECK(cudaMallocHost(&output, bytes));
    CHECK(cudaMalloc(&d, bytes));          // GPU 内存。
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    // 用 Async 版本并指定正在捕获的流，同一条流自然建立这四步的顺序。
    CHECK(cudaMemcpyAsync(d, input, bytes, cudaMemcpyHostToDevice, stream));
    scale<<<1, n, 0, stream>>>(d, n);
    CHECK(cudaGetLastError());
    bias<<<1, n, 0, stream>>>(d, n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(output, d, bytes, cudaMemcpyDeviceToHost, stream));
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    bool all_ok = true;
    for (int batch = 0; batch < 2; ++batch) {
        // 捕获不冻结 input 的内容；每次重放都会读取这次填写的数据。
        for (int i = 0; i < n; ++i) input[i] = static_cast<float>(batch * n + i + 1);
        CHECK(cudaGraphLaunch(exec, stream));
        // 等图内输出复制也完成，才读 output 或改写下一批 input。
        CHECK(cudaStreamSynchronize(stream));
        bool ok = true;
        printf("batch %d:", batch + 1);
        for (int i = 0; i < n; ++i) {
            printf(" %.0f", output[i]);
            if (output[i] != input[i] * 2.f + 1.f) ok = false;
        }
        printf("  %s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    CHECK(cudaFreeHost(input));
    CHECK(cudaFreeHost(output));
    return all_ok ? 0 : 1;
}
