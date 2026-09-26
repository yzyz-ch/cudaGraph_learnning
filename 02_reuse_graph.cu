// 入门：图只建一次，把新数据放到同一地址，再执行同一张图。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 -o build/02_reuse_graph 02_reuse_graph.cu
// 运行: ./build/02_reuse_graph

#include <cstdio>
#include <cuda_runtime.h>

// 检查 CUDA API 的返回值，出错时打印位置并退出 main。
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
    const float inputs[2][n] = {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}};
    float output[n];
    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 图记录的是指针 d 的地址值，不是这个地址中的数据快照。
    // 这里仅录制，不执行，所以暂时还不需要给 d 中的数据赋值。
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    scale<<<1, n, 0, stream>>>(d, n);
    CHECK(cudaGetLastError());
    bias<<<1, n, 0, stream>>>(d, n);
    CHECK(cudaGetLastError());
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    bool ok = true;
    for (int batch = 0; batch < 2; ++batch) {
        // 复制在捕获区域外；本例用同步复制，让操作顺序一目了然。
        CHECK(cudaMemcpy(d, inputs[batch], n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaGraphLaunch(exec, stream));
        // 等本轮完成，才读取结果或覆盖这块内存。
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));

        printf("batch %d:", batch + 1);
        for (int i = 0; i < n; ++i) {
            printf(" %.0f", output[i]);
            if (output[i] != inputs[batch][i] * 2.f + 1.f) ok = false;
        }
        printf("\n");
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return ok ? 0 : 1;
}
