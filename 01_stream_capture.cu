// 入门: 用 stream capture 把两段 kernel 录成一张 CUDA Graph, 再整段重放。
//
// 编译: mkdir -p build && nvcc -O2 -std=c++14 -o build/01_stream_capture 01_stream_capture.cu
// 运行: ./build/01_stream_capture
//
// 数据从 1 出发, 每次重放做 x = x * 2 + 1。
// 重放 3 次: 1 -> 3 -> 7 -> 15

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
    float h[4] = {1.f, 1.f, 1.f, 1.f};

    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));

    // stream（流）是按顺序执行任务的队列；本例自己创建一条流。
    // 不能捕获传统默认流 cudaStreamLegacy；这里无需展开默认流的其他模式。
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // capture（捕获）：只记录下面两个操作及其顺序，不执行它们。
    // Global 是捕获期间的安全检查模式，并非“捕获程序里的所有操作”。
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    scale<<<1, n, 0, stream>>>(d, n);
    CHECK(cudaGetLastError());
    bias<<<1, n, 0, stream>>>(d, n);
    CHECK(cudaGetLastError());
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));

    // instantiate（实例化）：把 graph 这份流程定义准备成可执行的 exec。
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    const int repeats = 3;
    for (int i = 0; i < repeats; ++i) {
        // replay（重放）：每次都执行“乘 2 -> 加 1”，两个 kernel 仍然独立。
        CHECK(cudaGraphLaunch(exec, stream));
    }
    // launch 是异步提交；等这条流上的工作完成后，再读取结果。
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("after %d replays:", repeats);
    for (int i = 0; i < n; ++i) printf(" %.0f", h[i]);
    printf("\n");

    // 用 CPU 算一遍期望结果，方便修改 repeats 做练习。
    float expected = 1.f;
    for (int i = 0; i < repeats; ++i) expected = expected * 2.f + 1.f;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (h[i] != expected) ok = false;
    }
    printf("%s (expected each element: %.0f)\n", ok ? "PASS" : "FAIL", expected);

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return ok ? 0 : 1;
}
