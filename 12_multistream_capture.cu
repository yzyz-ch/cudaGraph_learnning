// 12：用两条 stream 捕获“分支 -> 汇合”，不用手动添加 kernel 节点。
// stream1：复制输入 -> 记录 ready -> A=2*x -----------> 等 B -> C=A+B
// stream2：             等 ready -> B=ReLU(x) -> 记录 done
// event（事件）在这里表达执行依赖，不用于测时间。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 12_multistream_capture.cu -o build/12_multistream_capture
// 运行: ./build/12_multistream_capture
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        return 1;                                                             \
    }                                                                         \
} while (0)

__global__ void scale(const float* x, float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 2.f * x[i];
}
__global__ void relu(const float* x, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = x[i] > 0.f ? x[i] : 0.f;
}
__global__ void add_arrays(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) result[i] = a[i] + b[i];
}

int main() {
    const int n = 4;
    const size_t bytes = n * sizeof(float);
    const float values[n] = {-2.f, -1.f, 0.f, 3.f};
    float *input, *d_x, *d_a, *d_b, *d_result;
    float output[n];
    CHECK(cudaMallocHost(&input, bytes));
    for (int i = 0; i < n; ++i) input[i] = values[i];
    CHECK(cudaMalloc(&d_x, bytes));
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_result, bytes));
    cudaStream_t stream1, stream2;
    CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    cudaEvent_t ready, done;
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

    CHECK(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
    CHECK(cudaMemcpyAsync(d_x, input, bytes, cudaMemcpyHostToDevice, stream1));
    CHECK(cudaEventRecord(ready, stream1));
    // 这次等待使 stream2 加入同一次捕获，并依赖输入复制完成。
    CHECK(cudaStreamWaitEvent(stream2, ready, 0));
    scale<<<1, n, 0, stream1>>>(d_x, d_a, n);
    CHECK(cudaGetLastError());
    relu<<<1, n, 0, stream2>>>(d_x, d_b, n);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(done, stream2));
    // 回到起始流汇合：C 在 A 后面，还要等待 B。
    CHECK(cudaStreamWaitEvent(stream1, done, 0));
    add_arrays<<<1, n, 0, stream1>>>(d_a, d_b, d_result, n);
    CHECK(cudaGetLastError());
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream1, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaGraphLaunch(exec, stream1));
    CHECK(cudaStreamSynchronize(stream1));
    CHECK(cudaMemcpy(output, d_result, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    printf("两条流捕获后的结果:");
    for (int i = 0; i < n; ++i) {
        printf(" %.0f", output[i]);
        float expected = 2.f * values[i] + (values[i] > 0.f ? values[i] : 0.f);
        if (output[i] != expected) ok = false;
    }
    printf("  %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaEventDestroy(ready));
    CHECK(cudaEventDestroy(done));
    CHECK(cudaStreamDestroy(stream1));
    CHECK(cudaStreamDestroy(stream2));
    CHECK(cudaFreeHost(input));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_result));
    return ok ? 0 : 1;
}
