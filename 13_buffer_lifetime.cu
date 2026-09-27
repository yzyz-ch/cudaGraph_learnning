// 13：两套缓冲区轮流使用；只在复用某套缓冲区前，等待它上一次工作完成。
// 每套有自己的 CPU 输入/输出、GPU 内存、stream、exec 和完成事件。
// 不运行“提前释放或覆盖内存”的错误版本，因为那会产生不确定结果。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 13_buffer_lifetime.cu -o build/13_buffer_lifetime
// 运行: ./build/13_buffer_lifetime
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        return 1;                                                             \
    }                                                                         \
} while (0)

__global__ void transform(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 2.f * x[i] + 1.f;
}

bool check_batch(const float* output, int batch, int slot, int n) {
    bool ok = true;
    printf("第 %d 批，缓冲区 %d:", batch + 1, slot + 1);
    for (int i = 0; i < n; ++i) {
        printf(" %.0f", output[i]);
        if (output[i] != 2.f * (batch * 10 + i) + 1.f) ok = false;
    }
    printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    const int n = 4, batches = 6;
    const size_t bytes = n * sizeof(float);
    float *input[2], *output[2], *device[2];
    cudaStream_t streams[2];
    cudaEvent_t done[2];
    cudaGraph_t graphs[2];
    cudaGraphExec_t execs[2];
    int last_batch[2] = {-1, -1};  // -1 表示这套缓冲区还没提交过任务。

    for (int slot = 0; slot < 2; ++slot) {
        CHECK(cudaMallocHost(&input[slot], bytes));
        CHECK(cudaMallocHost(&output[slot], bytes));
        CHECK(cudaMalloc(&device[slot], bytes));
        CHECK(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking));
        CHECK(cudaEventCreateWithFlags(&done[slot], cudaEventDisableTiming));
        CHECK(cudaStreamBeginCapture(streams[slot], cudaStreamCaptureModeGlobal));
        CHECK(cudaMemcpyAsync(device[slot], input[slot], bytes, cudaMemcpyHostToDevice, streams[slot]));
        transform<<<1, n, 0, streams[slot]>>>(device[slot], n);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpyAsync(output[slot], device[slot], bytes, cudaMemcpyDeviceToHost, streams[slot]));
        CHECK(cudaStreamEndCapture(streams[slot], &graphs[slot]));
        CHECK(cudaGraphInstantiate(&execs[slot], graphs[slot], 0));
    }

    bool all_ok = true;
    for (int batch = 0; batch < batches; ++batch) {
        int slot = batch % 2;
        if (last_batch[slot] >= 0) {
            // 先等这套缓冲区上一次输出复制完成，读取结果，然后才能覆盖输入。
            CHECK(cudaEventSynchronize(done[slot]));
            if (!check_batch(output[slot], last_batch[slot], slot, n)) all_ok = false;
        }
        for (int i = 0; i < n; ++i) input[slot][i] = static_cast<float>(batch * 10 + i);
        CHECK(cudaGraphLaunch(execs[slot], streams[slot]));
        // 事件排在整张图后面，完成意味着该批输入/输出都已用完。
        CHECK(cudaEventRecord(done[slot], streams[slot]));
        last_batch[slot] = batch;
        // 此处不等待，CPU 可以准备另一套缓冲区的数据。
    }
    // 最后两套缓冲区可能仍在使用中；等完、校验后再释放资源。
    for (int slot = 0; slot < 2; ++slot) {
        if (last_batch[slot] >= 0) {
            CHECK(cudaEventSynchronize(done[slot]));
            if (!check_batch(output[slot], last_batch[slot], slot, n)) all_ok = false;
        }
        CHECK(cudaGraphExecDestroy(execs[slot]));
        CHECK(cudaGraphDestroy(graphs[slot]));
        CHECK(cudaEventDestroy(done[slot]));
        CHECK(cudaStreamDestroy(streams[slot]));
        CHECK(cudaFree(device[slot]));
        CHECK(cudaFreeHost(input[slot]));
        CHECK(cudaFreeHost(output[slot]));
    }
    return all_ok ? 0 : 1;
}
