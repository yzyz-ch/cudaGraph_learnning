// 17：输入规模只在 4 和 8 之间切换，各保留一张图及自己的 GPU 缓冲区。
// 第一次遇到某种规模才构建；再次遇到直接复用，不更新、不重新实例化。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 17_graph_cache.cu -o build/17_graph_cache
// 运行: ./build/17_graph_cache
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

int main() {
    const int sizes[] = {4, 8};
    const int requests[] = {4, 8, 4, 8, 4};
    float input[8], output[8];
    float* buffers[2] = {nullptr, nullptr};
    cudaGraph_t graphs[2] = {nullptr, nullptr};
    cudaGraphExec_t execs[2] = {nullptr, nullptr};
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    int builds = 0;
    bool all_ok = true;

    for (int round = 0; round < 5; ++round) {
        int n = requests[round];
        int slot = n == sizes[0] ? 0 : (n == sizes[1] ? 1 : -1);
        if (slot < 0) {
            fprintf(stderr, "本例只准备了 n=4 和 n=8 的缓存。\n");
            return 1;
        }
        bool first_use = (execs[slot] == nullptr);
        if (first_use) {
            CHECK(cudaMalloc(&buffers[slot], n * sizeof(float)));
            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            transform<<<1, n, 0, stream>>>(buffers[slot], n);
            CHECK(cudaGetLastError());
            CHECK(cudaStreamEndCapture(stream, &graphs[slot]));
            CHECK(cudaGraphInstantiate(&execs[slot], graphs[slot], 0));
            ++builds;
        }
        for (int i = 0; i < n; ++i) input[i] = static_cast<float>(round * 10 + i);
        // 始终把新数据放到该缓存条目自己的地址，图里的指针无需改变。
        CHECK(cudaMemcpy(buffers[slot], input, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaGraphLaunch(execs[slot], stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output, buffers[slot], n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        printf("第 %d 次执行，n=%d，%s:", round + 1, n, first_use ? "首次构建" : "复用缓存");
        for (int i = 0; i < n; ++i) {
            printf(" %.0f", output[i]);
            if (output[i] != 2.f * input[i] + 1.f) ok = false;
        }
        printf("  %s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    printf("构建次数: %d\n", builds);
    for (int slot = 0; slot < 2; ++slot) {
        if (execs[slot] != nullptr) {
            CHECK(cudaGraphExecDestroy(execs[slot]));
            CHECK(cudaGraphDestroy(graphs[slot]));
            CHECK(cudaFree(buffers[slot]));
        }
    }
    CHECK(cudaStreamDestroy(stream));
    return all_ok ? 0 : 1;
}
