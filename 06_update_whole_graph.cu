// 06：流程仍是“乘系数 -> 加偏置”，同时更改参数和 GPU 数据地址。
// 重新 capture 得到新 graph，再更新已有 exec；无需逐个寻找节点。
// 注意：重新捕获得到定义，不等于重新实例化。这里仅第一轮实例化。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 06_update_whole_graph.cu -o build/06_update_whole_graph
// 运行: ./build/06_update_whole_graph

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

__global__ void scale(float* x, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}

__global__ void bias(float* x, int n, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += offset;
}

int main() {
    const int n = 4;
    const float input[n] = {1.f, 2.f, 3.f, 4.f};
    const float factors[] = {2.f, 3.f};
    const float offsets[] = {1.f, 10.f};
    float output[n];
    float* buffers[2];
    CHECK(cudaMalloc(&buffers[0], n * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graphs[2];
    cudaGraphExec_t exec = nullptr;
    bool all_ok = true;

    for (int round = 0; round < 2; ++round) {
        float* d = buffers[round];  // 第二轮确实换成另一块 GPU 内存。
        CHECK(cudaMemcpy(d, input, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        scale<<<1, n, 0, stream>>>(d, n, factors[round]);
        CHECK(cudaGetLastError());
        bias<<<1, n, 0, stream>>>(d, n, offsets[round]);
        CHECK(cudaGetLastError());
        CHECK(cudaStreamEndCapture(stream, &graphs[round]));

        if (round == 0) {
            CHECK(cudaGraphInstantiate(&exec, graphs[round], 0));
            printf("round 1: instantiate\n");
        } else {
            // 拓扑（节点及连接关系）没变，函数和捕获顺序也相同。
            // 本例的参数变化可更新；是否真的成功，仍须检查 API 返回值。
            cudaGraphExecUpdateResultInfo info{};
            CHECK(cudaGraphExecUpdate(exec, graphs[round], &info));
            if (info.result != cudaGraphExecUpdateSuccess) {
                fprintf(stderr, "Unexpected update result: %d\n", static_cast<int>(info.result));
                return 1;
            }
            printf("round 2: update succeeded; new buffer, factor and offset\n");
        }
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        printf("output:");
        for (int i = 0; i < n; ++i) {
            printf(" %.0f", output[i]);
            if (output[i] != input[i] * factors[round] + offsets[round]) ok = false;
        }
        printf("  %s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    CHECK(cudaGraphExecDestroy(exec));
    for (int i = 0; i < 2; ++i) {
        CHECK(cudaGraphDestroy(graphs[i]));
        CHECK(cudaFree(buffers[i]));
    }
    CHECK(cudaStreamDestroy(stream));
    return all_ok ? 0 : 1;
}
