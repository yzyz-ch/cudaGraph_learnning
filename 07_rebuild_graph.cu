// 07：流程从两个节点变成三个节点，不能用整图更新来改变这个结构。
// 先尝试更新，识别预期的不兼容结果，再用新 graph 重新实例化。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 07_rebuild_graph.cu -o build/07_rebuild_graph
// 运行: ./build/07_rebuild_graph

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

__global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.f;
}

int main() {
    const int n = 4;
    const int steps[] = {2, 3};
    float output[n];
    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graphs[2];
    cudaGraphExec_t exec = nullptr;
    bool all_ok = true;

    for (int round = 0; round < 2; ++round) {
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < steps[round]; ++k) {
            add_one<<<1, n, 0, stream>>>(d, n);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaStreamEndCapture(stream, &graphs[round]));
        if (round == 0) {
            CHECK(cudaGraphInstantiate(&exec, graphs[round], 0));
        } else {
            cudaGraphExecUpdateResultInfo info{};
            cudaError_t status = cudaGraphExecUpdate(exec, graphs[round], &info);
            if (status == cudaErrorGraphExecUpdateFailure) {
                // 本实验预期的拒绝原因是“拓扑改变”，不是计算出错。
                if (info.result != cudaGraphExecUpdateErrorTopologyChanged) {
                    fprintf(stderr, "Unexpected rejection: %d\n", static_cast<int>(info.result));
                    return 1;
                }
                printf("update rejected: topology changed; instantiate a new exec\n");
                // 清掉本次已处理的错误记录；其他错误仍需报告。
                cudaError_t last = cudaGetLastError();
                if (last != cudaSuccess && last != cudaErrorGraphExecUpdateFailure) CHECK(last);
                // 前一轮已经同步。先创建替代品，再销毁旧 exec。
                cudaGraphExec_t replacement;
                CHECK(cudaGraphInstantiate(&replacement, graphs[round], 0));
                CHECK(cudaGraphExecDestroy(exec));
                exec = replacement;
            } else {
                CHECK(status);  // 不能把所有 CUDA 错误都当成“需要重建”。
                printf("update succeeded: topology stayed compatible\n");
            }
        }
        // 每轮从 0 开始，两个节点得到 2，三个节点得到 3。
        CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        printf("%d nodes:", steps[round]);
        for (int i = 0; i < n; ++i) {
            printf(" %.0f", output[i]);
            if (output[i] != static_cast<float>(steps[round])) ok = false;
        }
        printf("  %s\n", ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graphs[0]));
    CHECK(cudaGraphDestroy(graphs[1]));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return all_ok ? 0 : 1;
}
