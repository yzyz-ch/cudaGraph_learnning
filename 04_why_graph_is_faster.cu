// CUDA Graph 为什么能加速？先看这一段，再看 main。
//
// 普通 launch：CPU 每次调用 CUDA，提交一个 kernel。
// Graph：先记录 kernel 和执行顺序，实例化时提前做好一部分准备；
//        重放时通过一次调用提交整套流程，减少重复提交和调度开销。
//        GPU 仍执行原来的 kernel，计算量没有减少，也没有自动融合。
//
// 本实验：每种方式都执行 20000 个“加 1”kernel，只改变每次提交多少个。
// 普通方式：       20000 次 kernel launch，每次 1 个 kernel。
// 每图 1 个：      20000 次 graph launch，没有减少提交次数。
// 每图 5 个：       4000 次 graph launch。
// 每图 20 个：      1000 次 graph launch。
// 每图 100 个：      200 次 graph launch。
//
// 观察：提交次数减少后，总耗时是否下降？单节点图不保证更快；
// 图变大也不保证一直加速，因为实际计算和 GPU 调度仍然需要时间。
// 构图、实例化、预热均不计时，所以测到的差异不是实例化成本。
// submit_ms 是 CPU 提交循环经过的时间，可能包含队列拥塞带来的等待；
// total_ms 是从开始提交到 GPU 全部完成的时间，不是两段时间相加。
//
// 编译: mkdir -p build && nvcc -O2 -std=c++14 04_why_graph_is_faster.cu -o build/04_why_graph_is_faster
// 运行: ./build/04_why_graph_is_faster

#include <chrono>
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
    const int n = 256;
    const int total_kernels = 20000;  // 每种方式的正式测试工作量完全相同。
    const int warmup_kernels = 1000;
    const int cases[] = {0, 1, 5, 20, 100};  // 0 表示普通 launch，其他值为每图节点数。
    float output[n];
    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    printf("Each case: %d kernels, each element starts at 0 and ends at %d.\n",
           total_kernels, total_kernels);
    printf("mode     kernels/submit  submits  submit_ms  total_ms  check\n");
    bool all_ok = true;
    for (int nodes : cases) {
        const bool use_graph = (nodes > 0);
        const int kernels_per_submit = use_graph ? nodes : 1;
        // 修改 cases 做练习时，保证每种方式仍执行相同数量的 kernel。
        if (total_kernels % kernels_per_submit || warmup_kernels % kernels_per_submit) {
            fprintf(stderr, "Each group size must divide total_kernels and warmup_kernels.\n");
            return 1;
        }
        const int submits = total_kernels / kernels_per_submit;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        if (use_graph) {
            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            // CPU 循环录出 nodes 个节点；它们在同一条流上依次执行。
            for (int k = 0; k < nodes; ++k) {
                add_one<<<1, n, 0, stream>>>(d, n);
            }
            CHECK(cudaGetLastError());
            CHECK(cudaStreamEndCapture(stream, &graph));
            CHECK(cudaGraphInstantiate(&exec, graph, 0));
        }

        // 预热包含 Graph 首次执行，避免把首次准备成本混入计时。
        CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
        for (int i = 0; i < warmup_kernels / kernels_per_submit; ++i) {
            if (use_graph) {
                CHECK(cudaGraphLaunch(exec, stream));
            } else {
                add_one<<<1, n, 0, stream>>>(d, n);
            }
        }
        CHECK(cudaGetLastError());
        // 同一条流保证：预热做完，再清零。等待完成后才开始计时。
        CHECK(cudaMemsetAsync(d, 0, n * sizeof(float), stream));
        CHECK(cudaStreamSynchronize(stream));

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < submits; ++i) {
            if (use_graph) {
                CHECK(cudaGraphLaunch(exec, stream));
            } else {
                add_one<<<1, n, 0, stream>>>(d, n);
            }
        }
        CHECK(cudaGetLastError());
        const auto submitted = std::chrono::steady_clock::now();
        CHECK(cudaStreamSynchronize(stream));
        const auto finished = std::chrono::steady_clock::now();
        const double submit_ms = std::chrono::duration<double, std::milli>(submitted - start).count();
        const double total_ms = std::chrono::duration<double, std::milli>(finished - start).count();

        // 不计入计时：检查每一种分组都完成了相同的计算。
        CHECK(cudaMemcpy(output, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (output[i] != static_cast<float>(total_kernels)) ok = false;
        }
        all_ok = all_ok && ok;
        printf("%-8s %14d %8d %10.3f %9.3f  %s\n",
               use_graph ? "graph" : "normal", kernels_per_submit, submits,
               submit_ms, total_ms, ok ? "PASS" : "FAIL");
        if (use_graph) {
            CHECK(cudaGraphExecDestroy(exec));
            CHECK(cudaGraphDestroy(graph));
        }
    }
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d));
    return all_ok ? 0 : 1;
}
