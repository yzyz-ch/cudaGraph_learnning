// 14：逐项测试 CUDA 12.6 的更新行为。每项从全新的原图和 exec 开始。
// 第一组：kernel 的 7 种变化，分别测试单节点更新、整图更新。
// 第二组：一维 D2D 复制的 3 种变化，同样测试两种更新方式。
// 第三组：节点类型、节点数量、依赖变化，整图更新应被拒绝。
// 本文件较长，可先只读 kernel_case；它不代表所有设备和节点类型的完整兼容表。
// 编译: mkdir -p build && nvcc -O2 -std=c++14 14_update_support.cu -o build/14_update_support
// 运行: ./build/14_update_support
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 本文件有几个辅助函数，出错时直接结束测试程序，而不是只退出某个函数。
#define CHECK(call) do {                                                       \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)
#define REQUIRE(condition) do {                                                \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);   \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

const int capacity = 256;  // 所有测试都保留足够大的缓冲区。

__global__ void transform(const float* x, float* y, int n, float value) {
    extern __shared__ float scratch[];  // 大小由 sharedMemBytes 指定。
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    scratch[threadIdx.x] = i < n ? x[i] : 0.f;
    __syncthreads();
    if (i < n) y[i] = scratch[threadIdx.x] + value;
}
__global__ void other_transform(const float* x, float* y, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.f * x[i] + value;
}
__global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.f;
}

cudaKernelNodeParams kernel_params(void* function, void** args) {
    cudaKernelNodeParams p{};
    p.func = function;
    p.gridDim = dim3(2);
    p.blockDim = dim3(128);
    p.sharedMemBytes = 128 * sizeof(float);
    p.kernelParams = args;
    return p;
}

// method=0：单节点更新；method=1：提供新 graph，更新原 exec。
void kernel_case(int method, int test, float* src0, float* src1, float* dst,
                 cudaStream_t stream) {
    const char* names[] = {"数值参数", "输入地址", "元素数量 n", "grid 大小",
                           "block 大小", "动态共享内存大小", "kernel 函数"};
    int n = 128;
    float value = 1.f;
    float* src = src0;
    void* args[] = {&src, &dst, &n, &value};
    auto p = kernel_params(reinterpret_cast<void*>(transform), args);
    cudaGraph_t graph, candidate = nullptr;
    cudaGraphNode_t node;
    CHECK(cudaGraphCreate(&graph, 0));
    CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    // 每项只改一个因素；没有变化的配置保留原值。
    switch (test) {
        case 0: value = 3.f; break;
        case 1: src = src1; break;
        case 2: n = 64; break;
        case 3: p.gridDim = dim3(4); break;
        case 4: p.blockDim = dim3(64); break;
        case 5: p.sharedMemBytes = 256 * sizeof(float); break;
        case 6: p.func = reinterpret_cast<void*>(other_transform); break;
    }
    if (method == 0) {
        CHECK(cudaGraphExecKernelNodeSetParams(exec, node, &p));
    } else {
        CHECK(cudaGraphCreate(&candidate, 0));
        cudaGraphNode_t new_node;
        CHECK(cudaGraphAddKernelNode(&new_node, candidate, nullptr, 0, &p));
        cudaGraphExecUpdateResultInfo info{};
        CHECK(cudaGraphExecUpdate(exec, candidate, &info));
        REQUIRE(info.result == cudaGraphExecUpdateSuccess);
    }
    CHECK(cudaMemsetAsync(dst, 0, capacity * sizeof(float), stream));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    float output[capacity];
    CHECK(cudaMemcpy(output, dst, sizeof(output), cudaMemcpyDeviceToHost));
    for (int i = 0; i < capacity; ++i) {
        float x = (src == src0 ? 1.f : -1.f) * (i + 1);
        float expected = i < n ? (test == 6 ? 2.f * x : x) + value : 0.f;
        REQUIRE(output[i] == expected);  // 同时检查没有处理的尾部仍为 0。
    }
    printf("%s / kernel / %s: PASS\n", method == 0 ? "单节点更新" : "整图更新", names[test]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    if (candidate) CHECK(cudaGraphDestroy(candidate));
}

void memcpy_case(int method, int test, float* src0, float* src1,
                 float* dst0, float* dst1, cudaStream_t stream) {
    const char* names[] = {"源地址", "目标地址", "复制大小"};
    float* src = src0;
    float* dst = dst0;
    int count = 128;
    cudaGraph_t graph, candidate = nullptr;
    cudaGraphNode_t node;
    CHECK(cudaGraphCreate(&graph, 0));
    CHECK(cudaGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src,
                                  count * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    if (test == 0) src = src1;
    if (test == 1) dst = dst1;
    if (test == 2) count = 64;
    if (method == 0) {
        CHECK(cudaGraphExecMemcpyNodeSetParams1D(exec, node, dst, src,
                                                count * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {
        CHECK(cudaGraphCreate(&candidate, 0));
        cudaGraphNode_t new_node;
        CHECK(cudaGraphAddMemcpyNode1D(&new_node, candidate, nullptr, 0, dst, src,
                                      count * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaGraphExecUpdateResultInfo info{};
        CHECK(cudaGraphExecUpdate(exec, candidate, &info));
        REQUIRE(info.result == cudaGraphExecUpdateSuccess);
    }
    CHECK(cudaMemsetAsync(dst0, 0, capacity * sizeof(float), stream));
    CHECK(cudaMemsetAsync(dst1, 0, capacity * sizeof(float), stream));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    // 两个目标都检查：被选中的收到数据，未选中的和尾部保持 0。
    float* destinations[] = {dst0, dst1};
    for (float* buffer : destinations) {
        float output[capacity];
        CHECK(cudaMemcpy(output, buffer, sizeof(output), cudaMemcpyDeviceToHost));
        for (int i = 0; i < capacity; ++i) {
            float expected = buffer == dst && i < count ? (src == src0 ? 1.f : -1.f) * (i + 1) : 0.f;
            REQUIRE(output[i] == expected);
        }
    }
    printf("%s / memcpy / %s: PASS\n", method == 0 ? "单节点更新" : "整图更新", names[test]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    if (candidate) CHECK(cudaGraphDestroy(candidate));
}

void rejected_case(int test, float* src, float* dst, cudaStream_t stream) {
    const char* names[] = {"Kernel 改成 Memcpy", "一个节点改成两个", "去掉节点间依赖"};
    int n = 128;
    void* args[] = {&dst, &n};
    auto p = kernel_params(reinterpret_cast<void*>(add_one), args);
    cudaGraph_t graphs[2];
    // graph[0] 是原图，graph[1] 是不兼容的新定义。
    for (int version = 0; version < 2; ++version) {
        CHECK(cudaGraphCreate(&graphs[version], 0));
        if (test == 0 && version == 1) {
            cudaGraphNode_t node;
            CHECK(cudaGraphAddMemcpyNode1D(&node, graphs[version], nullptr, 0, dst, src,
                                          n * sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            int nodes = (test == 2 || (test == 1 && version == 1)) ? 2 : 1;
            cudaGraphNode_t previous = nullptr;
            for (int i = 0; i < nodes; ++i) {
                bool depends = i > 0 && !(test == 2 && version == 1);
                cudaGraphNode_t node;
                CHECK(cudaGraphAddKernelNode(&node, graphs[version], depends ? &previous : nullptr,
                                              depends ? 1 : 0, &p));
                previous = node;
            }
        }
    }
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graphs[0], 0));
    cudaGraphExecUpdateResultInfo info{};
    cudaError_t status = cudaGraphExecUpdate(exec, graphs[1], &info);
    REQUIRE(status == cudaErrorGraphExecUpdateFailure);
    auto expected = test == 0 ? cudaGraphExecUpdateErrorNodeTypeChanged : cudaGraphExecUpdateErrorTopologyChanged;
    REQUIRE(info.result == expected);
    // 只清除已识别的预期错误，其他错误仍然报告。
    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess && last != cudaErrorGraphExecUpdateFailure) CHECK(last);
    // 更新失败后旧 exec 仍可使用；只执行旧图，不运行那个缺少依赖的新图。
    CHECK(cudaMemsetAsync(dst, 0, capacity * sizeof(float), stream));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    float output[capacity];
    CHECK(cudaMemcpy(output, dst, sizeof(output), cudaMemcpyDeviceToHost));
    for (int i = 0; i < capacity; ++i) {
        REQUIRE(output[i] == (i < n ? (test == 2 ? 2.f : 1.f) : 0.f));
    }
    printf("整图更新 / %s: 按预期拒绝 (%s)，旧 exec 校验 PASS\n", names[test],
           test == 0 ? "NodeTypeChanged" : "TopologyChanged");
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graphs[0]));
    CHECK(cudaGraphDestroy(graphs[1]));
}

int main() {
    float *src0, *src1, *dst0, *dst1;
    CHECK(cudaMalloc(&src0, capacity * sizeof(float)));
    CHECK(cudaMalloc(&src1, capacity * sizeof(float)));
    CHECK(cudaMalloc(&dst0, capacity * sizeof(float)));
    CHECK(cudaMalloc(&dst1, capacity * sizeof(float)));
    float positive[capacity], negative[capacity];
    for (int i = 0; i < capacity; ++i) {
        positive[i] = static_cast<float>(i + 1);
        negative[i] = -positive[i];
    }
    CHECK(cudaMemcpy(src0, positive, sizeof(positive), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(src1, negative, sizeof(negative), cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    for (int method = 0; method < 2; ++method) {
        for (int test = 0; test < 7; ++test) kernel_case(method, test, src0, src1, dst0, stream);
        for (int test = 0; test < 3; ++test) memcpy_case(method, test, src0, src1, dst0, dst1, stream);
    }
    for (int test = 0; test < 3; ++test) rejected_case(test, src0, dst0, stream);
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(src0));
    CHECK(cudaFree(src1));
    CHECK(cudaFree(dst0));
    CHECK(cudaFree(dst1));
    printf("23 项测试全部 PASS（包括 3 项预期拒绝）。\n");
    return 0;
}
