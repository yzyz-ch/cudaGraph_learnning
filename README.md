# CUDA Graph 学习

一组可以独立编译的小例子，用来按顺序学习 CUDA Graph：先捕获并重放固定流程，再处理参数变化、重新构建、数据复制和分支依赖。

## 怎么学

先读阶段说明，再编译运行对应的 `.cu`。每节先预测输出，再对照结果，最后做文末的小练习。

| 阶段 | 说明 | 示例 |
| --- | --- | --- |
| Step 1 | [入门：捕获、复用、性能与加速原理](step1.md) | 01–04 |
| Step 2 | [进阶：参数更新、重建、复制与依赖](step2.md) | 05–11 |
| Step 3 | [应用：多流、内存、更新测试、计时、节点开关与缓存](step3.md) | 12–17 |

先完成 Step 1，再进入 Step 2。Step 2 里建议先学 05–07，能区分“换数据、更新参数、重新构建”后，再学 08–09，最后对比 10 的替换 kernel 函数和 11 的添加新节点。

完成前 11 节后进入 Step 3。想先做更新能力测试，可以直接学 14、15；想先处理实际数据流程，可以从 12、13 开始。14 比其他例子长，按函数分组阅读即可。

`step1.md` 里写到的“下一阶段学习说明”就是 [step2.md](step2.md)，第三阶段内容集中在 [step3.md](step3.md)。

## 示例

| 文件 | 这一节在做什么 |
| --- | --- |
| [01_stream_capture.cu](01_stream_capture.cu) | 捕获两次 kernel，重放三次 |
| [02_reuse_graph.cu](02_reuse_graph.cu) | 图不变，换同一块 GPU 内存里的输入 |
| [03_launch_vs_graph.cu](03_launch_vs_graph.cu) | 对比普通 launch 和 Graph 的耗时 |
| [04_why_graph_is_faster.cu](04_why_graph_is_faster.cu) | 固定总计算量，改变每次提交包含的 kernel 数 |
| [05_update_kernel_node.cu](05_update_kernel_node.cu) | 更新单个 kernel 节点的参数 |
| [06_update_whole_graph.cu](06_update_whole_graph.cu) | 重新捕获，并更新整张可执行图 |
| [07_rebuild_graph.cu](07_rebuild_graph.cu) | 节点数量变化时，更新被拒绝后重新实例化 |
| [08_capture_memcpy.cu](08_capture_memcpy.cu) | 把输入复制、计算、输出复制放进同一张图 |
| [09_graph_dependencies.cu](09_graph_dependencies.cu) | 两条分支汇合后再继续 |
| [10_update_kernel_function.cu](10_update_kernel_function.cu) | 保留同一个 exec，把 B 从 ReLU 改为求立方 |
| [11_add_node.cu](11_add_node.cu) | 在 C 后添加 D 节点，重新实例化后执行新流程 |
| [12_multistream_capture.cu](12_multistream_capture.cu) | 用 event 将两条流的分支捕获为一张图 |
| [13_buffer_lifetime.cu](13_buffer_lifetime.cu) | 两套缓冲区轮流使用，复用前等待对应任务完成 |
| [14_update_support.cu](14_update_support.cu) | 单节点与整图更新的 23 项行为测试 |
| [15_graph_timing.cu](15_graph_timing.cu) | 区分准备、首次执行、更新和重放耗时 |
| [16_enable_node.cu](16_enable_node.cu) | 启用、禁用已有节点，复用同一个 exec |
| [17_graph_cache.cu](17_graph_cache.cu) | 按输入规模保存并复用多个 exec |

原理、编译命令、预期输出和练习都写在对应阶段的说明里。

## 编译运行

需要 NVIDIA GPU 和 CUDA Toolkit。示例按 CUDA 12.6 编写，并在 RTX 4070 Ti SUPER 上验证过。

```bash
mkdir -p build
nvcc -O2 -std=c++14 01_stream_capture.cu -o build/01_stream_capture
./build/01_stream_capture
```

把源文件名换成其他小节即可。`build/` 只放编译产物，已写入 `.gitignore`。

程序会检查 CUDA 错误和计算结果，失败时返回非零退出码。沙箱或没有显卡的环境里 CUDA 可能初始化失败，请在能访问 GPU 的终端运行。

## 先记住这一点

CUDA Graph 提前记录一组 GPU 操作及其先后关系，之后用一次 Graph launch 提交整个流程。它减少的是重复提交和调度的开销，不会自动把 kernel 融合，也不会减少计算量。固定流程反复运行、每个 kernel 又很短时，通常更值得尝试。
