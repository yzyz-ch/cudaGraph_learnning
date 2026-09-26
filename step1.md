# CUDA Graph：先会用，再深入

你已经完成 01–04，理解了捕获、重放和加速原理。下一阶段从 05 开始，学习参数更新、流程变化和更完整的图。

## 学习计划与当前进度

| 阶段 | 小节 | 学完能做什么 | 进度 |
| --- | --- | --- | --- |
| 入门 | 01 捕获与重放；02 复用数据；03 性能对比；04 加速原理 | 录制并重复执行固定流程 | 已完成 |
| 参数变化 | [05 单节点更新](05_update_kernel_node.cu) | 把图里的“加 1”改成“加 10” | 接下来学习 |
| 参数变化 | [06 整图更新](06_update_whole_graph.cu) | 同时更改多个参数及 GPU 数据地址 | 待学习 |
| 流程变化 | [07 重新构建](07_rebuild_graph.cu) | 处理节点数变化及更新被拒绝的情况 | 待学习 |
| 完整流程 | [08 捕获数据复制](08_capture_memcpy.cu) | 一次重放完成输入复制、计算、输出复制 | 待学习 |
| 依赖关系 | [09 分支与汇合](09_graph_dependencies.cu) | 表达两条独立分支，以及需要等待两者的后续操作 | 待学习 |

**05–09 的讲解、编译命令、预期结果和练习见 [下一阶段学习说明](NEXT_STEPS.md)。** 建议先学 05–07，能区分“换数据、更新参数、重新构建”后，再学 08–09。每节先预测输出，再运行，再做一个小修改。

下面保留 01–04 的说明，方便回看。

## 先抓住本质

**CUDA Graph 提前记录一组 GPU 操作及其先后关系，以后用一次 Graph launch 提交整个流程。**

以“乘 2，再加 1”为例：

```text
普通方式，每轮 CPU 提交：  launch scale → launch bias
Graph 方式，每轮 CPU 提交：launch graph
                               └─ GPU 执行：scale → bias
```

Graph 中仍有两个 kernel。它优化重复提交和调度的开销，不会自动把两个 kernel 融合，也没有减少乘法、加法的计算量。固定流程反复运行、每个 kernel 又很短时，通常更值得尝试；如果计算本身很耗时，收益可能不明显。[官方原理说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#cuda-graphs)

先认识这些词就够了：

| 术语 | 在本例中是什么意思 |
| --- | --- |
| stream（流） | 按顺序执行 GPU 任务的队列 |
| node（节点） | 图里的一个操作，例如一次 `scale` kernel |
| dependency（依赖） | 先完成 `scale`，再执行 `bias` |
| capture（捕获） | 把提交到流上的操作记录成图 |
| instantiate（实例化） | 将图准备成可执行的形式，主要做一次 |
| replay（重放） | 再次执行这张图；对应 `cudaGraphLaunch` |

## 01：录一次，跑三次

文件：[01_stream_capture.cu](01_stream_capture.cu)

**解决的问题：** 怎样把现有的两次 kernel launch 改成 Graph？

核心只有这几步，内存分配、错误处理和销毁都在完整代码中：

```cpp
// 1. 记录流程。这时两个 kernel 都不执行。
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
scale<<<1, n, 0, stream>>>(d, n);
bias<<<1, n, 0, stream>>>(d, n);
cudaStreamEndCapture(stream, &graph);

// 2. 准备可执行图。graph 是流程定义，exec 是可执行版本。
cudaGraphInstantiate(&exec, graph, 0);

// 3. 重复执行。每一轮使用上一轮留下的数据。
for (int i = 0; i < 3; ++i) {
    cudaGraphLaunch(exec, stream);
}
cudaStreamSynchronize(stream);  // CPU 等待 GPU 做完，再读结果。
```

`<<<1, n, 0, stream>>>` 的最后一项指定流；前面依次是 block 数、每个 block 的线程数、动态共享内存字节数。

`cudaStreamCaptureModeGlobal` 是捕获期间的安全检查模式，并不表示把全程序的操作都录下来。例子只向这条流提交两个 kernel，它们的执行顺序会成为图中的依赖。

```bash
mkdir -p build
nvcc -O2 -std=c++14 01_stream_capture.cu -o build/01_stream_capture
./build/01_stream_capture
```

预期输出：

```text
after 3 replays: 15 15 15 15
PASS (expected each element: 15)
```

每个元素都从 `1` 开始：`1 → 3 → 7 → 15`。捕获不是额外执行一次，所以不会得到 `31`。

**小练习：** 将 `repeats` 改成 `1`，重新编译运行，应得到四个 `3`。代码中的 CPU 校验会跟着重放次数变化。

## 02：图不变，换一批输入

文件：[02_reuse_graph.cu](02_reuse_graph.cu)

**解决的问题：** 每次输入数据不同，是否就必须重新建图？

本例分配同一块 GPU 内存 `d`，只构建一次图，然后重复：

```cpp
cudaMemcpy(d, new_input, bytes, cudaMemcpyHostToDevice);
cudaGraphLaunch(exec, stream);
cudaStreamSynchronize(stream);
cudaMemcpy(output, d, bytes, cudaMemcpyDeviceToHost);
```

复制操作在捕获区域外；上面是流程示意，完整代码中的输入为 `inputs[batch]`。

**图记录指针地址，不会把该地址中的数据冻结。** 因此，把新输入放入同一个地址，再重放即可。每轮结束后才读取结果或覆盖输入。

这里要区分三种改动：

| 改动 | 已有图的行为 |
| --- | --- |
| 把新数据复制到原来的 `d` | kernel 会处理新数据 |
| CPU 上把变量 `d` 改成另一个地址 | 图仍然使用捕获时的地址 |
| CPU 上修改传给 kernel 的标量参数，例如 `n` | 图仍然使用捕获时的参数值 |

后两种情况需要更新图或重新构建图，继续学习时见 [05–07 的讲解与示例](NEXT_STEPS.md)。[官方参数捕获说明](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)

```bash
mkdir -p build
nvcc -O2 -std=c++14 02_reuse_graph.cu -o build/02_reuse_graph
./build/02_reuse_graph
```

预期输出：

```text
batch 1: 3 5 7 9
batch 2: 11 13 15 17
PASS
```

与例 01 的区别：01 累积计算；02 每轮用新输入覆盖原数据，独立算一次 `x * 2 + 1`。

**小练习：** 把第二组输入改为 `{0.f, 10.f, 20.f, 30.f}`，输出应为 `1 21 41 61`，不需要改构图代码。

## 03：看看能节省多少时间

文件：[03_launch_vs_graph.cu](03_launch_vs_graph.cu)

**解决的问题：** 相同的工作，Graph 是否比普通 launch 快？

每个 kernel 给 256 个元素各加 1；每轮执行 20 个 kernel，重复 1000 轮。

| 方式 | 每轮 CPU 提交 | GPU 的工作 |
| --- | --- | --- |
| 普通 launch | 20 次 kernel launch | 20 个加法 kernel |
| Graph | 1 次 graph launch | 相同的 20 个加法 kernel |

这里 capture 区域里的 CPU `for` 循环会记录出 **20 个节点**，图中没有一个会自行执行的 CPU 循环。

代码的计时流程是：

```text
分配内存、构图 → 预热 → 清零并等待完成
    → 开始 CPU 计时 → 提交 1000 轮 → 等 GPU 完成 → 停止计时
    → 复制结果并校验
```

两条路径分别预热 10 轮、清零后再计时。普通路径不会每个 kernel 都同步；两条路径都只在整批提交结束后等待 GPU 完成。

```bash
mkdir -p build
nvcc -O2 -std=c++14 03_launch_vs_graph.cu -o build/03_launch_vs_graph
./build/03_launch_vs_graph
```

应看到 `normal` 和 `graph` 两行都为 `PASS`，每个元素都是 `20000`。计时数字依机器和当时负载变化。

- `us/round`：平均一轮耗时，单位微秒；一轮包含 20 个 kernel。
- `speedup`：普通方式耗时 ÷ Graph 耗时；大于 1 表示这次 Graph 更快。
- CPU 计时包含提交工作及等待 GPU 完成；CPU 提交与 GPU 执行可以重叠，所以这个数不是两者耗时的简单相加，也不是单次 launch 的纯开销。
- 构图、实例化、内存分配和数据复制不计入测量。它展示的是复用图后的耗时；真正应用还需考虑前期准备成本。

这是观察现象的小实验，不承诺固定加速比。短测量会受显卡频率、其他进程和系统调度影响，也可能测到 Graph 更慢。[官方性能示例](https://developer.nvidia.com/blog/cuda-graphs/)

**小练习：** 将 `steps` 从 `20` 改成 `1`，再编译运行。每轮两种方式都只需一次提交，Graph 优势可能缩小；结果应为 `1000`。不要用一次测量判断所有场景。

## 04：计算量不变，为什么 Graph 能更快？

文件：[04_why_graph_is_faster.cu](04_why_graph_is_faster.cu)。先读文件开头的中文解释，再运行。

**原理是提前准备、重复利用：** 普通方式需要 CPU 逐次调用 CUDA 提交 kernel；Graph 提前记录整个流程，在实例化时做一部分准备，以后一次调用就可以启动这套流程，减少重复的提交和调度工作。GPU 可以按已准备好的依赖关系执行后续节点，而不必由 CPU 对每个节点重新调用 launch。[官方原理说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#cuda-graphs)

这也可能缩短 kernel 之间的调度间隔。普通 launch 本身也是异步的，CPU 可以连续提交任务；Graph 进一步改善的是组织执行这些任务的开销。[NVIDIA 性能分析](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)

这次固定 **20000 个 kernel**，每个给 256 个元素各加 1，只改变分组方式：

| 方式 | 每次提交包含几个 kernel | 正式计时内的 launch 调用次数 | 本次总耗时（毫秒） |
| --- | --- | --- | --- |
| 普通 launch | 1 | 20000 | 88.284 |
| Graph | 1 | 20000 | 135.334 |
| Graph | 5 | 4000 | 39.948 |
| Graph | 20 | 1000 | 25.148 |
| Graph | 100 | 200 | 15.436 |

2026-09-26，本机 CUDA 12.6 编译无警告，五种方式运行全部为 `PASS`。上表为这次实测；频率、其他进程和测试顺序都会影响时间，不要求你复现相同数字。

每个 kernel 仍独立执行，全部方式的结果都必须是 `20000`。这里的“提交次数”是程序调用 kernel launch 或 `cudaGraphLaunch` 的次数，不代表驱动内部向硬件提交命令的次数。

```bash
mkdir -p build
nvcc -O2 -std=c++14 04_why_graph_is_faster.cu -o build/04_why_graph_is_faster
./build/04_why_graph_is_faster
```

输出中：

- `kernels/submit`：一次提交包含多少个 kernel。
- `submits`：提交多少次。两列相乘始终为 `20000`。
- `submit_ms`：CPU 从开始到提交循环结束经过的毫秒数。
- `total_ms`：从同一起点到 GPU 全部完成经过的毫秒数。
- `check`：所有 256 个元素是否正确，应全部为 `PASS`。

**两种时间不能相加：CPU 提交期间 GPU 就可能已经在计算。** `submit_ms` 也不等于纯 CPU launch 开销；队列拥塞时，launch 调用可能等待。仅靠这两个数不能精确区分 CPU 开销和 GPU 调度开销，也不能证明 kernel 之间的间隔缩短了多少。

重点看三件事：

1. 单节点图没有减少提交次数，因此可能与普通方式相近，也可能更慢。
2. 多节点图减少了重复提交的次数，可能缩短完成同样工作的时间。
3. 图变大后不一定一直更快；GPU 的计算和调度仍然需要时间，不能随着提交次数减少而无限加速。

构图、实例化、预热和数据复制都在计时外，因此单节点图更慢不能归因于这些准备成本。这里没有强行制造 kernel 间的等待，也不保证每次运行都呈现单调加速。

**小练习：** 在 `cases` 中增加 `10`，重新运行。应看到 `2000` 次提交、所有元素仍为 `20000`。新增分组大小需整除正式测试和预热的 kernel 数；程序会检查这一点。

## 套到自己的代码时，先记四点

1. 先找一段需要反复执行、参数和操作顺序稳定的 kernel 流程，把构图放到重复执行循环外。
2. 本例用显式创建的 stream；传统默认流 `cudaStreamLegacy` 不能捕获。所有待捕获 kernel 都传入这条流。
3. 本例将内存分配、同步复制和等待完成放在捕获区域外。不要在捕获中加入 `cudaStreamSynchronize` 来等待被记录的工作。
4. 图执行期间，用到的 GPU 内存必须保持有效；等执行完成、后续不再重放后，再释放内存。

上面的 stream 捕获限制可查阅 [CUDA 12.6 捕获说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture)。图也能包含其他类型的操作，这里先掌握最常用的单流 kernel 流程即可。

这些程序都检查 CUDA 错误和计算结果，失败会返回非零退出码。代码中的 `CHECK` 只是错误检查辅助，不属于 Graph API。为方便独立阅读，每个文件各自保留了这一小段辅助代码。

## 验证记录

05–09 已使用 CUDA 12.6 编译（启用 `-Wall,-Wextra`，无警告），并在 RTX 4070 Ti SUPER 上运行通过。包括 05 的三种参数状态、06 更新成功、07 结构不兼容后重新实例化、08 两批完整复制流程、09 分支汇合。完整输出见 [下一阶段学习说明](NEXT_STEPS.md)。

以下是入门阶段 01–03 使用原始默认参数时的记录；03 当时的 `steps` 为 `20`。

2026-09-26：使用 CUDA Toolkit 12.6（nvcc 12.6.85）、NVIDIA GeForce RTX 4070 Ti SUPER、驱动 566.03 验证。

- 三个示例均以 `-O2 -std=c++14 -Xcompiler -Wall,-Wextra` 编译成功，无编译警告。
- 01 输出四个 `15`，02 输出 `3 5 7 9` 和 `11 13 15 17`，均为 `PASS`。
- 03 两种方式都通过全部 256 个元素的校验，结果为 `20000`，本次计时如下。

| 普通方式 | Graph | 普通耗时 / Graph 耗时 |
| --- | --- | --- |
| 139.223 微秒/轮 | 25.239 微秒/轮 | 5.52 倍 |

这是单次实测，不是固定加速承诺。测试时显卡也在承担显示任务。

沙箱内 CUDA 初始化失败，以上运行结果是在获准访问 GPU 的沙箱外取得的。若你的受限环境也无法初始化 CUDA，请在能访问显卡的终端运行。

`build/` 存放本次编译结果，目录里原有的 `01_stream_capture` 可执行文件保留；请运行 `build/` 下的版本。
