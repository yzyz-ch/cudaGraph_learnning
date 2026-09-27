# 第三阶段：在实际程序中使用 CUDA Graph

前 11 节已经覆盖构图、重放和更新。这一阶段学习怎样组织多条流、管理正在使用的内存，以及验证功能和性能。

| 小节 | 先回答的问题 |
| --- | --- |
| 12 多流捕获 | 已有两条 stream 的程序，怎样整体录成图？ |
| 13 内存与同步 | 下一批输入什么时候能覆盖上一批的缓冲区？ |
| 14 更新能力测试 | 具体哪些变化可以更新，哪些变化会被拒绝？ |
| 15 计时与时间线 | 准备、更新、提交、执行分别花了多少时间？ |
| 16 节点开关 | 某一步偶尔不需要执行，能否直接跳过？ |
| 17 图缓存 | 在几个固定场景之间切换，能否避免反复构建？ |

全部是独立 `.cu` 文件，使用 CUDA 12.6。除 14 的测试矩阵外，每个例子约 70–120 行。先读中文注释，再看核心 API，不需要一次记下所有细节。

## 12：捕获两条 stream 的分支与汇合

文件：[12_multistream_capture.cu](12_multistream_capture.cu)

计算沿用 09：A 做 `2*x`，B 做 `ReLU(x)`，C 将两者相加。不同之处在于，这次通过 stream capture 建图。

```text
stream1：复制输入 → 记录 ready → A ─────────────→ 等 done → C
stream2：               等 ready → B → 记录 done
```

**event（事件）**可以理解为一个执行进度标记。本例不测 event 的时间，所以使用 `cudaEventDisableTiming` 创建它。

- `cudaEventRecord(ready, stream1)`：标记输入复制完成的位置。
- `cudaStreamWaitEvent(stream2, ready, 0)`：让 stream2 的后续工作依赖这个位置，并加入同一次捕获。
- B 后记录 `done`，stream1 等待 `done` 后再提交 C：把分支汇合回来。

`cudaStreamWaitEvent` 表达 GPU 工作之间的依赖，不是让 CPU 在这里等待。捕获期间，这些操作记录的是图中的关系；重放时才执行工作。

**在哪条流 BeginCapture，就在哪条流 EndCapture；加入捕获的分支需要先汇合回来。** 两条流是同一张图的一部分，不需要分别对它们 BeginCapture。[官方多流捕获说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#cross-stream-dependencies-and-events)

```bash
mkdir -p build
nvcc -O2 -std=c++14 12_multistream_capture.cu -o build/12_multistream_capture
./build/12_multistream_capture
```

预期：`两条流捕获后的结果: -4 -2 0 9  PASS`。

A、B 没有互相等待的依赖，允许并行；四个数太少，本例不承诺实际重叠或性能提升。

**小练习：** 将输入改成 `{-3.f, 0.f, 2.f, 4.f}`，应得到 `-6 0 6 12`。

## 13：两套缓冲区轮流使用

文件：[13_buffer_lifetime.cu](13_buffer_lifetime.cu)

之前每次 `cudaGraphLaunch` 后都马上同步。这次准备两套独立资源，第一套处理一批时，CPU 可以准备第二套。

每套都有：CPU 输入、CPU 输出、GPU 内存、stream、exec 和一个完成事件。图内流程为 `输入复制 → x=2*x+1 → 输出复制`。

```text
第 1 批 → 缓冲区 1，提交后继续
第 2 批 → 缓冲区 2，提交后继续
第 3 批 → 等缓冲区 1 的第 1 批完成、读出结果，再填新输入
第 4 批 → 等缓冲区 2 的第 2 批完成、读出结果，再填新输入
```

关键是把完成事件记录在图后面：

```cpp
cudaGraphLaunch(execs[slot], streams[slot]);
cudaEventRecord(done[slot], streams[slot]);
```

复用同一套缓冲区前，调用 `cudaEventSynchronize(done[slot])`。这里 CPU 才真正等待，等的是该事件之前的工作，而不是全设备的所有工作。

**每套输入、输出和 GPU 内存都必须保持有效，直到相关工作完成。** 不能刚提交就覆盖输入，也不能只保留 `exec` 却把它使用的 GPU 内存释放。程序退出前还要等待最后两批，读出结果后再释放资源。

```bash
mkdir -p build
nvcc -O2 -std=c++14 13_buffer_lifetime.cu -o build/13_buffer_lifetime
./build/13_buffer_lifetime
```

六批结果分别为：

```text
第 1 批，缓冲区 1: 1 3 5 7  PASS
第 2 批，缓冲区 2: 21 23 25 27  PASS
第 3 批，缓冲区 1: 41 43 45 47  PASS
第 4 批，缓冲区 2: 61 63 65 67  PASS
第 5 批，缓冲区 1: 81 83 85 87  PASS
第 6 批，缓冲区 2: 101 103 105 107  PASS
```

这是一种安全复用缓冲区的安排，不是“必然加速”的基准。不要通过删除同步来验证错误：未完成的异步访问可能造成偶发错误，甚至碰巧得到正确结果。

**小练习：** 把 `batches` 从 6 改为 2，仍应输出前两批且全部通过。观察循环结束后的等待为何不可省略。

## 14：逐项测试更新支持范围

文件：[14_update_support.cu](14_update_support.cu)

每项测试都重新建立基准图和 exec，只改变一个因素，然后检查 API 的结果和所有输出元素。这样前一项更新不会干扰后一项。

先读 `kernel_case`，再读 `memcpy_case`，最后读 `rejected_case`。`CHECK` 检查 CUDA 错误；`REQUIRE` 检查测试预期，任一意外失败都会打印位置并以非零退出码结束程序。

两种更新方式分别测试，结果不能互相代替：

| 方式 | kernel 使用的 API | memcpy 使用的 API |
| --- | --- | --- |
| 单节点更新 | `cudaGraphExecKernelNodeSetParams` | `cudaGraphExecMemcpyNodeSetParams1D` |
| 整图更新 | `cudaGraphExecUpdate` | `cudaGraphExecUpdate` |

本机 CUDA 12.6 下，以下变化在两种方式中都应成功：

| 测试 | 原配置 → 新配置 |
| --- | --- |
| kernel 数值参数 | 加 1 → 加 3 |
| kernel 输入地址 | 正数数组 → 另一块负数数组 |
| 元素数量 | n=128 → n=64，尾部保持清零状态 |
| grid | 2 个 block → 4 个 block |
| block | 每个 block 128 线程 → 64 线程 |
| 动态共享内存 | 512 字节 → 1024 字节 |
| kernel 函数 | `x+value` → `2*x+value` |
| memcpy 源地址 | 原 GPU 输入 → 另一块 GPU 输入 |
| memcpy 目标地址 | 原 GPU 输出 → 另一块 GPU 输出 |
| memcpy 大小 | 128 个 float → 64 个 float |

配置仍需合法：例如共享内存必须够用、grid 与 block 必须覆盖本例需要处理的数据、缓冲区不能越界。grid/block/shared memory 测试验证 API 接受该配置且结果正确，不是在比较它们的性能。

另外三个测试只通过整图更新表达，预期被拒绝：

| 变化 | 预期详细原因 |
| --- | --- |
| Kernel 节点变为 Memcpy 节点 | `cudaGraphExecUpdateErrorNodeTypeChanged` |
| 一个节点变成两个 | `cudaGraphExecUpdateErrorTopologyChanged` |
| 去掉两个节点之间的依赖 | `cudaGraphExecUpdateErrorTopologyChanged` |

三项都应返回 `cudaErrorGraphExecUpdateFailure`。代码核对原因、处理这次已知错误后，继续运行原 exec 并检查结果。依赖缺失的新图不被执行，避免运行一个可能有读写冲突的流程。

**测试通过不一定代表更新成功：对于预期拒绝的项，正确拒绝也是通过。** 这里采用“允许的项必须成功，不允许的项必须按指定原因失败”的验证方式。

```bash
mkdir -p build
nvcc -O2 -std=c++14 14_update_support.cu -o build/14_update_support
./build/14_update_support
```

最后应看到：`23 项测试全部 PASS（包括 3 项预期拒绝）。`

范围是同一设备上的普通 kernel，以及同一设备内的一维复制（D2D，即 GPU 内存到 GPU 内存）。没有覆盖所有复制方向、二维数组、多设备、条件节点等情况；不能把此表当作所有 CUDA Graph 的通用兼容表。[官方 API 限制](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__GRAPH.html)

**小练习：** 把数值参数测试里的 `value = 3.f` 改成 `5.f`。测试仍应通过，因为 CPU 校验也使用本项的新参数值。

## 15：分开测量准备、更新和执行

文件：[15_graph_timing.cu](15_graph_timing.cu)

图中有 20 个“加一个数”的节点。它先运行原图，再将第一个节点改为加 2，其余节点仍加 1，所以更新后每轮增加 21。

程序依次测量：

| 输出 | 计时范围 |
| --- | --- |
| 构图 | 创建 graph、添加 20 个节点和依赖 |
| 实例化 | 一次 `cudaGraphInstantiate` |
| 首次提交 | 这个 exec 第一次 `cudaGraphLaunch` 调用经过的时间 |
| 首次到完成 | 从首次提交前，到等待 GPU 完成 |
| 单节点更新平均 | 1000 次单节点更新循环总耗时除以 1000 |
| 重放提交阶段 | 提交 1000 轮经过的时间，包含起始 event 的记录 |
| 重放到全部完成 | 从同一起点，到末尾 event 完成 |
| GPU event 区间 | GPU 执行起始、结束两个 event 之间的时间 |

构图前先初始化 CUDA 和 kernel，因此“首次”指本 exec 的首次运行，不是整个进程的冷启动。稳定重放前已预热，并在更新后再次预热；清零、分配和结果复制都不包含在重放计时里。

这里更新的只是第一个节点，计时指标不代表整图更新耗时。CPU 计时也包含小量循环和错误检查开销。

**这些时间不能相加。** CPU 提交与 GPU 执行会重叠；提交阶段可能包含队列拥塞导致的等待；GPU event 区间也可能包含“GPU 等待 CPU 提交后续工作”的空闲时间，它不等于所有 kernel 纯计算时间之和。

```bash
mkdir -p build
nvcc -O2 -std=c++14 15_graph_timing.cu -o build/15_graph_timing
./build/15_graph_timing
```

数字会随环境变化，默认配置最终应输出：`最终每个元素应为 21000: PASS`。

### 用 Nsight Systems 看实际执行过程

本机已单独升级到 Nsight Systems CLI 2026.5.1，CUDA Toolkit 仍为 12.6，驱动仍为 566.03。新版位于 `~/.local/opt/nsight-systems/2026.5.1`，`nsys` 默认使用新版；原来随 CUDA 安装的 2024.5.1 保留。已有终端先执行 `source ~/.bashrc`，再用 `nsys --version` 检查版本。

先对 12 采集，图较小，容易看清 A、B、C：

```bash
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --output=build/12_multistream_trace_nsys2026_5 ./build/12_multistream_capture
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum build/12_multistream_trace_nsys2026_5.nsys-rep
```

本次已生成报告，重复采集时可以改用另一个输出文件名。**升级后验证通过：12 的报告包含 `scale`、`relu`、`add_arrays` 各一次 GPU 执行记录；15 的报告包含 20,421 次 `add_value` 执行记录（包含初始化、首次执行和预热）。两个程序的数值检查均为 PASS。**

之前 2024.5.1 的报告缺少 GPU kernel 活动，并提示不支持驱动报告的 CUDA 12.7 版本。升级后的报告不再出现这个版本警告，GPU kernel 时间线已恢复。当前环境使用软件方式采集，报告还提示统一内存追踪受限；这不影响本次两个示例的 kernel 采集。

使用 Nsight Systems 2026.5.1 或更高版本的图形界面打开新版 `.nsys-rep` 文件，放大 GPU 时间线。本次只安装了 WSL 命令行工具；查看报告的图形界面版本应不低于采集版本。[官方版本兼容说明](https://developer.nvidia.com/nsight-systems/get-started#supported-platforms)

1. 找到输入复制和 `scale`、`relu`、`add_arrays` 三个 kernel。
2. 看 A、B 的时间区间是否重叠；没有重叠也符合依赖定义。
3. 看 C 是否在两者完成后开始。
4. 看 CPU 上的 `cudaGraphLaunch` 调用与 GPU 工作的时间关系。

`stats` 给的是汇总统计，判断重叠需要看时间线。要观察 15，可把采集命令的程序改成 `./build/15_graph_timing`，输出改成 `build/15_graph_trace_nsys2026_5`。

`--cuda-graph-trace=node` 记录各节点活动，会引入额外开销。用它理解执行关系；比较正常性能时，使用不带 profiler 的运行结果。[Nsight Systems 官方说明](https://docs.nvidia.com/nsight-systems/UserGuide/)

**小练习：** 把 `repeats` 从 1000 改成 100，最终应为 `2100`。观察总耗时与准备耗时的变化，不要求时间严格缩短到十分之一。

## 16：启用和禁用已有节点

文件：[16_enable_node.cu](16_enable_node.cu)

先构建 `乘 2 → 加 10`。图一直包含这两个节点，每轮通过下面的调用决定是否执行第二步：

```cpp
cudaGraphNodeSetEnabled(exec, optional, 1);  // 启用
cudaGraphNodeSetEnabled(exec, optional, 0);  // 禁用
```

禁用的节点相当于一个不做事的空节点，其依赖位置保留，节点参数也保留；重新启用后恢复执行。它不是删除节点，因此这里不需要重新实例化。[官方节点开关说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__GRAPH.html)

```bash
mkdir -p build
nvcc -O2 -std=c++14 16_enable_node.cu -o build/16_enable_node
./build/16_enable_node
```

预期输出：

```text
第 1 次执行，加 10 节点启用: 6 8 10 16  PASS
第 2 次执行，加 10 节点禁用: -4 -2 0 6  PASS
第 3 次执行，加 10 节点启用: 6 8 10 16  PASS
```

每轮恢复同一份输入，所以不会累积结果。程序同时查询节点开关状态并校验数值。

本例的加法是原地可选操作，跳过后数据仍有效。如果某节点负责生成后续节点必需的输入，不能简单禁用它后还期望下游读到新数据。

**小练习：** 把 `enabled` 改成 `{0, 0, 1}`，前两次应为 `-4 -2 0 6`，第三次仍为 `6 8 10 16`。

## 17：保存两张图，按输入规模选择

文件：[17_graph_cache.cu](17_graph_cache.cu)

**图缓存**就是把准备好的 exec 留着下次再用。本例只支持 n=4 和 n=8 两种输入规模，使用长度为 2 的数组保存资源，不引入容器或缓存框架。

```text
请求：     4       8       4       8       4
动作：   建图4   建图8   复用4   复用8   复用4
```

每个条目同时保留自己的 GPU 缓冲区。新的输入先复制到该地址，再启动对应 exec，因此图里的指针不会因为请求改变而失效。

```bash
mkdir -p build
nvcc -O2 -std=c++14 17_graph_cache.cu -o build/17_graph_cache
./build/17_graph_cache
```

五次执行均应 `PASS`，最后输出 `构建次数: 2`。第一次结果为 `1 3 5 7`，第五次为 `81 83 85 87`，说明复用图仍在处理新的数据。

本例只用 n 区分条目，因为函数、数据类型、设备都固定，内存也由条目自己管理。真实程序要把所有影响图复用的因素考虑进去，不能把“n 相同”当作万能条件。

这里在同一条流上逐次执行并同步，展示的是图的选择与复用，不是请求并发。多个 exec 会占用额外资源，缓存规模增长时还需要决定什么时候释放不用的条目；释放前须确保相关工作完成。

输入大小改变并不必然要求多张图：也可以选择更新参数。缓存是用更多资源换取直接复用的一种方案，是否更合适要结合场景。

**小练习：** 将请求序列改成 `{4, 4, 4, 4, 4}`，应全部通过，最后构建次数变成 1。

## 验证记录与学习完成标准

2026-09-27，CUDA Toolkit 12.6.85、RTX 4070 Ti SUPER、驱动 566.03。

12–17 均以 `-O2 -std=c++14 -Xcompiler -Wall,-Wextra` 编译无警告，默认配置运行通过。14 的 23 项测试均符合预期；13 正确处理了 6 批缓冲区复用；16 验证了禁用再启用；17 只构建两次。

六节的小练习也已在临时副本中编译运行通过，包括 13 仅提交两批后的收尾等待、16 改变开关序列、17 仅使用一种规模时只构建一次。目录中的源文件保留默认配置。

15 一次未启用 profiler 的实测如下，仅记录这次结果：

| 指标 | 实测 |
| --- | --- |
| 构图 | 452.127 us |
| 实例化 | 61.775 us |
| 首次提交 / 首次到完成 | 230.926 / 252.563 us |
| 单节点更新平均 | 0.086 us/次 |
| 1000 轮提交 / 到全部完成 | 11.932 / 17.322 ms |
| GPU event 区间 | 16.776 ms |

单独升级 Nsight Systems CLI 到 2026.5.1 后，已重新生成 `build/12_multistream_trace_nsys2026_5.nsys-rep`、`build/15_graph_trace_nsys2026_5.nsys-rep`，位于 git 忽略的 `build/` 目录。已确认可查看 CPU API 统计和 GPU kernel 活动，分别记录到 3 和 20,421 次 kernel 执行；原有缺少 GPU 活动的旧报告保留。是否实际并行仍需查看 A、B 的时间区间，不能仅凭汇总次数判断。

完成这一阶段后，应能独立说明：多流怎样汇合成一张图、缓冲区何时可以复用、每个更新测试在改变什么，以及一项计时到底包含了哪些工作。
