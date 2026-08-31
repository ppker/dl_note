
CUDA 性能分析工具接口（CUPTI）提供了一个基于 C 语言的接口，用于创建针对 CUDA 应用程序的**性能分析和跟踪工具**。

CUPTI Python 是一个库，它为创建专门针对 CUDA Python 应用程序的剖析和跟踪工具提供了 Python API。**当前版本支持 CUPTI C Activity and Callback API 的子集**。需要注意的是，此库仅适用于 Linux（x86_64）和 Linux（aarch64 sbsa）平台。

CUPTI APIs 根据获取性能测试数据的类别又细分为两类：`tracing` 和 `profiling`。 官方文档中 CUPTI API 功能简述如图1.

- `tracing`: 针对收集整个 cuda 程序中的 cuda 函数的时间戳和附加信息，有助于确定 CUDA 代码中哪些部分需要较长时间，其包含 Activity 和 Callback API 两种。
- `profiling`: 针对单独的或一系列的 kernel (核函数)，收集 kernel 的性能指标，使用 replay (回放机制) 收集 gpu 上性能指标，除上述两种 APIs，其他 API 都属于 profiling。

两者区别：**tracing 是系统级，侧重于了解整个程序的执行情况；profiling 是 kernel 级别，侧重于收集与 kernel 相关的性能指标**。类似 NVIDIA Nsight 两款软件 System 和 Compute 之间的关系。

## 一 CUPTI 原理

CUPTI 的核心思想是**事件驱动的性能数据收集**。它通过在软件栈的不同层面（Runtime API、Driver API、芯片硬件）插入“钩子”（`Hooks`）或“回调”（`Callbacks`），当特定的事件发生时（例如 API 调用、Kernel 启动、内存拷贝完成），CUPTI 会被通知，然后它可以执行相应的操作，例如：

- **记录时间戳**：在事件开始和结束时捕获高精度的时间戳，形成活动时间线（Timeline）。
- **读取硬件性能计数器**：在 Kernel 执行前后，读取硬件 `PMU` (Performance Monitoring Unit) 的计数器值，通过差值计算出该 Kernel 执行期间的硬件事件发生次数（如 `L1 Cache Miss`, 指令执行数等）。
- **收集元数据**：记录事件的上下文信息，如 API 参数、Kernel 名称、线程块配置等。

## 二 CUPTI API 汇总

> 以下函数分为几个类别：Activity API、Callback API、Event API、Metric API、Sampling / PC Sampling /辅助查询等。

### 1. Activity API （来自 `cupti_activity.h`）

API 描述：能够以异步方式 (asynchronous) 收集 (record) CUDA 程序中各种函数 (driver, runtime, kernel 等) 时间戳和附加信息。

CUPTI Activity API 收集性能参数逻辑：维护一个类似队列性质的数据容器 Buffer，整体逻辑如图2。

- 图2 上半部分：对于一个 CUDA 程序可能包含 driver， runtime 和 kernel 函数，这些函数分别运行在 host 和 device 之上，并存在一个程序自身运行的时间线 time_1。
- 图2 下半部分：CUPTI 自己会创建一个独立于原始程序的 CUPTI Thread (工作线程) ，能够在较小影响原始程序执行的情况下，完成性能参数的收集. Collect 过程 (图中 Collect)：CUPTI 能够根据开发者的需要，自动按照 CUPTI 头文件中固定的数据结构构造 record (记录)，并将指向 record 的指针放入创建好的 Buffer (类似队列容器) 中，创建 record 的时机是 CUDA 函数执行后，其余数据成员 CUPTI 会自动填充到 record 中. CUPTI Thread 位于另一条时间线 time_2(独立于 time_1)。

record 数据结构中存放了一些描述函数运行情况的数据成员，例如：函数种类，函数执行的起止时间，线程值，进程值等. 得到 record (图中 Check & Pop)：CUPTI 能够按照开发者的定义，按顺序 (record 加入 queue 中) 检查 queue 中每一个 record 数据结构，如果数据信息收集完整，则从 queue 中得到 record。

![cupti_activity_logic](../../images/cupti_api/cupti_activity_logic.jpg)

| 函数 / 接口 | 功能简述 |
|---|---|
| `cuptiActivityEnable(CUpti_ActivityKind kind)` | 启用指定类型的 Activity 记录（kernel, memcpy, memset, PC Sampling, etc.） |
| `cuptiActivityDisable` | 禁用某种 Activity 类型 |
| `cuptiActivityEnableContext(CUcontext context, CUpti_ActivityKind kind)` | 针对指定 context 启用某种 activity |
| `cuptiActivityDisableContext(CUcontext, CUpti_ActivityKind)` | 禁用 context 的某种 activity |
| `cuptiActivityRegisterCallbacks(buffRequested, buffCompleted)` | 注册 buffer 请求 / buffer 完成的回调，用于异步获取 record buffers。 |
| `cuptiActivityFlushAll(uint32_t flags)` | 请求把所有 pending 的 activity record buffer 送回（flush）给用户。|
| `cuptiActivityFlushPeriod(uint32_t time)` | 设置自动 flush 周期（时间间隔）使 worker thread 定期 flush buffers。 |
| `cuptiActivityGetNextRecord(uint8_t *buffer, size_t validSize, CUpti_Activity **record)` | 从一个 buffer 中迭代取出 activity records。|
| `cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t streamId, size_t *dropped)` | 查询因 buffer 不足等原因被丢弃的 record 数量。|
| `cuptiActivityGetAttribute(CUpti_ActivityAttribute attr, size_t *valueSize, void *value)` | 查询 Activity API 的属性（例如是否支持某种 activity kind、buffer 大小等设置）。 |
| `cuptiActivitySetAttribute(CUpti_ActivityAttribute attr, size_t *valueSize, void *value)` | 设置 Activity API 的某些属性。 |
| `cuptiActivityRegisterTimestampCallback(funcTimestamp)` | 注册一个回调，用于使用用户提供的 timestamp 方法／时钟替代默认 timestamp。 |
| `cuptiComputeCapabilitySupported(int major, int minor, int *support)` | 检查某个 compute capability 是否被当前 CUPTI 支持。 |
| `cuptiDeviceSupported(CUdevice dev, int *support)` | 检查某个设备是否被支持。  |
| `cuptiDeviceVirtualizationMode(CUdevice dev, CUpti_DeviceVirtualizationMode *mode)` | 查询设备是否在某种虚拟化模式下（比如是否被虚拟 GPU 等）。 |
| `cuptiGetContextId(CUcontext context, uint32_t *contextId)` | 获取 context 的 id。 |
| `cuptiGetDeviceId(CUcontext context, uint32_t *deviceId)` | 获取 context 所属 device 的 device id。|
| `cuptiGetStreamId(CUcontext, CUstream, uint32_t *streamId)` | 获取 stream 的 id。 |
| `cuptiGetStreamIdEx(CUcontext, CUstream, uint8_t perThreadStream, uint32_t *streamId)` | 拓展版本的 stream id 获取（更细粒度／每线程流等）。  |
| `cuptiGetTimestamp(uint64_t *timestamp)` | 获取一个时间戳（CPU／Host 时钟）／默认 timestamp 方法。|
| `cuptiGetLastError(void)` | 返回 CUPTI 最近的错误码。 |
| `cuptiFinalize(void)` | 结束 / 清理 CUPTI 相关资源，detach profiling 等。 |

### 2. Callback API (`cupti_callbacks.h` 或类似）

CUDA 事件回调机制，用于通知订阅者特定 CUDA 事件已执行，例如“进入 CUDA 运行时内存复制”。

| 函数 / 接口 | 功能简述 |
|---|---|
| `cuptiSubscribe(CUpti_CallbackFunc subscriber, void *userdata)` | 注册一个 subscriber，用以接收 callback 事件（runtime / driver / resource / synchronize / NVTX 等 domain）。 |
| `cuptiSubscribe_v2(...)` | 新版 subscribe（带更好的错误／旧订阅者检测等）。 |
| `cuptiUnsubscribe(CUpti_SubscriberHandle handle)` | 注销订阅者，不再接收 callback。 |
| `cuptiEnableCallback(uint32_t enabled, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain, CUpti_CallbackId callbackId)` | 在指定 domain / callbackId 上启用 callback 拦截（例如当调用某个 driver/runtime API 时触发）。  |
| `cuptiGetCallbackState(...)` | 查询某个 callback 是否已启用／状态信息。 |

### 3. Host Profiling 

**主机性能分析**：用于枚举、配置和评估性能指标的宿主 API。头文件: `cupti_profiler_host.h`。

### 4. Range Profiling  

范围分析: 收集执行范围性能指标的目标 API。头文件: `cupti_range_profiler.h`

### 5. PC Sampling类 API

warp 程序计数器和 warp 调度器状态（阻塞原因）的采样。

| 函数 / 接口 | 功能简述 |
|---|---|
| `cuptiActivityConfigurePCSampling(const CUpti_ActivityPCSamplingConfig *config)` | 配置 PC Sampling（采样频率／周期／采样类型等）若硬件支持。 |
| `cuptiDeviceGetNumDevices(int *deviceCount)` | 查询系统中的 CUDA 设备数目 |
| `cuptiDeviceGetProperties(CUdevice device, cuptiDeviceProps *props)` | 获取设备属性（例如 name / compute capability / clock / memory size 等） |
| `cuptiDeviceEnumMetrics(CUdevice device, uint32_t *numMetrics, CUpti_MetricID *metrics)` | 枚举设备支持的全部 metrics |
| `cuptiEventGetAttribute` / `cuptiEventDomainGetAttribute` | 查询事件域或事件的属性，像 instance count,是否可采样等 |
| `cuptiGetMetricValueFromEvents(...)` | (在某些版本中) 是计算 metric 的辅助／旧版函数 |

### 6. SASS Metrics

使用 SASS 补丁在源级别收集内核性能度量。头文件: `cupti_sass_metrics.h`

### 7. PM Sampling  

PM 采样：通过定期以固定间隔采样 **GPU 性能监控器**（`PM`）来收集硬件指标。头文件：`cupti_pmsampling.h`。

### 8. Profiling  

性能分析：用于收集各种执行性能指标的针对目标 API。**在 CUDA 13.0 版本中，性能分析 API 已弃用，建议使用范围性能分析 API**。头文件：cupti_profiler_target.h、nvperf_host.h。

### 9. Checkpoint  

检查点：提供对自动保存和恢复 CUDA 设备功能状态的支撑。头文件：cupti_checkpoint.h

### 备注

- 不同 CUDA Toolkit 版本（如 10.x / 11.x / 12.x）之间 API 接口可能增加 /改动 /弃用。  
- 某些函数仅在支持特定 GPU 架构的硬件上有效（Compute capability、驱动版本、是否启用虚拟化、是否支持采样等）。  

## 三 thrive 芯片的类 cupti 库

多 PE 架构芯片上实现性能监控的核心原理是利用硬件内置的**性能监控单元 (PMU) 和事件追踪机制**。

1. 硬件层支持：
    - **分布式计数器**： 在芯片内部的关键位置（如每个功能单元的输入/输出端、数据通路的交叉点、队列的读写端口）部署微型硬件计数器。这些计数器可以追踪数据包数量、字节数、繁忙周期、空闲周期、反压信号等。
    - **全局时间戳发生器**： 一个高精度、单调递增的硬件时钟，为所有性能事件提供统一的时间参考。
    - **事件触发逻辑**： 硬件模块能够检测并报告特定事件的发生（例如，一个操作完成、一个队列变为满状态）。
    - **硬件追踪缓冲区**： 一块专门的片上内存，用于存储按时间顺序排列的关键事件记录（时间戳、事件类型、相关 ID）。这允许进行非侵入式的数据流追踪。
2. **驱动程序层 (Kernel-mode Driver)**：
    - **硬件寄存器访问**： 驱动程序负责通过总线（如 PCIe）**读写 PMU 的控制和状态寄存器**。
    - **中断处理**： 处理 PMU 生成的中断（例如，追踪缓冲区满，需要驱动程序读取）。
    - **内存管理**： 将硬件的追踪缓冲区映射到内核空间，以便高效读取。
    - 数据流图上下文管理： 当数据流图被加载或执行时，驱动程序通知硬件，可能通过设置特定的硬件寄存器，让 PMU 能够关联性能数据与正在运行的图。
3. **用户空间库 (User-mode Library)**：
    - **API 封装**：提供一套抽象的 API 接口，封装了与驱动程序的交互细节。
    - **事件/度量定义**：维护一个已知的事件和度量指标列表，包括它们的名称、描述和计算公式（对于度量指标）。
    - **数据收集与解析**：从驱动程序获取原始计数器值和追踪数据，并解析成有意义的事件和活动记录。
    - **回调机制**：允许用户注册回调函数，以便实时或异步接收性能数据。
    - **数据后处理**：可能包含一些基本的数据处理逻辑，将原始数据转换为更高级的度量指标（例如，计算吞吐量、利用率）。
    - **上下文关联**：将收集到的性能数据与应用程序中正在执行的特定数据流图实例关联起来。

要实现我们自己芯片的 MyPEPTI (My PE Profiling Tools Interface) 库，其核心架构可以分为三层：

1. **硬件层** (Hardware)：芯片本身必须提供性能监控单元 (PMU)，包括各种性能计数器和高精度时钟。
2. **驱动层** (Driver)：驱动程序需要提供底层的接口，用于访问和控制硬件的 PMU、查询设备状态、并在关键路径（如命令提交、任务完成）上提供通知机制。
3. **分析库层** (MyPEPTI)：这个库作为中间件，连接上层应用/运行时 (Runtime) 和底层驱动。它负责**实现 Callback 机制、管理事件和计数器、聚合数据**，并向上层工具**提供统一的 API**。

### 3.1 需要了解的硬件信息

硬件需要提供**Performance Monitoring Unit (PMU)**的能力。

1. **全局高精度时钟 (High-Resolution Timestamp Counter)**：
   * **需求**: 必须有一个在整个芯片（或至少所有 PE 和控制器）中同步的、高频率的、不会轻易回绕的硬件时钟。
   * **对接问题**:
       * 时钟的频率是多少？（决定了时间精度）
       * 时钟寄存器的位宽是多少？（决定了回绕周期）
       * 如何从软件（通过驱动）读取这个时钟值？是内存映射的寄存器 (MMIO) 吗？
2. **可配置的硬件性能计数器 (Configurable Performance Counters)**：
  * **需求**: 每个 PE (或 PE Cluster) 都应该有多个可配置的硬件计数器。每个计数器都能被配置为对一种特定的硬件事件进行计数。
  * **对接问题**:
    * 总共有多少个计数器？
    * 这些计数器是每个 PE 独享，还是多个 PE 共享？
    * 每个计数器可以配置来统计哪些硬件事件？**需要一份详细的、可统计的硬件事件列表**。例如：
        * **周期类**: `active_cycles`, `stall_cycles`
        * **指令类**: `instructions_issued`, `fp32_ops`, `int32_ops`, `memory_load_ops`, `memory_store_ops`
        * **内存类**: `$L1$` Cache Hit/Miss, `$L2$` Cache Hit/Miss, Global Memory Read/Write Bytes, Shared Memory Bank Conflicts
        * **流水线类**: Stall by reason (e.g., `stall_on_memory_dependency`, `stall_on_instruction_fetch`)
        * **互联类**: NoC (Network-on-Chip) traffic in/out
    * 如何配置、启动、停止和读取这些计数器？（通过哪些寄存器操作）

3. **硬件 Trace 单元 (Optional, but powerful)**：
* **需求**: **更高级的硬件可以直接将事件和时间戳写入一个片上 SRAM 或指定的内存区域**，形成 Trace Log。这可以极大降低分析工具的性能开销。
* **对接问题**: 是否有这样的硬件单元？它的工作原理是什么？如何配置它的目标缓冲区、如何启用和解析它生成的数据？

### 3.2 需要驱动提供的接口

驱动是连接 `MyPEPTI` 库和硬件的桥梁。我们需要驱动提供非常底层的、细粒度的控制接口。

1. **PMU 访问接口**:
* **需求**: 驱动必须暴露接口，让 `MyPEPTI` 能够安全地控制硬件 PMU。
* **接口定义**:
    * `int my_driver_pmu_configure(pe_id, counter_id, event_id)`: 配置某个 PE 的某个计数器去统计某个事件。
    * `int my_driver_pmu_start(pe_mask, counter_mask)`: 在指定的 PE 组上启动指定的计数器。
    * `int my_driver_pmu_stop(pe_mask, counter_mask)`: 停止计数器。
    * `uint64_t my_driver_pmu_read(pe_id, counter_id)`: 读取计数器的值。
    * `uint64_t my_driver_get_timestamp()`: 读取全局硬件时钟。

2. **命令/任务生命周期通知**:

* **需求**: 这是实现时间线 (Timeline) 和精确 Kernel 性能分析的核心。当驱动向硬件提交一个任务（如一个 Kernel）时，`MyPEPTI` 需要知道。当这个任务在硬件上**真正开始执行**和**真正执行结束**时，`MyPEPTI` 必须得到通知。注意，这与主机侧的 API 调用返回是**不同步**的。
* **接口定义**:
  * **回调机制**: 这是最好的方式。驱动提供注册回调的接口。
    `int my_driver_register_task_callback(callback_type, function_pointer, user_data)`
    其中 `callback_type` 可以是 `TASK_SUBMIT` (提交到队列), `TASK_START_HW` (硬件开始执行), `TASK_FINISH_HW` (硬件执行完毕)。
  * **事件通知**: 如果回调不可行，可以使用事件（Event/Fence）机制。驱动在提交任务时可以返回一个“完成事件”句柄，`MyPEPTI` 可以查询该事件的状态。同时，驱动需要将任务开始和结束的时间戳写入一块共享内存，供 `MyPEPTI` 读取。

3. **设备拓扑和属性查询接口**:
* **需求**: `MyPEPTI` 需要知道芯片的架构信息。
* **接口定义**:
    * `int my_driver_get_device_properties(struct MyDeviceProps* props)`: 获取 PE 数量、时钟频率、内存大小等。
4. **上下文关联**:
* **需求**: 驱动提交的每个任务都应该有一个唯一的 ID。当驱动回调 `MyPEPTI` 时，需要传递这个 ID，以便 `MyPEPTI` 将硬件事件与上层的软件事件（如哪个 Kernel 调用）关联起来。

-----

### 3.3 需要收集的数据

1. **活动记录 (Activity Records)**: 用于构建时间线。
   * **Kernel 执行记录**: Kernel 名称、关联的上下文/流 ID、提交时间、硬件开始执行时间、硬件结束执行时间、Grid/Block 配置、使用的共享内存大小。
   * **内存拷贝记录**: 源地址、目标地址、大小、拷贝类型（H2D, D2H, D2D）、提交时间、开始时间、结束时间。
   * **Runtime API 调用记录**: 函数名称、参数、调用线程ID、开始时间、结束时间。

2. **性能指标 (Metrics)**: 从硬件计数器派生出的有意义的指标。
   * **原始事件值 (Raw Events)**: 例如 `$L1$_cache_misses`, `fp32_instructions`。
   * **派生指标 (Derived Metrics)**:
       * **IPC (Instructions Per Cycle)**: `instructions_issued / active_cycles`
       * **$L1$ Cache Miss Rate**: ` $L1$_cache_misses / ($L1$_cache_hits +  `$L1$\_cache\_misses)\`
       * **Occupancy**: 理论可达的并发度 vs 实际的并发度。
       * **Memory Bandwidth (GB/s)**: `(global_mem_read_bytes + global_mem_write_bytes) / kernel_duration_in_seconds`

### 总结

1.  **从对接开始**：必须从驱动和硬件那里获得稳定、可靠的底层接口和详尽的硬件文档。
2.  **分步实现**：
    * **第一步：实现活动追踪 (Activity Tracing)**。这是最直观、最有用的功能。只要驱动能提供 `Kernel` 和 `Memcpy` 的 `start/end` 回调和时间戳，就可以实现。
    * **第二步：实现事件/指标收集 (Event/Metric Collection)**。这需要驱动对 PMU 的完整支持。可以先支持几个关键的指标（如 IPC、Cache Miss），然后逐步扩展。
    * **第三步：API 回调**。这是更高级的功能，允许工具在运行时对程序行为进行干预或更细致的分析。
3.  **性能开销**：时刻关注 `MyPEPTI` 库自身的性能开销。数据收集（特别是高频事件）可能会影响目标程序的性能。考虑使用高效的无锁数据结构、批量处理数据、以及利用硬件 Trace 等技术来降低开销。
4.  

## 四 需要收集哪些数据

### 4.1 层次一：应用时间线与 API 追踪数据 (Host-Device Interaction)

这个层次的目标是理解应用程序的宏观行为，特别是主机（CPU）和你的加速设备（Multi-PE芯片）之间的交互。这通常是性能分析的第一步。

| 数据点 (Data Point)                 | 数据类型                 | 来源                           | 说明与分析价值                                                                   |
| ----------------------------------- | ------------------------ | ------------------------------ | -------------------------------------------------------------------------------- |
| **Runtime API 调用** | 字符串, `uint64_t` 时间戳 | `MyPEPTI` 库对 Runtime API 的 Hook | 记录 `mylibLaunchKernel`, `mylibMemcpy` 等函数的 **主机侧** 开始和结束时间。用于分析 API 调用开销、主机线程的同步点。 |
| **Kernel 执行** | 字符串, `uint64_t` 时间戳 | 驱动层回调 (`TASK_START/FINISH`) | 记录 Kernel 在 **设备上** 的实际开始和结束时间。与 API 调用时间对比，可看出任务排队延迟。这是时间线图的核心。   |
| **内存拷贝** | `uint64_t` 时间戳, size, kind | 驱动层回调                     | 记录 `memcpy` 在 **设备上** 的实际开始和结束时间、传输字节数、方向(H2D/D2H/D2D)。用于分析数据传输瓶颈。  |
| **上下文/流信息** | `uint32_t` ID            | Runtime API Hook/驱动          | 记录每个活动（Kernel, Memcpy）发生的上下文(Context)和流(Stream) ID。用于理解任务的并行和依赖关系。 |
| **关联 ID (Correlation ID)** | `uint32_t`               | `MyPEPTI` 库生成               | 一个唯一的ID，用于将主机侧的 API 调用（如 `mylibLaunchKernel`）与其对应的设备侧执行活动关联起来。至关重要！ |

**最终呈现形式**：一个类似 NVIDIA Nsight Systems 的 **Gantt 图**，清晰地展示了 CPU 和 PE 设备上的活动，可以轻松发现：
* 设备空闲时间（`Gaps`）：为什么我的 PE 芯片没在工作？
* 数据传输过长：是否可以和计算并行（Overlap）？
* 任务提交延迟：主机侧太慢还是驱动队列太深？

### 4.2 层次二：Kernel 级别性能摘要 (Kernel-Level Summary)

当你在时间线上发现一个耗时很长的 Kernel 后，下一步是深入分析这个 Kernel 本身的性能特征。

| 指标 (Metric)          | 计算公式/来源                                         | 说明与分析价值                                                                         |
| ---------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **执行时长 (Duration)** | `end_timestamp - start_timestamp`                     | 最基本的指标，性能优化的目标。                                                           |
| **占用率 (Occupancy)** | `(Active Warps/Threads per PE) / (Max Warps/Threads per PE)` | 衡量 PE 的计算资源（如寄存器、共享内存）是否被充分利用以隐藏延迟。低占用率可能是性能瓶颈的信号。 |
| **指令吞吐量 (IPC)** | `instructions_executed / active_cycles`               | 每周期执行的指令数。衡量 PE 流水线的效率。低 IPC 意味着流水线经常停顿（Stall）。           |
| **计算吞吐量 (Throughput)** | `(fp_ops / duration)` 或 `(int_ops / duration)`      | GFLOPS/s 或 GIOPS/s。衡量 Kernel 的有效计算能力，可以和芯片的理论峰值对比，评估计算效率。 |
| **内存带宽 (Bandwidth)** | `(global_mem_bytes_read + write) / duration`          | GB/s。衡量 Kernel 对全局内存的访问压力，可以和芯片的理论峰值对比，判断是否是内存带宽瓶颈。 |
| **停顿分析 (Stall Analysis)** | `stall_cycles / total_cycles`                       | PE 流水线停顿周期占总周期的比例。这是最重要的指标之一，直接指向性能瓶颈。              |


### 4.3 层次三：底层硬件架构指标 (Low-Level Architectural Metrics)

为了搞清楚 **为什么** IPC 低、**为什么** 会停顿，我们需要深入到硬件 PMU 提供的原始计数器。这些数据通常是针对单个 Kernel 执行期间收集的。

##### 计算核心 (Per PE) 指标

| 硬件事件 (Hardware Event)          | 数据类型   | 来源 | 说明与分析价值                                                              |
| ---------------------------------- | ---------- | ---- | --------------------------------------------------------------------------- |
| `active_cycles`                    | `uint64_t` | PMU  | PE 核心正在执行指令的周期数。                                               |
| `stall_cycles`                     | `uint64_t` | PMU  | PE 核心因各种原因停顿的周期数。                                             |
| `stall_reason_memory`              | `uint64_t` | PMU  | 因等待内存（L1, L2, Global Memory）而停顿的周期数。如果此值高，说明是访存瓶颈。 |
| `stall_reason_dependency`          | `uint64_t` | PMU  | 因指令数据依赖（例如，等待前一条指令的结果）而停顿的周期数。通常和编译器调度有关。 |
| `instructions_issued`              | `uint64_t` | PMU  | 发射的指令总数。                                                            |
| `fp32_instructions_executed`       | `uint64_t` | PMU  | 执行的 32 位浮点指令数。                                                    |
| `int32_instructions_executed`      | `uint64_t` | PMU  | 执行的 32 位整数指令数。                                                    |
| `branch_instructions`              | `uint64_t` | PMU  | 执行的分支指令数。                                                          |
| `divergent_branch`                 | `uint64_t` | PMU  | 发生线程束/Warp 内部分化的分支数。高分化会严重降低 SIMD/SIMT 架构的效率。   |

#### 3.2 内存子系统 (Memory Subsystem) 指标

| 硬件事件 (Hardware Event)                  | 数据类型   | 来源 | 说明与分析价值                                                                   |
| ------------------------------------------ | ---------- | ---- | -------------------------------------------------------------------------------- |
| `l1_cache_read_requests`                   | `uint64_t` | PMU  | L1 缓存收到的读请求总数。                                                      |
| `l1_cache_read_hits`                       | `uint64_t` | PMU  | L1 缓存读命中次数。                                                              |
| `l1_cache_read_misses`                     | `uint64_t` | PMU  | L1 缓存读未命中次数。**高 L1 Miss Rate** 意味着数据局部性差。                    |
| `l2_cache_read_requests`                   | `uint64_t` | PMU  | L2 缓存收到的读请求总数（通常来自 L1 的 Miss）。                               |
| `l2_cache_read_hits`                       | `uint64_t` | PMU  | L2 缓存读命中次数。                                                              |
| `l2_cache_read_misses`                     | `uint64_t` | PMU  | L2 缓存读未命中次数。**高 L2 Miss Rate** 意味着需要访问更慢的全局内存。        |
| `global_memory_read_bytes`                 | `uint64_t` | PMU  | 从全局内存（DRAM）读取的总字节数。                                             |
| `global_memory_write_bytes`                | `uint64_t` | PMU  | 写入到全局内存（DRAM）的总字节数。                                             |
| `shared_memory_bank_conflicts`             | `uint64_t` | PMU  | （如果你的架构有）共享内存的 Bank 冲突次数。高冲突数会使共享内存访问串行化，降低性能。 |


### 4.4 层次四：派生指标 (Derived Metrics)

原始的硬件计数器值通常不够直观。将它们组合起来，可以得到更具洞察力的派生指标。

| 派生指标名称          | 计算公式                                                         | 说明与分析价值                                                                         |
| --------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **L1 缓存命中率** | `l1_cache_read_hits / l1_cache_read_requests`                      | 衡量数据访问的空间和时间局部性。低命中率表明算法或数据结构需要优化。                     |
| **L2 缓存命中率** | `l2_cache_read_hits / l2_cache_read_requests`                      | 衡量更大范围数据访问的局部性。                                                         |
| **算术强度 (AI)** | `(fp32_ops + int32_ops) / (global_memory_read_bytes + write_bytes)` | 每访问 1 字节内存，能执行多少次计算。高 AI 的程序是计算密集型，低 AI 的是内存密集型。  |
| **实际内存带宽利用率** | `(Achieved Bandwidth) / (Peak Theoretical Bandwidth)`              | 衡量你的程序利用可用内存带宽的能力。低利用率可能由访存模式不佳（非合并访问）导致。     |
| **实际计算吞吐利用率** | `(Achieved GFLOPS/s) / (Peak Theoretical GFLOPS/s)`                | 衡量你的程序利用 PE 计算单元的能力。低利用率可能由内存瓶颈、指令依赖、分支分化等导致。 |


### 4.5 层次五：静态与配置数据 (Static & Configuration Data)

这些数据在程序运行期间通常不变，但为解释性能数据提供了必要的上下文。没有这些信息，性能数据就是无源之水。

| 数据点 (Data Point)      | 数据类型 | 来源             | 说明                                                                             |
| ------------------------ | -------- | ---------------- | -------------------------------------------------------------------------------- |
| **设备属性 (Device Props)** | struct   | 驱动 API         | 芯片名称、PE 数量、核心时钟频率、内存时钟频率、L1/L2 缓存大小、理论峰值性能等。    |
| **Kernel 属性** | struct   | 编译器/Runtime   | Kernel 名称、静态分配的共享内存大小、每个线程/Work-item 使用的寄存器数量。       |
| **程序与环境信息** | string   | `MyPEPTI` 库     | 进程 ID (PID)、使用的设备 ID、Runtime 库版本、驱动版本。                         |
| **启动配置** | struct   | Runtime API Hook | `Grid` 和 `Block` (或等价概念) 的维度配置。                                      |
