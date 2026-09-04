
### 多进程协作

| 进程名称 | 功能描述 |
| :--- | :--- |
| **主进程** | 统筹调度，读取循环内存池图像帧，左右半场分发 |
| **广角相机采集进程** | 按照 30FPS（或60FPS）采集广角相机图像 3840x1080 分辨率，拆分为左右半场图像 1920x1080 分辨率，存入循环内存池（队列长度10） ，记录索引地址|
| **广角相机录制进程** | 根据主进程分发的左右半场图像，分别录制成本地分段视频（或实时推流到云端RTMP服务器） |
| **双路长焦相机录制进程** | 左右半场两个长焦相机输出H264图像信息分别录制成本地分段视频（或实时推流到云端RTMP服务器） |
| **文件上传进程** | 上传玩家子图和JSON文件到腾讯云对象存储（COS），返回文件 URL |
| **MQTT 通信进程** | 接收连接/开始/停止指令，上传片段（或回合）数据 JSON |
| **模型推理子进程** | 各模型独立子进程，并行执行推理任务 |

*![多进程架构](/assets/eagle-eye-ai-process.png)*

### 核心执行流程

1. **启动阶段**：MQTT 收到连接请求 → 检查运行环境初始化完成并返回响应 → MQTT 收到开始请求 → 广角相机采集进程启动 → 图像帧写入循环内存池
2. **预处理阶段**：主进程读取循环内存池图像帧，左右半场 → 分别转发给广角录制/推流进程 → 同时启动双路长焦推流
3. **热身阶段**：前 120 帧快速检测球场关键点，完成变换矩阵求解（预计算）
4. **主循环阶段**：
   - 球员检测（yolo11n-det）与正反手识别（yolo11n-cls）**并行执行**
   - 网球轨迹追踪（TrackNetV2）**并行执行**
5. **事件检测**：每 60 帧调用一次 TCN 关键帧事件检测模型（30 帧历史 + 60 帧当前 = 90 帧）
6. **片段上传**：收集到完整片段 → 文件上传进程 → 腾讯云 COS → 拼接 JSON → MQTT 上传
7. **停止阶段**：MQTT 收到停止请求 → 停止采集 → 重新初始化，等待下次连接


```text
MQTT 收到连接请求
       │
       ▼
检查运行环境初始化完成
       │
       ▼
返回连接响应
       │
       ▼
MQTT 收到开始请求
       │
       ▼
广角相机采集进程启动
       │
       ▼
采集 3840x1080 图像 (30/60 FPS)
       │
       ▼
拆分为左右半场 (1920x1080)
       │
       ▼
存入循环内存池 (队列长度10，记录索引地址)
       │
       ▼
主进程读取循环内存池，取出图像帧
       │
       ▼
拆分左右半场
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
广角相机录制进程    双路长焦录制进程      前120帧热身阶段
       │                  │                  │
录制全景分段视频    录制左右半场分段视频    球场关键点检测
或 RTMP 推流       或 RTMP 推流          (YOLO11n-Pose)
       │                  │                  │
       │                  │                  ▼
       │                  │           求解变换矩阵预计算
       │                  │                  │
       └──────────────────┴──────────────────┘
                              │
                              ▼
                       主循环阶段并行执行
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    球员检测            正反手识别          网球轨迹追踪
  (YOLO11n-Det)    (YOLO11n-Cls)      (TrackNetV2优化版)
                    3帧灰度堆叠
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
                        每60帧触发
                              │
                              ▼
                    TCN 事件检测 (历史30帧+当前60帧=90帧)
                              │
                              ▼
                 输出事件标签: none/serve/bounce/stroke/net_fault
                              │
                              ▼
                       收集完整片段
                              │
                              ▼
                   文件上传进程 → 腾讯云COS
                              │
                              ▼
                    返回文件URL，拼接片段数据JSON
                              │
                              ▼
                       MQTT上传片段数据


MQTT 收到停止请求
       │
       ▼
停止相机采集
       │
       ▼
重新初始化，等待下次连接
```
---

## 🤖 模型详解

### 1. 球场关键点检测（YOLO11n-Pose）

| 属性 | 说明 |
| :--- | :--- |
| **任务** | 左右球场各 12 个关键点检测 |
| **模型** | YOLO11n-Pose（轻量化姿态估计） |
| **用途** | 计算球场变换矩阵，建立“图像坐标 → 物理坐标”的映射关系 |

**技术方案：**
- 每帧检测左右半场各 12 个关键点（共 24 点）
- 将当前帧检测点与参考球场关键点进行**变换矩阵求解**（透视变换）
- 仅需**前 120 帧**完成一次求解，后续帧复用矩阵（除非场景大幅变化）

**参考球场：**
*![球场关键点](/assets/rl/court_reference_text.png)* | *![球场关键点坐标](/assets/rl/left_right_court_reference.png)*

**参考关键点选择：**
***左半场（12种方式）：***
*![左参考点1](/assets/rl/left_court_conf/left_court_conf_1.jpg)* | *![左参考点2](/assets/rl/left_court_conf/left_court_conf_2.jpg)* 
*![左参考点3](/assets/rl/left_court_conf/left_court_conf_3.jpg)* | *![左参考点4](/assets/rl/left_court_conf/left_court_conf_4.jpg)* 
*![左参考点5](/assets/rl/left_court_conf/left_court_conf_5.jpg)* | *![左参考点6](/assets/rl/left_court_conf/left_court_conf_6.jpg)* 
*![左参考点7](/assets/rl/left_court_conf/left_court_conf_7.jpg)* | *![左参考点8](/assets/rl/left_court_conf/left_court_conf_8.jpg)* 
*![左参考点9](/assets/rl/left_court_conf/left_court_conf_9.jpg)* | *![左参考点10](/assets/rl/left_court_conf/left_court_conf_10.jpg)* 
*![左参考点11](/assets/rl/left_court_conf/left_court_conf_11.jpg)* | *![左参考点12](/assets/rl/left_court_conf/left_court_conf_12.jpg)* 
***右半场（12种方式）：***
*![右参考点1](/assets/rl/right_court_conf/right_court_conf_1.jpg)* | *![右参考点2](/assets/rl/right_court_conf/right_court_conf_2.jpg)* 
*![右参考点3](/assets/rl/right_court_conf/right_court_conf_3.jpg)* | *![右参考点4](/assets/rl/right_court_conf/right_court_conf_4.jpg)* 
*![右参考点5](/assets/rl/right_court_conf/right_court_conf_5.jpg)* | *![右参考点6](/assets/rl/right_court_conf/right_court_conf_6.jpg)* 
*![右参考点7](/assets/rl/right_court_conf/right_court_conf_7.jpg)* | *![右参考点8](/assets/rl/right_court_conf/right_court_conf_8.jpg)* 
*![右参考点9](/assets/rl/right_court_conf/right_court_conf_9.jpg)* | *![右参考点10](/assets/rl/right_court_conf/right_court_conf_10.jpg)* 
*![右参考点11](/assets/rl/right_court_conf/right_court_conf_11.jpg)* | *![右参考点12](/assets/rl/right_court_conf/right_court_conf_12.jpg)* 

**检测结果示例：**
<!-- TODO: 添加关键点检测结果图 -->
*![左球场关键点检测](/assets/rl/left_00005.jpg)* | *![右球场关键点检测](/assets/rl/right_00005.jpg)* 
*左右球场各 12 个关键点检测结果示意*

---

### 2. 网球轨迹追踪（LightTrackNet）

| 属性 | 说明 |
| :--- | :--- |
| **任务** | 网球在高帧率视频中的轨迹追踪 |
| **模型** | TrackNetV2（轻量化优化版） |
| **部署优化** | 网络结构轻量化 + TensorRT FP16/INT8 量化 |

**技术方案：**
- 基于 TrackNetV2 进行**网络结构轻量化**
- 导出 ONNX 格式，使用 TensorRT 进行 **FP16 / INT8 量化**
- 部署至 Jetson 平台，实现**高帧率、低延迟**的实时轨迹推理

**检测结果示例：**
<!-- TODO: 添加轨迹追踪效果图 -->
![网球轨迹追踪](/assets/tennis-track.png)
*网球轨迹追踪与落点统计示意*

---

### 3. TCN 关键帧事件检测

| 属性 | 说明 |
| :--- | :--- |
| **任务** | 识别比赛中的关键时刻 |
| **类别** | `['none', 'serve', 'bounce', 'stroke', 'net_fault']` |
| **模型** | TCN（时间卷积网络） |
| **输入** | 90 帧连续图像（历史 30 帧 + 当前 60 帧） |
| **触发频率** | 每 60 帧检测一次 |

**技术方案：**
- 相比 RNN/LSTM，TCN 具有**并行计算**和**更长的有效记忆**能力
- 拼接 **历史 30 帧 + 当前 60 帧** 共 90 帧作为输入
- 输出 5 分类事件标签，精度与实时性优于传统方案

**检测结果示例：**
<!-- TODO: 添加事件检测结果图 -->
![左半场TCN 事件检测](/assets/tcn_res_left.png) | ![右半场TCN 事件检测](/assets/tcn_res_right.png)
*关键事件（发球、击球、出界等）检测结果示意*

---

### 4. 球员检测（YOLO11n-Det）

| 属性 | 说明 |
| :--- | :--- |
| **任务** | 区分场内球员与场外其他人员 |
| **模型** | YOLO11n-Det（轻量化目标检测） |
| **输出** | 球员边界框 + 类别标签 |

**技术方案：**
- 训练 YOLO11n-Det 模型，识别“场上球员”与“场外人员”
- 结合球场关键点进行空间过滤，仅保留场上球员

**检测结果示例：**
<!-- TODO: 添加球员检测结果图 -->
![左半场球员检测](/assets/player_det_left.jpg) | ![右半场球员检测](/assets/player_det_right.jpg)
*球员检测及身份区分示意*

---

### 5. 正反手识别（YOLO11n-Cls）

| 属性 | 说明 |
| :--- | :--- |
| **任务** | 识别球员击球时使用的是正手还是反手 |
| **模型** | YOLO11n-Cls（轻量化分类模型） |
| **输入** | 连续 3 帧灰度图堆叠（3 × 1 通道 → 3 通道） |
| **部署优化** | 复用 YOLO11n 推理框架，减少额外开销 |

**技术创新 —— 输入通道压缩：**
- 常规方案：连续 3 帧 RGB 图 = **9 通道**输入
- 本方案：连续 3 帧灰度图堆叠 = **3 通道**输入
- **效果**：在保持识别精度的同时，**大幅降低显存占用与计算开销**

**检测结果示例：**
<!-- TODO: 添加正反手识别结果图 -->
![左半场正反手识别](/assets/shot_cls_left.jpg) | ![右半场正反手识别](/assets/shot_cls_right.jpg)
*正手 / 反手识别结果示意*

---

## 📊 项目成果

| 指标 | 成果 |
| :--- | :--- |
| **球员检测精度** | 高精度，有效区分场上球员与场外人员 |
| **轨迹追踪延迟** | 边缘端低延迟推理，满足实时分析需求 |
| **事件检测** | TCN 模型精准捕捉发球、击球、出界等关键事件 |
| **系统架构** | 三期迭代，从云端实时流→边缘端MQTT按需触发→片段化击球数据上报，降低带宽与云端算力成本 |
| **模型轻量化** | TensorRT FP16/INT8 量化 + 通道压缩，适配 Jetson 平台 |

---

## 🔧 技术栈

| 类别 | 技术 |
| :--- | :--- |
| **编程语言** | Python |
| **深度学习框架** | PyTorch |
| **模型优化** | TensorRT、ONNX |
| **目标检测** | YOLO11n（Pose / Det / Cls） |
| **轨迹追踪** | TrackNetV2（轻量化版） |
| **时序模型** | TCN（时间卷积网络） |
| **边缘硬件** | NVIDIA Jetson（nano / NX） |
| **通信协议** | MQTT |
| **云存储** | 腾讯云 COS |

---

## 📝 总结

本项目实现了一个**完整的边缘端网球智能分析系统**，涵盖多模型协同推理、实时数据流处理、云边协同上传等多个技术挑战。核心创新包括：

1. **TCN 事件检测**：替代传统的 RNN/LSTM 方案，在实时性和精度上取得更好平衡
2. **输入通道压缩**：创新的 9→3 通道压缩策略，显著降低边缘端计算负担
3. **多进程架构**：采集、推理、录制、上传四层解耦，确保系统实时性与可扩展性
4. **三期架构演进**：从云端到边缘端的系统性优化，体现架构思维

---

## 📎 参考资料

- TrackNetV2 论文及开源实现
- YOLO11 官方文档
- NVIDIA Jetson 部署文档

---

*最后更新：2026-09-04*