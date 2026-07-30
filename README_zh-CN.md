# OpenD4RT WorldTrack 推理与分析

[English](README.md) | 简体中文

这个 fork 保留了原始 OpenD4RT 代码库，并新增了一组小型、可公开的 WorldTrack
推理、评测、可视化和分析工具。目标是让推理行为更容易检查：在 WorldTrack 上运行
OpenD4RT，比较 32 帧片段和 64 帧 anchor-clip 序列，并可视化 D4RT `t_cam`
查询参数的坐标系含义。

这不是对 D4RT 论文的完整复现。我没有训练模型，没有修改发布的 checkpoint，也没有
运行大规模超参数搜索。

## 上游项目

OpenD4RT 提供从视频进行稠密 4D 重建与跟踪的代码。这里使用的发布 checkpoint 是
`OpenD4RT_32CLIP_9Dataset_NoAUG`，其模型片段长度为 32 frames。

关于方法、训练设置和完整 benchmark 背景，请参考原项目和论文：

- 原始仓库： https://github.com/Lijiaxin0111/Open-d4rt
- 项目主页： https://d4rt-paper.github.io/
- 论文： https://arxiv.org/abs/2504.13152

## 我在这个 fork 中新增的内容

上游 OpenD4RT 提供模型架构、发布的 checkpoint 和核心查询接口。本 fork 新增：

- FP16 推理设置和按显存控制的查询分块。
- 用于 APD/EPE 风格 3D 跟踪指标的 WorldTrack 评测入口。
- WorldTrack 演示构建器，可导出输入视频、2D 叠加可视化、3D 轨迹、元数据和运行摘要。
- 轻量的公开结果构建脚本：
  `scripts/build_public_artifacts.py`。
- `t_cam` 查询语义分析：对同一批查询点比较 `t_cam=t_tgt` 和 `t_cam=0`。
- 在同一个 `juggle_5` 序列上做 32 帧 vs 64 帧推理案例分析。

64 帧设置使用同一个 32-clip checkpoint 和 anchor-clip inference。它没有把模型
上下文从 32 frames 增加到 64 frames。

![OpenD4RT WorldTrack 演示概览](artifacts/opend4rt_demo_overview.png)

所选 WorldTrack 演示的概览：输入帧、2D GT/pred 叠加可视化、3D GT 轨迹和 3D 预测轨迹。

![D4RT t_cam 查询语义](artifacts/t_cam_query_semantics.png)

`t_cam` 的坐标系语义：同一批查询点分别在当前相机坐标系和固定参考坐标系中可视化。这不是精度对比。

![32-frame vs 64-frame 推理对比](artifacts/frame_count_comparison.png)

一个轻量案例分析：使用同一个发布的 32-clip checkpoint，比较 32 帧推理和 64 帧
anchor-clip inference。

## 公开结果文件

本仓库公开包含以下结果文件：

| 文件 | 说明 |
| --- | --- |
| `artifacts/opend4rt_demo_overview.png` | 输入帧、2D GT/pred 叠加可视化、3D GT 轨迹和 3D 预测轨迹。 |
| `artifacts/t_cam_query_semantics.png` | 同一批查询点在当前相机坐标系和固定参考坐标系下的结果。 |
| `artifacts/frame_count_comparison.png` | 32 vs 64 frames 的 APD、EPE、dynamic APD/EPE 和运行时间对比。 |
| `artifacts/worldtrack_evaluation_summary.json` | 完整 WorldTrack mini 评测摘要，使用 strict JSON `null`，不使用 bare `NaN`。 |
| `artifacts/t_cam_query_semantics.json` | 查询点、可见性、运动分数、运行时间和预测轨迹。 |
| `artifacts/frame_count_comparison.json` | 两次 `juggle_5` 运行的可解析质量/运行时间对比。 |

未公开：数据集、checkpoint 权重、虚拟环境、临时日志和完整演示视频包。

## 结果摘要

使用发布的 32-clip checkpoint 的 WorldTrack mini 评测：

| Subset | APD global | EPE global | Dynamic APD | Dynamic EPE | Queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.6992 | 0.2965 | 0.6975 | 0.3629 | 22,187 |
| `po_mini` | 0.6600 | 0.3405 | 0.7329 | 0.2734 | 53,468 |
| `pstudio_mini` | 0.7861 | 0.1813 | 0.7861 | 0.1813 | 8,720 |
| `ds_mini` | 0.7266 | 0.2945 | 0.7519 | 0.2701 | 52,462 |

同一个 `pstudio_mini/juggle_5.npz` sample：

| 情况 | APD | EPE | Dynamic APD | Dynamic EPE | 有效查询数 | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 frames | 0.9916 | 0.0533 | 0.9916 | 0.0533 | 256 | 65.1s |
| 64 frames, anchor-clip | 0.9554 | 0.0693 | 0.9554 | 0.0693 | 256 | 148.3s |

在这个所选序列上，通过 anchor-clip inference 扩展时间范围增加了运行时间，并产生了
略低的跟踪质量。这个单序列比较不应被解读为一般性的 benchmark 结论。

## 环境配置

按照上游说明安装项目依赖，然后将发布的 checkpoint 文件放在：

```text
checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml
checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
```

将 WorldTrack 数据放在：

```text
data/worldtrack_release/
```

本仓库只跟踪轻量配置文件。Checkpoint 权重和数据集文件需要单独提供。

## 运行评测

完整 mini 评测示例：

```bash
EXP=worldtrack_mini \
PRECISION=fp16 \
NUM_FRAMES=64 \
QUERY_CHUNK_SIZE=512 \
MAX_GPU_MEMORY_GIB=20 \
bash run_eval_worldtrack.sh
```

评测会在配置的输出目录下写入 summary JSON。

## 生成 demo 包

生成 64 帧演示包：

```bash
DEMO_CASE=pstudio_mini/juggle_5.npz \
OUTPUT_DIR=tmp/worldtrack_demo_64f \
PRECISION=fp16 \
NUM_FRAMES=64 \
QUERY_CHUNK_SIZE=512 \
MAX_GPU_MEMORY_GIB=20 \
bash run_build_worldtrack_demo.sh
```

生成 32 帧对比演示包：

```bash
DEMO_CASE=pstudio_mini/juggle_5.npz \
OUTPUT_DIR=tmp/worldtrack_demo_32f \
PRECISION=fp16 \
NUM_FRAMES=32 \
QUERY_CHUNK_SIZE=512 \
MAX_GPU_MEMORY_GIB=20 \
bash run_build_worldtrack_demo.sh
```

## 生成公开结果文件

两个演示包都存在后，重新生成小型 PNG/JSON 文件：

```bash
python scripts/build_public_artifacts.py \
  --summary-json artifacts/worldtrack_evaluation_summary.json \
  --demo-64-dir tmp/worldtrack_demo_64f \
  --demo-32-dir tmp/worldtrack_demo_32f \
  --device cuda \
  --precision fp16 \
  --max-gpu-memory-gib 20 \
  --num-frames 64 \
  --query-chunk-size 512
```

## 范围与限制

- 本仓库聚焦于推理工程、评测和分析，使用发布的
  `OpenD4RT_32CLIP_9Dataset_NoAUG` checkpoint；没有进行模型训练或 fine-tuning。
- 32/64 帧对比是基于一个 WorldTrack 序列的轻量案例分析。两种设置都使用同一个
  32-clip checkpoint，其中 64 帧序列通过 anchor-clip inference 处理。

## 许可证

这个 fork 保留上游 Apache-2.0 license。使用该方法或发布模型时，请引用原始 D4RT
paper/project。
