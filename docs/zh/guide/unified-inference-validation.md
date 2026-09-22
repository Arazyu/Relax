---
outline: deep
---

# 统一推理服务验收记录

## 验收结论

| RFC 要求 | 结论 | 证据 |
| --- | --- | --- |
| Rollout、GenRM、Teacher 共用 Gateway 和 Manager | **通过** | 共用 `InferenceGateway`、`InferenceManager`、生命周期、路由和 client 契约 |
| 静态模型不参与动态权重更新 | **通过** | 静态 DCS/update guard 与 endpoint 测试；GPU 训练只更新 Rollout |
| Gateway/direct 路由一致，discovery 不暴露 TP/PP worker | **通过** | PP/PD 拓扑检查和 head-only discovery 断言 |
| 非法布局在启动前拒绝 | **通过** | capacity、节点跨度、重叠、defer 和 co-residency 测试 |
| defer 无资源冲突，Teacher 结果在训练前回写 | **通过** | deferred 发布测试，以及四节点 split/decoupled 训练结果 |
| 生命周期幂等、失败回滚、PG ownership 和旧接口兼容 | **通过** | lifecycle、cleanup、借用 PG、兼容性和恢复契约测试 |
| 多节点 GPU 验证完成 | **通过** | `6654867` PP/PD 与 Megatron、`6772050` split、`6780723` decoupled |

## 测试结论

推理契约套件结果为 **91 passed、1 skipped**。唯一 skip 是当前工作站未安装 Ray 时的 Ray 专用端口分配测试。Python 编译、启动脚本语法和 diff 检查均通过。

## GPU 结论

必需的多节点 GPU 场景均通过：

- PP/PD 路由与 discovery 通过。
- split 训练通过，Teacher 在 GenRM 前完成回写，Megatron step 完成。
- decoupled 训练通过，训练前已有 Teacher 结果，Rollout 动态更新完成。
- 完整 Megatron 数据通路和 checkpoint 发布通过。

实现和验证范围严格对应 RFC #71 明文验收项。

## 复现命令、GPU 与环境

| 项目 | 配置 |
| --- | --- |
| 模型 | Qwen3-0.6B |
| GPU 平台 | GH200，每节点 1 张 GPU |
| PP/PD 与 Megatron 作业 | `6654867`，节点 `nid010294`、`nid010319` |
| split 作业 | `6772050`，节点 `nid010235`、`nid010244`、`nid010252`、`nid010256` |
| decoupled 作业 | `6780723`，节点 `nid010231`、`nid010240`、`nid010254`、`nid010255` |
| 运行环境 | SGLang 0.5.17、Ray 2.58.0、PyTorch 2.11.0+cu129、Transformer Engine 2.14.1 |

```bash
# CPU 契约测试
python -m pytest -q tests/inference

# 双节点 PP/PD 拓扑
INFERENCE_RUN_GPU_TESTS=1 \
INFERENCE_PD_BACKEND=mooncake_tcp MC_FORCE_TCP=1 \
python -m pytest -q tests/integration/test_unified_inference_topology_gpu.py

# 四节点三角色训练
INFERENCE_LAYOUT=decoupled \
MODEL_DIR=/shared/Qwen3-0.6B \
INFERENCE_TRAIN_OUTPUT=/shared/unified-decoupled \
bash scripts/training/hpc/run-unified-inference-3role.sh

INFERENCE_LAYOUT=split \
MODEL_DIR=/shared/Qwen3-0.6B \
INFERENCE_TRAIN_OUTPUT=/shared/unified-split \
bash scripts/training/hpc/run-unified-inference-3role.sh
```
