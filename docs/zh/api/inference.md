---
outline: deep
---

# 统一推理服务

对应 [RFC #71](https://github.com/redai-studio/Relax/issues/71)。Rollout、GenRM、Teacher 保留各自业务入口，共用 `InferenceGateway`、`InferenceManager` 状态机和 `SGLangEngine`。Gateway 使用 CPU，不占用引擎的 GPU placement group。只有 Rollout 允许动态权重更新和 DCS 注册。

## API 与路由

每个角色分别部署在 `/rollout`、`/genrm`、`/teacher`，提供：

| 路径 | 功能 |
| --- | --- |
| `GET /engines` | 模型、逻辑引擎、状态、router URL 和 `topology_revision` |
| `GET /v1/models` | 可配置的模型 ID |
| `GET /health` | 控制面健康状态与模型状态 |
| `POST /generate` | SGLang 原始请求，保留 logprob/base64 等字段 |
| `POST /v1/chat/completions` | OpenAI chat 请求与 SSE 流式响应 |
| `POST /chat/completions` | chat 兼容别名 |

模型选择顺序为 `model`、`route_key`、默认模型。未知显式模型或 route key 返回 400；没有默认模型时必须指定模型。Gateway 和 direct client 使用同一套选择函数。PD 模型通过 router 请求，禁止直连 prefill/decode worker；普通多节点引擎只发布 head HTTP 端点，不发布 TP/PP follower。

`sleeping`、`draining`、`onloading`、`failed`、`dead` 或尚未完成权重同步的模型不接收新请求，Gateway 返回 503 和 `Retry-After`。请求不会隐式激活模型。`topology_revision` 会随状态和端点变化更新，已返回的 snapshot 不会被后续修改。

旧 GenRM `messages + sampling_params + route_key` 请求和 `{"response": ...}` 响应保留；旧 Teacher URL、Manager onload/offload 和 GenRMEngine 名称保留。启用新 placement/discovery 的 Teacher 恢复后允许更换端点；旧的原始 URL 模式保留恢复约束。

```python
from relax.utils.inference_client import InferenceClient

async with InferenceClient(service_url, direct=True) as client:
    result = await client.generate(
        {"input_ids": [1, 2, 3], "return_logprob": True}, model="default"
    )
    async for chunk in client.stream(
        {"messages": [{"role": "user", "content": "Hello"}]},
        path="v1/chat/completions", model="default",
    ):
        consume_sse_bytes(chunk)
```

`direct=False` 通过 Gateway 转发。direct 模式使用原始 SGLang payload；旧 GenRM 消息模板转换继续由 GenRM Gateway 完成。Client 每次请求读取 discovery，不在传输失败后自动重放生成请求。

## Placement 与阶段切换

- **Decoupled**：角色使用独立 GPU 池；Teacher 专用 PG 由 Manager 创建和回收。
- **Split**：`--colocate` 下，Rollout、非 deferred GenRM、非 deferred Teacher 按顺序占用 Actor 池内互不重叠的区间，训练阶段复用该池。
- **Defer**：`--opd-teacher-defer` 和 `--defer-reward-to-post-process` 分别启用 Teacher、GenRM 延后处理。延后角色从共享池起点分配，但分属不同执行阶段；同一角色的多个模型仍需互不重叠。

共享 Actor 池要求训练和推理双方 offload。规划器在创建引擎前检查总量、replica 完整性、TP×PP、节点宽度、局部跨度、配置覆盖和阶段重叠。decoupled 或 hybrid 的显式 Rollout GPU 数量超过独立资源预算时，在 Teacher 或 PG 创建前拒绝。PG 就绪后再检查实际节点与物理 GPU 连续性。首次 deferred 引擎按需启动，避免启动时共同驻留。

defer 流程为：完成生成并排空未保留的请求 → Rollout offload → Teacher onload、完整写回、offload → 必要的学生 top-k 补算 → GenRM onload、奖励计算、offload → 发布训练数据。按原 Sample 对象写回，不依赖 sample index 唯一。Teacher 失败或字段长度不完整时不发布数据。group RM 保留 prompt 分组；评测奖励不做训练用的归一化。deferred 生成和评测互斥。

生命周期支持重复调用和分批恢复 `weights`、`kv_cache`、`cuda_graph`，全部恢复前不发布 READY。失败时尝试清理所有候选；清理未确认时保留引擎句柄、PG ownership 和阶段锁，discovery 保持不可用，并阻止后续激活。Engine 在子进程启动后立即记录句柄，初始化中途失败也能重试清理。Manager 重试清理成功不会自动恢复失败 batch 或释放 Coordinator 保留的阶段锁。Manager 不删除借用的 PG。

一期拒绝同卡同阶段共同驻留。defer 只支持同步 colocate、框架 SGLang 完整 batch 流程；不支持 fully async、partial rollout、agentic/custom rollout，以及依赖尚未生成奖励的 dynamic sampling filter。

## 验证

CPU 回归覆盖路由、SSE、discovery、状态幂等、静态权重保护、失败清理、PG ownership、原有 Teacher/GenRM 接口和 deferred 数据发布。部分 engine 测试直接加载源文件中的纯 HTTP 方法，避免导入 GPU 库；它们不等同于实际 GPU 执行。

```bash
uv run --no-sync python -m pytest -q tests/inference
```

双节点冒烟测试需要在兼容的 SGLang 环境内显式启用，并提供同构 GPU 节点宽度及支持对应 TP 的共享 HF checkpoint。TP 等于节点 GPU 数的两倍，`MODEL_DIR` 在此测试中指向一个具体 checkpoint：

```bash
INFERENCE_RUN_GPU_TESTS=1 uv run --no-sync python -m pytest -q \
  tests/integration/test_unified_inference_gpu.py
```

必需环境变量：`RAY_ADDRESS`、`MODEL_DIR`、`INFERENCE_GPUS_PER_NODE`。未显式启用时测试跳过。

2026-09-17 在 Isambard 两个 GH200 节点、每节点一张 GPU 上，使用 Qwen3-0.6B、TP=2 完成测试：**5 项通过、0 项跳过，用时 630.11 秒**。运行环境为 SGLang 0.5.17、Ray 2.58.0、PyTorch 2.11.0+cu129，使用 Triton attention 和 decode CUDA graph。所有测试重试复用同一个三小时作业 `6631103`，最终验证为 step `6631103.11`。

三个角色测试检查 offload 前后生成一致、重复生命周期调用、分批恢复、仅发布逻辑 head、静态模型拒绝 DCS 注册，以及借用 PG 保留。第四个测试注入 offload 和 shutdown RPC 失败，确认真实 GPU 模型仍能生成时，discovery 已关闭、句柄和阶段锁仍保留、后续激活被拒绝；恢复 RPC 后清理重试成功，借用 PG 保持存在。

第五个测试在同一组双 GPU 上依次执行真实 Rollout 生成、经 CPU Gateway 的 Teacher 打分和 GenRM 推理；检查原 Sample 的 Teacher 字段写回、重复 sample index、与相同学生 checkpoint 的分数一致性，以及所有角色进入 sleeping 后才转为训练 TensorDict 并发布。该测试的最终发布端使用记录数据的测试 sink；真实 Megatron 消费由下述完整训练验证。

相关本地回归 **175 项通过、29 项跳过**；跳过原因是 GPU 未显式启用或缺少训练依赖。另在 SGLang 环境内运行的 **145 项**契约测试全部通过、无跳过，覆盖原有 GenRM/Teacher 行为、清理失败重试、PG ownership，以及资源创建前拒绝非法布局。全仓库 pre-commit 检查通过。18 个选定实现与回归测试文件的本地和远端 SHA-256 完全一致。

2026-09-18 使用另一项三小时双节点分配 `6654867`（每节点一张 GH200），新增 **3 项 TP=1 / PP=2 的角色生命周期测试通过**，用时 261.63 秒。`tests/integration/test_unified_inference_topology_gpu.py` 进一步完整构造生产 `RolloutManager`：**PP 测试通过**（158.58 秒），**PD 测试通过**（179.97 秒），均无跳过。覆盖真实 router 注册、Gateway/direct 生成、offload 和部分恢复时不可用、重复生命周期调用、恢复后 logprob 一致，以及借用 PG 保留。PP 只发布一个逻辑 head；PD 的 prefill/decode 端点不可直连，必须经过 router。

PD 将 prefill 和 decode 分别放在两个节点，通过 Mooncake TCP 传输 KV（`INFERENCE_PD_BACKEND=mooncake_tcp`、`MC_FORCE_TCP=1`）。拓扑测试额外要求共享目录 `INFERENCE_SHARED_TEST_DIR`，以及恰好两个每节点一张 GPU 的节点：

```bash
INFERENCE_RUN_GPU_TESTS=1 INFERENCE_PD_BACKEND=mooncake_tcp MC_FORCE_TCP=1 \
  uv run --no-sync python -m pytest -q tests/integration/test_unified_inference_topology_gpu.py
```

剩余三角色训练矩阵使用 `scripts/training/hpc/run-unified-inference-3role.sh`：
`INFERENCE_LAYOUT=decoupled` 验证独立角色资源池，`INFERENCE_LAYOUT=split`
验证同一 actor placement group 内不重叠的 Rollout/Teacher/GenRM 分片。split
默认使用 torch_memory_saver；只有后端不支持 TMS 时才设置
`INFERENCE_SELECTIVE_OFFLOAD=1`。

补齐完整训练环境后，再在同一 pytest 进程中连续执行 PP、PD：**2 项通过、0 项跳过，281.28 秒**，step `6654867.18`。

同一分配内，完整 Megatron 训练作业 `raysubmit_tg4D86NcgNJpSNJF`（step `6654867.17`）成功结束。配置为 Qwen3-0.6B、训练 TP=1 / PP=1 / DP=2、推理 TP=2、selective offload、Teacher defer 和 GenRM defer。生产 Controller 完成两轮生成、Teacher 回写、GenRM 奖励、训练及 checkpoint 保存，每轮 4 条样本、1 次 optimizer step。两轮 loss 为 0.236228 和 0.487990，梯度范数为 19.0095 和 26.6086；8 条训练样本的 Teacher 概率完整、有限且与 response 对齐，GenRM 实际请求 8 次，rollout 权重版本为 1→2。比较原始 HF 权重与两次保存结果，抽查的 Q projection 参数在两次更新后均发生变化。

训练脚本通过仓库入口提交，需要独立的双节点 Ray 集群、Megatron-Bridge、Transformer Engine、Apex，以及共享输出目录。`MEGATRON` 应指向已配置的 Megatron 源码目录：

```bash
export RAY_ADDRESS=ray-head:6379
export RAY_DASHBOARD=http://ray-head:8265
export RAY_DASHBOARD_ADDRESS="$RAY_DASHBOARD"
export RAY_API_SERVER_ADDRESS="$RAY_DASHBOARD"
MODEL_DIR=/shared/Qwen3-0.6B \
INFERENCE_TRAIN_OUTPUT=/shared/new-validation-output \
RAY_NO_WAIT=1 bash scripts/entrypoint/ray-job.sh \
  scripts/training/hpc/run-unified-inference-2gpu.sh
```

此脚本验证完整训练流程，不评估模型质量。实测训练使用 TE 2.14.1、仓库指定的 Megatron-Bridge/MCore 版本和补丁。split 验证使用默认 TMS 路径；对于需要显式切换的后端，selective offload 仍作为 fallback。

原有角色测试使用测试 Pool；拓扑测试完整构造生产 RolloutManager、router 与真实引擎；三角色验证使用完整 Controller，合计覆盖 RFC 明文验收项。

## 下一步

- [Rollout API](./rollout.md)
- [GenRM API](./genrm.md)
