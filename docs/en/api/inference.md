---
outline: deep
---

# Unified inference service

[RFC #71](https://github.com/redai-studio/Relax/issues/71) shares `InferenceGateway`, the `InferenceManager` state machine, and `SGLangEngine` across Rollout, GenRM, and Teacher while preserving their workloads. Each role has a CPU gateway outside its GPU placement group. Static GenRM and Teacher engines reject DCS registration and dynamic weight updates.

## API and routing

The role prefixes `/rollout`, `/genrm`, and `/teacher` expose `/engines`, `/health`, `/v1/models`, `/generate`, `/v1/chat/completions`, and the `/chat/completions` alias. Raw generation preserves logprob/base64 fields; chat supports SSE.

| Path | Purpose |
| --- | --- |
| `GET /engines` | Models, logical engines, states, router URL, and `topology_revision` |
| `GET /v1/models` | Configured model IDs |
| `GET /health` | Control-plane health and model states |
| `POST /generate` | Raw SGLang requests, including logprob/base64 fields |
| `POST /v1/chat/completions` | OpenAI chat requests and SSE |
| `POST /chat/completions` | Compatible chat alias |

Gateway and `relax.utils.inference_client.InferenceClient` share model selection: explicit `model`, then `route_key`, then a configured default. Unknown explicit selections return 400. Unavailable models return 503 with `Retry-After`; requests never wake a sleeping model. Discovery publishes immutable snapshots and a changing `topology_revision`. Only logical HTTP heads are exposed. PD workloads always use the router.

Models in `sleeping`, `draining`, `onloading`, `failed`, or `dead` states, or awaiting weight synchronization, reject new requests. When no default is configured, clients must select a model explicitly.

Use `InferenceClient(service_url, direct=True)` for raw SGLang requests to discovered endpoints, or `direct=False` for gateway requests. `generate()` returns JSON; `stream()` yields SSE bytes. The client refreshes discovery per request and does not replay generation after a transport failure.

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

Legacy GenRM message-template requests and `{"response": ...}` responses remain available through its gateway. Existing Teacher URLs, Manager lifecycle methods, and the `GenRMEngine` name remain supported. Planned Teacher deployments can publish replacement endpoints after recovery; legacy raw-URL deployments retain their original recovery restrictions.

## Placement and lifecycle

- **Decoupled:** separate role GPU pools; dedicated Teacher PGs belong to their manager.
- **Split:** synchronous colocate partitions the Actor pool between Rollout, inline GenRM, and inline Teacher. Training reuses this pool after inference offloads.
- **Defer:** `--opd-teacher-defer` and `--defer-reward-to-post-process` reuse the shared pool in separate Teacher and GenRM phases. Models within a phase still occupy disjoint slices.

The planner checks capacity, complete replicas, TP×PP, local node spans, overrides, and overlap before engine creation. An explicit decoupled or hybrid Rollout GPU request cannot exceed its independent resource budget; this is rejected before Teacher or PG creation. Actual PG node/device mappings are checked before actors launch. Shared Actor/inference pools require both training and inference offload. Deferred engines are created lazily to avoid startup co-residency.

Deferred batches finish generation and drain surplus requests, offload Rollout, run Teacher, optionally restore the original student weights for top-k queries, run GenRM, offload all overlapping models, and only then publish training samples. Teacher failures or incomplete fields prevent publication. Prompt groups are preserved for group rewards; evaluation does not apply training normalization. Deferred generation and evaluation are serialized.

Onload/offload/shutdown are idempotent. Partial restoration of `weights`, `kv_cache`, and `cuda_graph` remains unavailable until complete. Failed activation triggers cleanup. Unconfirmed cleanup keeps the engine handle, PG ownership, and phase lease; discovery stays unavailable and further activation is blocked. Engines record their child process immediately after spawning, so failed initialization can still retry cleanup. Successful manager cleanup alone does not resume a failed batch or release its retained coordinator lease. Managers never remove borrowed PGs.

Same-GPU co-residency within one phase is unsupported. Defer requires synchronous colocate and the framework's complete SGLang batch pipeline; fully async, partial rollout, agentic/custom rollout, and reward-dependent dynamic filtering are rejected.

## Validation

CPU contracts cover routing, SSE, discovery, idempotency, static weight protection, failure cleanup, PG ownership, legacy interfaces, and deferred publication. Some tests load real HTTP lifecycle methods from the source without importing GPU libraries; they do not establish GPU correctness.

```bash
uv run --no-sync python -m pytest -q tests/inference
```

The opt-in smoke test requires a compatible SGLang runtime, `RAY_ADDRESS`, `MODEL_DIR` pointing to one shared HF checkpoint, and `INFERENCE_GPUS_PER_NODE`. The checkpoint must support TP equal to twice the homogeneous node width. Without explicit opt-in, these tests skip.

```bash
INFERENCE_RUN_GPU_TESTS=1 uv run --no-sync python -m pytest -q \
  tests/integration/test_unified_inference_gpu.py
```

On 2026-09-17, all **5 GPU tests passed in 630.11 seconds**, with no skips, on two Isambard GH200 nodes, one GPU per node, using Qwen3-0.6B with TP=2. The runtime was SGLang 0.5.17, Ray 2.58.0, and PyTorch 2.11.0+cu129, with Triton attention and decode CUDA graphs. All attempts reused one three-hour allocation, job `6631103`; the final run used step `6631103.11`.

The three role tests verify generation before/after offload, repeated lifecycle calls, partial restoration, logical-head-only discovery, static DCS rejection, and borrowed PG preservation. A fourth test injects failed offload and shutdown RPCs while the actual GPU model remains able to generate. It verifies closed discovery, retained handles and phase lease, blocked activation, successful cleanup retry, and preservation of the borrowed PG.

The fifth test runs actual Rollout generation, Teacher scoring through its CPU Gateway, and GenRM inference in sequential phases on the same two GPUs. It verifies Teacher fields on the original samples, duplicate sample-index handling, scores against the identical student checkpoint, and conversion into training TensorDicts only after every role sleeps. Its final publication sink records data; the full training run below verifies actual Megatron consumption.

The related local suite passed **175 tests**, with **29 skips** for GPU opt-in or missing training dependencies. A separate **145-test** contract run passed inside the SGLang runtime without skips, covering legacy GenRM/Teacher, failed cleanup retries, PG ownership, and invalid layouts rejected before resource creation. Full-repository pre-commit checks passed. SHA-256 checks matched all 18 selected implementation and regression-test files between the local workspace and remote run.

On 2026-09-18, an additional two-node allocation (`6654867`, one GH200 per node) passed **3 role lifecycle tests with TP=1 / PP=2** in 261.63 seconds. Full production `RolloutManager` construction also passed separate **PP** (158.58 seconds) and **PD** (179.97 seconds) tests in `tests/integration/test_unified_inference_topology_gpu.py`, with no skips. These tests cover actual router registration, Gateway/direct generation, unavailable responses during offload and partial restoration, repeated lifecycle calls, restored logprobs, and borrowed PG preservation. PP publishes one logical head; PD publishes prefill/decode endpoints as non-direct and requires its router.

The PD run places one prefill worker and one decode worker on different nodes and transfers KV through Mooncake TCP (`INFERENCE_PD_BACKEND=mooncake_tcp`, `MC_FORCE_TCP=1`). The topology suite additionally requires a shared `INFERENCE_SHARED_TEST_DIR` and exactly two nodes with one GPU each:

```bash
INFERENCE_RUN_GPU_TESTS=1 INFERENCE_PD_BACKEND=mooncake_tcp MC_FORCE_TCP=1 \
  uv run --no-sync python -m pytest -q tests/integration/test_unified_inference_topology_gpu.py
```

For the remaining three-role training matrix, use
`scripts/training/hpc/run-unified-inference-3role.sh`. Set
`INFERENCE_LAYOUT=decoupled` for independent role placement groups or
`INFERENCE_LAYOUT=split` for disjoint Rollout/Teacher/GenRM slices in one
actor placement group. Split defaults to torch_memory_saver; set
`INFERENCE_SELECTIVE_OFFLOAD=1` only when the backend cannot use TMS.

After installing the complete training runtime, the PP and PD cases also passed consecutively in a single pytest process: **2 passed, 0 skipped, 281.28 seconds**, step `6654867.18`.

Within the same allocation, full Megatron job `raysubmit_tg4D86NcgNJpSNJF` (step `6654867.17`) finished successfully. It used Qwen3-0.6B, training TP=1 / PP=1 / DP=2, inference TP=2, selective offload, deferred Teacher and deferred GenRM. The production Controller completed two rounds of generation, Teacher writeback, GenRM reward, training and checkpoint saving, with four samples and one optimizer step per round. Losses were 0.236228 and 0.487990, with gradient norms 19.0095 and 26.6086. All eight training samples had complete, finite Teacher probabilities aligned with their responses; GenRM received eight actual requests, and rollout weight versions advanced from 1 to 2. Comparing the original HF weights and both saved checkpoints confirmed that the inspected Q projection parameter changed after each optimizer step.

Submit the training script through the repository launcher, with an isolated two-node Ray cluster, Megatron-Bridge, Transformer Engine, Apex and a shared output directory. Set `MEGATRON` to the configured Megatron source directory:

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

This script checks the complete training path, not model quality. The run used TE 2.14.1 and the repository's pinned Megatron-Bridge/MCore revisions and patches. Split validation used the default TMS path; selective offload remains an explicit fallback for backends that require it.

The role tests use a test pool; the topology tests construct the production RolloutManager, its router and real engines; the three-role validation uses the complete Controller. Together they cover the explicit RFC acceptance checks.

## Next steps

- [Rollout API](./rollout.md)
- [GenRM API](./genrm.md)
