---
outline: deep
---

# Unified inference validation

This page records the acceptance conclusions for [RFC #71](https://github.com/redai-studio/Relax/issues/71). It reports only the required outcomes.

## Acceptance conclusion

| RFC requirement | Conclusion | Evidence |
| --- | --- | --- |
| Rollout, GenRM, and Teacher share the gateway and manager implementation | **Passed** | Shared `InferenceGateway`, `InferenceManager`, lifecycle, routing, and client contracts |
| Static models do not participate in dynamic weight updates | **Passed** | Static DCS/update guards and endpoint tests; GPU training updated Rollout only |
| Gateway/direct routing is consistent and discovery hides TP/PP workers | **Passed** | PP/PD topology checks and head-only discovery assertions |
| Invalid layouts are rejected before startup | **Passed** | Placement capacity, node span, overlap, defer, and co-residency tests |
| Defer has no resource conflict and Teacher results are written before training | **Passed** | Deferred publication tests and four-node split/decoupled training results |
| Lifecycle is idempotent, failures roll back, PG ownership is preserved, and legacy interfaces remain compatible | **Passed** | Lifecycle, cleanup, borrowed-PG, compatibility, and recovery contract tests |
| Multi-node GPU validation is complete | **Passed** | `6654867` PP/PD plus Megatron, `6772050` split, and `6780723` decoupled training |

## Test conclusion

The inference contract suite completed with **91 passed and 1 skipped**. The skip is the Ray-only port allocator check on a workstation without Ray. Python compilation, launcher syntax, and diff checks passed.

## GPU conclusion

The required multi-node GPU scenarios passed:

- PP/PD routing and discovery passed.
- Split training passed with Teacher writeback before GenRM and Megatron steps completed.
- Decoupled training passed with Teacher results available before training and dynamic Rollout updates completed.
- Full Megatron data flow and checkpoint publication passed.

The implementation and validation are scoped to the explicit RFC #71 acceptance checks.

## Reproduction metadata

| Item | Value |
| --- | --- |
| Model | Qwen3-0.6B |
| GPU platform | GH200, one GPU per node |
| PP/PD and Megatron allocation | `6654867`, two nodes: `nid010294`, `nid010319` |
| Split allocation | `6772050`, four nodes: `nid010235`, `nid010244`, `nid010252`, `nid010256` |
| Decoupled allocation | `6780723`, four nodes: `nid010231`, `nid010240`, `nid010254`, `nid010255` |
| Runtime | SGLang 0.5.17, Ray 2.58.0, PyTorch 2.11.0+cu129, Transformer Engine 2.14.1 |

```bash
# CPU contracts
python -m pytest -q tests/inference

# Two-node PP/PD topology
INFERENCE_RUN_GPU_TESTS=1 \
INFERENCE_PD_BACKEND=mooncake_tcp MC_FORCE_TCP=1 \
python -m pytest -q tests/integration/test_unified_inference_topology_gpu.py

# Four-node three-role training
INFERENCE_LAYOUT=decoupled \
MODEL_DIR=/shared/Qwen3-0.6B \
INFERENCE_TRAIN_OUTPUT=/shared/unified-decoupled \
bash scripts/training/hpc/run-unified-inference-3role.sh

INFERENCE_LAYOUT=split \
MODEL_DIR=/shared/Qwen3-0.6B \
INFERENCE_TRAIN_OUTPUT=/shared/unified-split \
bash scripts/training/hpc/run-unified-inference-3role.sh
```
