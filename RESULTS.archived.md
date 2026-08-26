# Agent Memory Benchmark — Archived Results

Archived results. LongMemEval is paused — see the [Datasets](README.md#datasets) section of the README for why — so nothing here is regenerated. This is the record as it stood when the dataset was last run, kept with the tag that produced it. The live results are in [RESULTS.md](RESULTS.md).

Results from `mem0/v0.2.0` (commit `fe5da80`), run 2026-08-23.

## Every run

Latest run per system x dataset x models. Each row's full per-question output lives in `runs/<dataset>/<system>/<mode>/<run-id>/` — or with an extra `<variant>` level (`runs/<dataset>/<variant>/<system>/<mode>/<run-id>/`) for a dataset run with `--variant`, e.g. LongMemEval's oracle/s/m, since those are different experiments sharing one dataset name, not different runs of the same one.

| dataset | variant | system | system_version | mode | k | ingestion_model | embedding_model | system_params | model | judge_model | max_turns | sample_seed | workers | num_questions | memory_tokens_total | retrieval_f1 | retrieval_precision | retrieval_recall | turn_f1 | turn_precision | turn_recall | search_p50_s | search_p95_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longmemeval | oracle | fraise | 0.1.0-beta.7 | direct | 1 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 394195.000 | 0.623 | 1.000 | 0.460 | 0.407 | 0.542 | 0.342 | 0.2176 | 0.2363 |
| longmemeval | oracle | fraise | 0.1.0-beta.7 | direct | 10 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 376373.000 | 0.912 | 1.000 | 0.872 | 0.432 | 0.307 | 0.825 | 0.2905 | 0.3756 |
| longmemeval | oracle | fraise | 0.1.0-beta.7 | direct | 3 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 387599.000 | 0.685 | 1.000 | 0.545 | 0.450 | 0.425 | 0.533 | 0.2197 | 0.2886 |
| longmemeval | oracle | fraise | 0.1.0-beta.7 | direct | 5 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 384634.000 | 0.821 | 1.000 | 0.752 | 0.477 | 0.382 | 0.675 | 0.2586 | 0.5353 |
| longmemeval | oracle | fraise | 0.1.0-beta.8 | direct | 5 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 7267650.000 | 0.838 | 0.934 | 0.794 | 0.431 | 0.348 | 0.662 | 0.1945 | 0.3372 |
| longmemeval | oracle | fraise | 0.1.0-beta.8 | direct | 3 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 7227416.000 | 0.794 | 0.944 | 0.728 | 0.474 | 0.440 | 0.584 | 0.2422 | 0.3518 |
| longmemeval | oracle | fraise | 0.1.0-beta.8 | direct | 1 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 7282748.000 | 0.706 | 0.946 | 0.604 | 0.454 | 0.585 | 0.400 | 0.2280 | 0.3634 |
| longmemeval | oracle | fraise | 0.1.0-beta.8 | direct | 10 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 7219017.000 | 0.856 | 0.938 | 0.817 | 0.349 | 0.248 | 0.723 | 0.2215 | 0.3305 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 5 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 1268726.000 | 0.919 | 1.000 | 0.878 |  |  |  | 0.2317 | 0.2865 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 1 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 1259994.000 | 0.623 | 1.000 | 0.460 |  |  |  | 0.2402 | 0.5248 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 3 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 1259325.000 | 0.860 | 1.000 | 0.803 |  |  |  | 0.2845 | 0.4127 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 10 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 1276025.000 | 0.933 | 1.000 | 0.897 |  |  |  | 0.3192 | 0.3517 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 5 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 23192824.000 | 0.898 | 0.960 | 0.866 | 0.000 | 0.000 | 0.000 | 0.2073 | 0.3307 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 3 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 23606629.000 | 0.862 | 0.956 | 0.816 | 0.000 | 0.000 | 0.000 | 0.1969 | 0.3141 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 1 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 23468224.000 | 0.727 | 0.968 | 0.625 | 0.000 | 0.000 | 0.000 | 0.1793 | 0.2670 |
| longmemeval | oracle | graphiti | 0.29.3 | direct | 10 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 23551637.000 | 0.928 | 0.958 | 0.913 | 0.000 | 0.000 | 0.000 | 0.2572 | 0.4465 |
| longmemeval | oracle | letta | 0.16.8 | direct | 3 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  |  | 20 | 163633.000 | 0.788 | 1.000 | 0.702 | 0.466 | 0.400 | 0.583 | 0.4091 | 0.4636 |
| longmemeval | oracle | letta | 0.16.8 | direct | 5 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  |  | 500 | 3139134.000 | 0.952 | 1.000 | 0.929 | 0.400 | 0.278 | 0.802 | 0.2866 | 0.4174 |
| longmemeval | oracle | letta | 0.16.8 | direct | 10 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  |  | 500 | 3139134.000 | 0.989 | 1.000 | 0.982 | 0.285 | 0.175 | 0.928 | 0.3511 | 0.5504 |
| longmemeval | oracle | letta | 0.16.8 | direct | 1 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  |  | 500 | 3139134.000 | 0.751 | 1.000 | 0.644 | 0.429 | 0.553 | 0.374 | 0.2620 | 0.4053 |
| longmemeval | oracle | letta | 0.16.8 | direct | 10 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  | 3 | 500 | 2850103.000 | 0.991 | 1.000 | 0.986 | 0.288 | 0.176 | 0.934 | 0.3354 | 0.4590 |
| longmemeval | oracle | letta | 0.16.8 | direct | 5 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  | 3 | 500 | 2850103.000 | 0.958 | 1.000 | 0.937 | 0.408 | 0.285 | 0.815 | 0.3149 | 0.4206 |
| longmemeval | oracle | letta | 0.16.8 | direct | 1 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  | 3 | 500 | 2850103.000 | 0.751 | 1.000 | 0.644 | 0.448 | 0.585 | 0.388 | 0.3158 | 0.5037 |
| longmemeval | oracle | letta | 0.16.8 | direct | 3 | openai/gpt-5-mini | openai/text-embedding-3-small |  |  |  |  |  | 3 | 500 | 2850103.000 | 0.898 | 1.000 | 0.850 | 0.467 | 0.376 | 0.685 | 0.4087 | 0.5522 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 1 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 967398.000 | 0.623 | 1.000 | 0.460 |  |  |  | 0.9611 | 1.1202 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 5 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 962165.000 | 0.945 | 1.000 | 0.920 |  |  |  | 1.0338 | 1.2642 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 3 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 964852.000 | 0.904 | 1.000 | 0.862 |  |  |  | 1.0393 | 1.2578 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 10 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  |  | 20 | 961743.000 | 0.968 | 1.000 | 0.948 |  |  |  | 0.9921 | 1.2829 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 10 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 18513695.000 | 0.993 | 1.000 | 0.989 |  |  |  | 0.5103 | 0.7981 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 5 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 18572785.000 | 0.977 | 1.000 | 0.964 |  |  |  | 0.6226 | 0.8972 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 3 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 18559135.000 | 0.942 | 1.000 | 0.912 |  |  |  | 0.5176 | 0.7698 |
| longmemeval | oracle | mem0 | 2.0.18 | direct | 1 | gpt-5-mini | text-embedding-3-small |  |  |  |  |  | 3 | 500 | 18579325.000 | 0.751 | 1.000 | 0.644 |  |  |  | 0.6114 | 0.9668 |

## Plots

Every metric is session-level — one exam, every system on it. Written by `amb plot all` into `plots/<dataset>/`, one directory per dataset (or `plots/<dataset>/<variant>/` for a dataset whose variants are separate experiments). Each dataset's section opens with its retrieval-vs-k sweep lines — the same overview the [README](README.md#results) shows — followed by the detail per k.

### longmemeval (oracle)

#### longmemeval (oracle): retrieval vs k

One line per system across the k sweep, newest run per system and k.

![Retrieval F1 vs k](plots/longmemeval/oracle/k_f1.png)

![Retrieval recall vs k](plots/longmemeval/oracle/k_recall.png)

![Retrieval precision vs k](plots/longmemeval/oracle/k_precision.png)

---

#### longmemeval (oracle): session-level comparison (k=1)

One bar per system, newest run.

![Session-level precision by system](plots/longmemeval/oracle/session_precision_k1.png)

![Session-level recall by system](plots/longmemeval/oracle/session_recall_k1.png)

![Session-level F1 by system](plots/longmemeval/oracle/session_f1_k1.png)

#### longmemeval (oracle): cross-metric trade-offs (k=1)

Retrieval quality against memory tokens spent and search latency.

![Retrieval precision vs memory tokens total](plots/longmemeval/oracle/tokens_precision_k1.png)

![Retrieval recall vs memory tokens total](plots/longmemeval/oracle/tokens_recall_k1.png)

![Retrieval F1 vs memory tokens total](plots/longmemeval/oracle/tokens_f1_k1.png)

![Retrieval precision vs search latency p50 (s)](plots/longmemeval/oracle/latency_precision_k1.png)

![Retrieval recall vs search latency p50 (s)](plots/longmemeval/oracle/latency_recall_k1.png)

![Retrieval F1 vs search latency p50 (s)](plots/longmemeval/oracle/latency_f1_k1.png)

![Search latency p50 (s) vs memory tokens total](plots/longmemeval/oracle/tokens_latency_k1.png)

---

#### longmemeval (oracle): session-level comparison (k=3)

One bar per system, newest run.

![Session-level precision by system](plots/longmemeval/oracle/session_precision_k3.png)

![Session-level recall by system](plots/longmemeval/oracle/session_recall_k3.png)

![Session-level F1 by system](plots/longmemeval/oracle/session_f1_k3.png)

#### longmemeval (oracle): cross-metric trade-offs (k=3)

Retrieval quality against memory tokens spent and search latency.

![Retrieval precision vs memory tokens total](plots/longmemeval/oracle/tokens_precision_k3.png)

![Retrieval recall vs memory tokens total](plots/longmemeval/oracle/tokens_recall_k3.png)

![Retrieval F1 vs memory tokens total](plots/longmemeval/oracle/tokens_f1_k3.png)

![Retrieval precision vs search latency p50 (s)](plots/longmemeval/oracle/latency_precision_k3.png)

![Retrieval recall vs search latency p50 (s)](plots/longmemeval/oracle/latency_recall_k3.png)

![Retrieval F1 vs search latency p50 (s)](plots/longmemeval/oracle/latency_f1_k3.png)

![Search latency p50 (s) vs memory tokens total](plots/longmemeval/oracle/tokens_latency_k3.png)

---

#### longmemeval (oracle): session-level comparison (k=5)

One bar per system, newest run.

![Session-level precision by system](plots/longmemeval/oracle/session_precision_k5.png)

![Session-level recall by system](plots/longmemeval/oracle/session_recall_k5.png)

![Session-level F1 by system](plots/longmemeval/oracle/session_f1_k5.png)

#### longmemeval (oracle): cross-metric trade-offs (k=5)

Retrieval quality against memory tokens spent and search latency.

![Retrieval precision vs memory tokens total](plots/longmemeval/oracle/tokens_precision_k5.png)

![Retrieval recall vs memory tokens total](plots/longmemeval/oracle/tokens_recall_k5.png)

![Retrieval F1 vs memory tokens total](plots/longmemeval/oracle/tokens_f1_k5.png)

![Retrieval precision vs search latency p50 (s)](plots/longmemeval/oracle/latency_precision_k5.png)

![Retrieval recall vs search latency p50 (s)](plots/longmemeval/oracle/latency_recall_k5.png)

![Retrieval F1 vs search latency p50 (s)](plots/longmemeval/oracle/latency_f1_k5.png)

![Search latency p50 (s) vs memory tokens total](plots/longmemeval/oracle/tokens_latency_k5.png)

---

#### longmemeval (oracle): session-level comparison (k=10)

One bar per system, newest run.

![Session-level precision by system](plots/longmemeval/oracle/session_precision_k10.png)

![Session-level recall by system](plots/longmemeval/oracle/session_recall_k10.png)

![Session-level F1 by system](plots/longmemeval/oracle/session_f1_k10.png)

#### longmemeval (oracle): cross-metric trade-offs (k=10)

Retrieval quality against memory tokens spent and search latency.

![Retrieval precision vs memory tokens total](plots/longmemeval/oracle/tokens_precision_k10.png)

![Retrieval recall vs memory tokens total](plots/longmemeval/oracle/tokens_recall_k10.png)

![Retrieval F1 vs memory tokens total](plots/longmemeval/oracle/tokens_f1_k10.png)

![Retrieval precision vs search latency p50 (s)](plots/longmemeval/oracle/latency_precision_k10.png)

![Retrieval recall vs search latency p50 (s)](plots/longmemeval/oracle/latency_recall_k10.png)

![Retrieval F1 vs search latency p50 (s)](plots/longmemeval/oracle/latency_f1_k10.png)

![Search latency p50 (s) vs memory tokens total](plots/longmemeval/oracle/tokens_latency_k10.png)

Every chart above is also browsable directly in [`plots/`](plots/), regenerated by `amb plot all` from the same report definition that wrote this file — nothing there or here is drawn or written by hand.
