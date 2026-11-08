# GLM-5.3-Flash on 8× V100

GLM-5.3-Flash on 8× V100-SXM2-32GB, set up for one agentic coding session: a main conversation that grows towards 200k tokens, subagent conversations next to it, and many short tool-call turns. A turn that extends a cached conversation prefills only the new tokens. The cache holds the linear-attention states of two conversations side by side, so a subagent call does not evict the main session. MTP only proposes tokens; the target model decides each one. On the check prompts, greedy output with MTP matched the server without MTP token for token.

## Get the model

The validated checkpoint is RadixArk's NVFP4 quantisation (ModelOpt). 190 GB. The MTP draft layer is in the same repo. The export carries a vision tower, which the wrapper now loads by default for image inputs. Set `GLM53_LANGUAGE_ONLY=1` to skip it and use the original text-only recipe. The model is MIT-licensed.

```bash
pip install -U "huggingface_hub[cli]"
hf download RadixArk/GLM-5.3-Flash-NVFP4 \
  --local-dir ~/models/GLM-5.3-Flash-NVFP4

export GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4
```

| | |
|---|---|
| checkpoint | [`RadixArk/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) |
| base model | [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) |
| quantisation | NVFP4 weights (W4A4 export) run as W4A16 with FP16 activations, FP16 KV cache. The draft layer's BF16 routed experts are quantised to NVFP4 at load |

## Talking to it

```bash
# OpenAI-compatible
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'

# Anthropic Messages API — Claude Code connects to this directly
curl http://127.0.0.1:11435/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"glm53-flash-nvfp4","max_tokens":256,"messages":[{"role":"user","content":"Hello"}]}'
```

The model reasons before it answers (`glm45` parser), and tool calls come back as structured `tool_calls` (`glm47` parser). An empty `content` with a large `completion_tokens` means the reply hit `max_tokens` while still thinking.

The model has three reasoning efforts: Low, High and Max. The server reads `reasoning_effort` (OpenAI) or `output_config.effort` (Anthropic, which Claude Code's effort setting sends) and maps it as follows: `minimal`/`low` to Low, `medium`/`high` to High, and `xhigh`/`max` (or no effort) to Max. Claude Code's default is `xhigh`, so it runs at Max.

## Image inputs

Use the pinned Transformers 5.17.0 from `requirements.txt`. Older V100
environments with Transformers 5.8.1 lack `Glm5NextProcessor` and can silently
load a tokenizer that drops images. The wrapper checks for the vision processor
before starting the server.

The wrapper enables the existing GLM vision processor and tower with FP16
activations and `--mm-attention-backend sdpa` on V100. It defaults to
`GLM53_MAX_IMAGE_TOKENS=8000` per image, the checkpoint-native budget. The
checkpoint processor keeps its normal geometry rules; the server adds no lower
resolution fallback. GLM preprocessing runs on CPU with one background worker
by default. On Volta, the encoder stages one frame at a time and packed SDPA
chunks queries against every key in that frame. This preserves full-frame
softmax and frame isolation without allocating a square attention mask. The
checkpoint's exclusion list keeps the vision tower unquantized.

`GLM53_MAX_IMAGES=4` bounds image packs. Larger packs receive HTTP 400 before
tensor-parallel dispatch, and subsequent valid requests remain usable. Raising
this limit or the processor worker count requires another capacity check.

Send an OpenAI image URL or an inline PNG/JPEG data URL:

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-nvfp4","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},{"type":"text","text":"Describe this image."}]}],"reasoning_effort":"low","max_tokens":2048}'
```

The Anthropic endpoint accepts image blocks with
`"source":{"type":"base64","media_type":"image/png","data":"..."}` alongside
text blocks. To check both endpoints against a running server:

```bash
python test/manual/test_glm53_flash_vision.py
```

Run that check with the default MTP recipe and again with
`GLM53_MTP_STEPS=0`. On 2026-11-07, both modes passed on eight V100-SXM2-32GB
GPUs: the OpenAI and Anthropic endpoints identified red and blue images,
including a 1120×1120 image resized to the 1,024-token limit. The initial checks
used an 8,192-token context and a 16,384-token KV pool. The checks also passed
at the full memory settings below, including MTP with FP8 E4M3 KV.

The following pool allocations were measured with the earlier 1,024-token
image smoke checks; they do not establish native-page capacity at those context
lengths. The wrapper retains these context defaults:

| Vision recipe | Memory fraction | Measured KV pool | Default context |
|---|---:|---:|---:|
| No MTP, FP16 KV | 0.88 | 181,760 | 178,176 |
| MTP 3 steps, FP16 KV | 0.935 | 201,280 | 196,608 |
| MTP 3 steps, FP8 E4M3 KV | 0.945 | 394,752 | 384,000 |

These are allocation and image smoke checks, not long-context accuracy or
throughput measurements. For the validated native-page workload, use the
131,072-token no-MTP profile below. Recheck capacity when changing context,
memory fraction, Mamba slots, or image admission limits. Use
`GLM53_LANGUAGE_ONLY=1` to retain the original text-only context defaults.

## Native document validation (2026-11-08)

[Issue #1](https://github.com/frankier/sglang-sxm2/issues/1) reproduced two
separate allocations: GPU image resizing and dense vision attention. CPU
preprocessing and bounded full-key query attention fix these without reducing
the native 8,000-token image budget. The original 8300×5834 page now reports
7,950 image tokens (31,800 patches), completes normally, and no longer kills
TP workers. A final-code no-MTP follow-up passed two OpenAI requests for
this page and an Anthropic request with a complete answer. The scheduler
reported 8,000 cached prompt tokens on the repeated request. API cache reporting
was disabled, so its cached-token counter remained zero; add
`--enable-cache-report` when you need that API counter. Four such frames in one request also completed with correct color
ordering and 31,800 total image tokens. Unequal frame packs and repeated mixed
image requests passed; a five-image request was rejected with HTTP 400 before
successful follow-ups.

Run the validated profile with:

```bash
export GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4
export GLM53_LANGUAGE_ONLY=0
export GLM53_MAX_IMAGE_TOKENS=8000
export GLM53_MAX_IMAGES=4
export GLM53_KV_CACHE_DTYPE=auto
export GLM53_MTP_STEPS=0
export GLM53_CONTEXT_LENGTH=131072
export GLM53_MEM_FRACTION=0.88
export GLM53_MAX_RUNNING=4
export GLM53_MAMBA_SLOTS=20
export GLM53_CUDA_GRAPH_BS_DECODE="1 2 4"
export GLM53_EXTRA_ARGS="--max-total-tokens 131072"
bash scripts/serve_glm53_flash_nvfp4_v100.sh
```

The single-client baseline used one active request, 12 Mamba slots and decode
graph batch size 1. All runs used TP8/EP1, FP16 activations and KV, no CPU weight
offload, SDPA vision, and 4,096-token prefill chunks. Hardware was eight
V100-SXM2-32GB GPUs on `dgx1-02`, driver 580.178.04, PyTorch 2.9.1+cu128,
Transformers 5.17.0 and Tokenizers 0.23.1. Checkpoint revision:
`f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`.

Each cohort used the same 16 OmniDocBench pages and original image bytes,
temperature 0, low reasoning effort and 8,192 output tokens. There were no
retries, output caps, loop guards, empty answers or missing usage: **80/80
responses completed**, and the unchanged official evaluator scored all 16
pages in every cohort.

| Connected clients | Observed active decode | Estimated overlapping output spans | Valid pages | Full-cohort tok/s | Median TTFT (s) | Max TTFT (s) | Highest sampled GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 16/16 | 63.23 | 6.49 | 21.78 | 30,148 |
| 2 | 2 | 2 | 16/16 | 84.30 | 6.82 | 22.00 | 30,392 |
| 4 | 4 | 4 | 16/16 | 97.51 | 14.83 | 37.66 | 30,392 |
| 6 | 4 | 4 | 16/16 | 106.94 | 29.37 | 46.46 | 30,648 |
| 8 | 4 | 4 | 16/16 | 107.94 | 46.06 | 68.97 | 30,648 |

Throughput is total API usage completion tokens, including reasoning, divided
by full cohort elapsed time. It is not SSE event count or decode-only speed.
Active decode counts come from scheduler logs. Output overlap is an estimate
using response-file completion timestamps and client elapsed/TTFT/decode
measurements; absolute request starts were not recorded. Output spans include
stalls. GPU memory was sampled every 200 ms, rather than measured as exact
allocator peaks. The single-client run followed the failure replay, and later
cohorts reused the four-active-request server; these are acceptance runs with
warm-cache effects, not a cold-cache throughput comparison. Six and eight
clients queue above four active requests.

Official scores (edit distances lower is better; CDM/TEDS higher is better):

| Clients | Text edit | Formula edit | Formula CDM | Table TEDS | Structure TEDS | Table edit | Reading-order edit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1, 2, 6, 8 | 0.058210 | 0.185039 | 0.989333 | 0.538981 | 0.575954 | 0.314073 | 0.274212 |
| 4 | 0.055162 | 0.185039 | 0.989333 | 0.538137 | 0.575130 | 0.313952 | 0.274212 |

The evaluator reported no match/CDM/TEDS timeouts or fallback pages. Small
batch-dependent output differences remain; exact text identity across batch
sizes is not claimed. The V100 attention regression compares FP16 packed,
unequal frames against dense masked attention at `atol=rtol=1e-3` (maximum
absolute difference 0.0009765625). The 31,800-patch case used 91.7 MiB peak
allocated attention-test memory, with maximum sampled-query difference
0.00001526. A two-layer random-weight vision encoder also matched batched and
frame-staged execution within `1e-3`; this is an operator check, not a second
checkpoint accuracy evaluation. CPU regressions use `1e-6` tolerances and
verify that changing one frame cannot affect another.

Run the GPU regression independently of a checkpoint:

```bash
python test/manual/test_glm53_native_vision_attention.py
```

[Machine-readable results](../../benchmark/glm53_native_vision_20261108/summary.json)
include per-rank sampled memory, usage, hashes, metrics and artifact locations.
Raw requests, server logs, telemetry, evaluator outputs and baseline health
bundles are retained in
`robefran@turso.cs.helsinki.fi:~/sglang-issue1-20261108/results/run-2945071`.
The original failed runs under
`~/llmturso-work/results/sglang-sxm2-vision/` remain intact. These native-page
results cover the no-MTP profile above; the earlier MTP/small-image smoke
checks do not establish native-page MTP capacity or long-context accuracy. An
additional MTP=3 startup attempt in this validation failed before serving, in
draft NVFP4 scale preparation (`NVFP4 block scales must be non-negative`); its
logs are retained under `results/mtp-2945090`. Use the tested no-MTP profile
for this document workload.

## Measured performance

These measurements used text-only mode (`GLM53_LANGUAGE_ONLY=1`).

The measured text-only recipe: TP=8, MTP 3 steps / 4 draft tokens, `--max-running-requests 1`, temperature 0. KV pool 242,880 tokens, context 240,640.

**Decode** (2026-10-06). Six chat prompts, 512 tokens each, streamed from `/generate`; tok/s excludes TTFT. Accept length is tokens per target forward.

| prompt | tok/s | accept length |
|---|---:|---:|
| code | 174.3 | 2.89 |
| refactor | 180.0 | 2.99 |
| explain | 152.3 | 2.52 |
| math | 196.6 | 3.26 |
| agent step | 179.0 | 2.98 |
| German prose | 175.3 | 2.89 |
| **all six** | **175.2** | |

**Agent session** (2026-10-02). Through `/v1/chat/completions` with a `bash` tool: source files as the first user message, then three turns that each append an assistant tool call and its result (~500–700 new tokens). TTFT includes rendering and tokenizing the whole chat on the server. The first request with a new set of tools also compiles its tool-call grammar, about 7 s once per tool set; the rows below were measured after that.

| context | cold first turn | tool turns 1 / 2 / 3 |
|---|---:|---:|
| 32,187 tokens | 16.1 s | 554 / 490 / 443 ms |
| 128,188 tokens | 68.1 s | 701 / 619 / 550 ms |
| 190,188 tokens | 104.7 s | 768 / 702 / 602 ms |

**Main session plus subagents** (2026-10-02). A 190k main conversation, then two fresh ~20k subagent conversations of three turns each, with a main turn after each subagent. The main turns stay cached:

| turn | TTFT |
|---|---:|
| main, cold / next turn | 103.9 s / 757 ms |
| subagent 1, cold / turns 2–3 | 9.68 s / 538, 515 ms |
| main | 694 ms |
| subagent 2, cold / turns 2–3 | 9.62 s / 531, 503 ms |
| main | 633 ms |

## Reference recipe

The wrapper is the supported entry. The numbers above were measured with `GLM53_LANGUAGE_ONLY=1`; the current default also loads vision. It defaults to MTP with 3 steps; `GLM53_MTP_STEPS=0` turns MTP off and switches the memory and context defaults to the no-MTP values below.

```bash
export GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4
bash scripts/serve_glm53_flash_nvfp4_v100.sh
```

### High-concurrency profile: four real decode requests

On 8x V100-SXM2, the no-MTP profile below forms a real batch of four. The
`--max-running-requests` value alone is not enough: the Mamba state pool caps
admission at `max_mamba_cache_size // 5` in this path. Use 20 Mamba slots and
capture graph batch sizes 1, 2, and 4 together.

```bash
export GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4
export GLM53_LANGUAGE_ONLY=1
export GLM53_MTP_STEPS=0
export GLM53_KV_CACHE_DTYPE=fp8_e4m3
export GLM53_CONTEXT_LENGTH=256000
export GLM53_MEM_FRACTION=0.945
export GLM53_MAX_RUNNING=4
export GLM53_MAMBA_SLOTS=20
export GLM53_CUDA_GRAPH_BS_DECODE="1 2 4"
bash scripts/serve_glm53_flash_nvfp4_v100.sh
```

Measured on 8x Tesla V100-SXM2-32GB: four simultaneous requests, 256 completion
tokens each, median aggregate throughput 250.5 tok/s over three rounds, with
`#running-req: 4` and `cuda graph: True` in the scheduler log. The 256k value is
the per-request maximum. Four requests near 240k each exceed the measured
751,040-token FP8 KV pool, so long requests still queue or receive a context
capacity error.

The verified MTP=1 dual-request profile is a useful alternative when two active
sessions matter more than four-way capacity: use `GLM53_MTP_STEPS=1`,
`GLM53_MAX_RUNNING=2`, `GLM53_MAMBA_SLOTS=20`, and
`GLM53_CUDA_GRAPH_BS_DECODE="1 2"`. On the same host it measured 188.5 tok/s
aggregate. MTP 2 or 3 steps measured about 55-60 tok/s with two simultaneous
requests, so the default MTP=3 profile is not the high-concurrency choice.

Expanded (the default vision-enabled launch with MTP). The env block is not implied by the CLI flags: it turns on the Volta decode kernels and the two-level all-reduce.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST=7.0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_ALGO=allreduce:tree
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_NUMA_BIND_V2=0
export SGLANG_SM70_GLM_NVFP4_GEMV=1
export SGLANG_SM70_GLM_NVFP4_MOE_DECODE=1
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_DSV41_HIER_AR=1
export SGLANG_DSV41_HIER_AR_CA=1
export SGLANG_DSV41_HIER_AR_PUSH=1
export SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE=1

python -m sglang.launch_server \
  --model-path "${GLM53_MODEL}" \
  --served-model-name glm53-flash-nvfp4 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend marlin \
  --mm-attention-backend sdpa \
  --mm-process-config '{"image":{"max_image_tokens":8000}}' \
  --limit-mm-data-per-request '{"image":4}' \
  --tensor-parallel-size 8 \
  --ep-size 1 \
  --attention-backend dsa \
  --linear-attn-backend triton \
  --kv-cache-dtype auto \
  --disable-custom-all-reduce \
  --disable-prefill-cuda-graph \
  --cuda-graph-bs-decode 1 \
  --max-running-requests 1 \
  --max-mamba-cache-size 12 \
  --mamba-max-states-per-path 2 \
  --mamba-full-memory-ratio 0.15 \
  --mamba-radix-cache-strategy extra_buffer \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --warmups prefix_reuse,sampling \
  --sleep-on-idle \
  --context-length 196608 \
  --mem-fraction-static 0.935 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "${GLM53_MODEL}" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| MTP | on, 3 steps / 4 draft tokens | Fastest measured. On an earlier build 2 steps gave 117 tok/s against about 123 for 3; 4 steps (5-row verify) falls off the 4-row kernels and dropped to 34 |
| draft experts | NVFP4 at load (`SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE=1`) | Draft weights 2.4 → 1.06 GB per rank, which buys KV pool, and acceptance is the same as the BF16 draft |
| `--ep-size` | 1 | Experts TP-sliced: every rank does the same MoE work per token. With EP=8 the other ranks waited for the one holding most routes |
| `--max-running-requests` | 1 by default; 4 in the no-MTP profile | Four requests require the Mamba state pool and graph batch sizes to be raised together |
| `--mem-fraction-static` | 0.935 | Vision: 201,280-token KV pool; text-only: 242,880. Without MTP use 0.88 |
| `--context-length` | 196,608 with vision; 240,640 text-only | Kept below the pool. Without MTP: 178,176 with vision; 223,232 text-only |
| `--max-mamba-cache-size` / `--mamba-max-states-per-path` | 12 / 2 by default; 20 / 2 for real batch 4 | In the no-MTP path the scheduler caps requests at `max_mamba_cache_size // 5`; 12 therefore silently caps four-request settings at 2 |
| prefill chunk | 4096 | Wrapper default; the older text-only benchmarks above used 2048 |
| `--warmups prefix_reuse,sampling` | on | Loads the kernels of a cache hit and a grammar-constrained decode at startup, and builds FlashInfer's sampling module. Otherwise they load on the first real tool turn, when little memory is free, and the first sampled request waits for the build |
| `--sleep-on-idle` | on | Without it the eight idle scheduler loops keep about 5 CPU cores busy. Decode speed is the same either way |
| `--disable-custom-all-reduce` + `SGLANG_DSV41_HIER_AR*` | on | The 8-GPU mesh is two NVLink quads plus bridges. Custom all-reduce inside each quad, then across the bridge pair |

Without MTP (`GLM53_MTP_STEPS=0` in the wrapper): drop the five `--speculative-*` lines and `SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE`, and use `--mem-fraction-static 0.88 --context-length 178176` with vision, or context 223232 in text-only mode.

## Limitations

- The default recipe is one request at a time. Use the high-concurrency profile above for real batch-4 decode.
- A fresh subagent conversation pays its own cold prefill (about 10 s at 20k tokens).
- The first request with a new set of tools compiles its tool-call grammar, about 7 s once per tool set.
- Checked with the benchmarks above and GSM8K (0.923 over 1,319 questions with MTP, on an earlier build of this recipe). No multi-hour soak yet.

## PCIe V100s

The recipe is measured on SXM2. The two-level all-reduce (`SGLANG_DSV41_HIER_AR*`) assumes two NVLink quads joined by bridges; on PCIe cards set `SGLANG_DSV41_HIER_AR=0` before the script. Untested on an 8-card PCIe box; expect slower decode.
