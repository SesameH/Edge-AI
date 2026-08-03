# Changelog

Versions are the tags on this repository; each one builds the native
libraries for every platform and publishes the Python wheel.

## 0.5.0

### Added

- **Continuous batching** on the llama.cpp backend. `--slots N` now puts the N
  handles on one context and decodes them in a single batch instead of running
  N contexts side by side. Eight clients, four slots, 64 tokens each:

  | | throughput | slowest request |
  |---|---|---|
  | 135M Q8_0, CPU | 84 → 630 tok/s | 6.09 → 0.81 s |
  | 135M Q8_0, Metal | 183 → 236 tok/s | 2.80 → 2.17 s |
  | 1.7B Q4_K_M, CPU | 8.2 → 75.7 tok/s | 62.2 → 6.8 s |
  | 1.7B Q4_K_M, Metal | 39.9 → 79.7 tok/s | 12.8 → 6.4 s |

  Part of the CPU figure is a defect removed rather than a win earned: separate
  contexts each built their own thread pool, so four slots ran four pools on
  eight cores. Splitting the threads by hand got separate contexts to 281 tok/s
  on the 135M; the batch still doubles that.

  Two things to know. Slots now share a pool of KV cells, so the same request
  can come back a token different depending on what else was decoding beside it
  -- the usual bargain for batched serving, and `--slots 1` still reproduces
  exactly. And saved KV sessions are now per sequence
  (`llama_state_seq_save_file`), so a session file written by 0.4.0 will not
  load.

  Available directly as `n_seq_max` on `unirt_ModelConfig` / `from_pretrained`.
  A handle with a draft model keeps a context to itself: speculative
  verification reads several positions of a batch it has to own.

- **Slots lend each other cached prompts.** A request starting on one slot
  looks for an idle sibling holding a longer prefix of the same prompt and
  takes it rather than evaluating it again; with a shared pool the cells are
  pointed at, not copied, so a system prompt is stored once however many slots
  read it. One client warms an 1854-token system prompt, then three more
  arrive: 11.67 s -> 0.29 s.

- **Long prompts no longer freeze the other streams.** A prompt is submitted
  one physical batch (`n_ubatch`) at a time, so a round carries a slice of it
  plus everyone else's next token. Three streams running, a 2000-token prompt
  arriving: worst gap between tokens 1650 ms -> 600 ms on CPU, 440 ms -> 145 ms
  on Metal, for 4-5% on the arriving request's own latency.

### Fixed

- A VLM opened with `n_seq_max` (which `--slots N` now passes) had its context
  divided between sequences it can never have — media position state is per
  handle, so the backend owns exactly one. It quietly left the caller a
  fraction of the window it asked for.
- Log records from several sequences interleaved into each other. The stream
  was always safe to share; a record is four insertions, and they arrived
  shuffled.

## 0.4.0

### Fixed — read this one if you use Gemma, Llama or Mistral

**BOS was dropped from every turn after the first** on any vocabulary that
adds it. The prompt was tokenized with `add_special` keyed off "is the KV
cache empty" instead of off `n_past`, so from turn two the model stopped
seeing a token it was trained to always see — and because the cached BOS then
sat where the prompt's first real token was, the prefix match failed at
position 0 and re-prefilled the whole transcript every turn. On Gemma the
reply came back empty. SmolLM2 sets `add_bos_token=false` and cannot catch
this, which is why it survived a green test suite; the suite now includes a
model that can.

`sliding_window` **now means what `unirt.h` says it means**, on both backends.
Overflow is reported as `context_length` unless the option asks for eviction.
Previously llama_cpp never read the field and shifted regardless, while mlx
shifted when unasked *and* rejected the option with `PARAM_NOT_SUPPORTED` when
asked. If you called `generate()` directly and relied on the old implicit
shifting, pass `sliding_window=True` — the server and `examples/chat.py` do,
so their behaviour is unchanged. When mlx shifts it now keeps a pinned head,
so the system prompt survives eviction.

Also: `hmac.compare_digest` crashed on a non-ASCII `--api-key` (rejected at
startup now); `rerank --json` reported a duplicated document's index wrong;
`/v1/stats` raised `StopIteration` with no model loaded.

### Added

- **Speculative decoding.** A draft model of the same vocabulary proposes
  tokens that the real model verifies in one batch. Opt-in via
  `draft_model=` / `--draft-model`, `n_draft` per request. The text is
  identical to decoding without it. Measured 13% *slower* on a 1.7B/135M pair
  at 46% acceptance — see the README for when the bet pays and when it does
  not.
- **CPU instruction-set dispatch** on the Linux and Windows-x64 wheels. One
  CPU backend per level ships and ggml picks the best the machine supports,
  instead of every machine running the architecture baseline. Measured on
  aarch64: prefill 408 → 820 tok/s, decode 112 → 142 tok/s.
- **Vulkan** on the Linux wheels: NVIDIA, AMD, Intel and the mobile vendors
  from one build. A machine with no driver loses the Vulkan backend and
  nothing else.
- **Concurrent serving.** `--slots N` decodes N requests at once over shared
  weights; both backends now share a loaded model between handles, so a slot
  costs a KV cache rather than a copy of the model (llama.cpp: +16 MB per
  extra slot on a 138 MB model; MLX: 287 → 329 MB for four slots, against
  1143 MB before). What it buys is head-of-line blocking, not throughput: a
  short request behind a long one went 2.6 s → 0.27 s.
- **Several models per server.** `--model` may be repeated, and a request's
  `model` field picks between them. All but the first load on demand;
  `--max-resident-models` and `--model-idle-timeout` give them back.
  `/v1/models` lists what is really served, and an unknown name is a 404
  rather than a silent answer from the wrong model.
- **Log-probabilities**, streaming and blocking, on both endpoints, in each
  one's own response shape. Taken before any sampler or grammar touches the
  distribution.
- **`/v1/completions`** — the pre-chat endpoint: prompt in, no chat template.
- **Parallel tool calls**: one turn may return several.
- Cross-backend conformance suite (`tests/python/test_conformance.py`): one
  contract, every text backend, with the capability differences recorded in a
  matrix rather than skipped.

## 0.3.0

macOS wheels built for the macOS they promise; MLX's metallib shipped, so the
backend works off the build machine.

## 0.2.2, 0.2.1 — yanked

Both carried the BOS defect above, and 0.2.2 additionally linked OpenSSL into
the macOS and Windows builds, which made the plugin fail to load where those
libraries were absent.

## 0.2.0, 0.1.4, 0.1.3

Initial published versions: the C ABI, the llama.cpp and MLX backends, Python
bindings and the OpenAI-compatible server.
