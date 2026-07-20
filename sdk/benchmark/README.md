# unirt-bench — C inference benchmark example

Single-file C example that drives the public unirt C API. One invocation
runs one `(plugin, device, model)` cell (warmup + repeated measured runs)
and prints / writes TTFT, prefill_tps, decode_tps, gen_tokens.

Flag naming follows llama.cpp's
[`llama-bench`](../../third-party/llama.cpp/tools/llama-bench/README.md) —
`-r / --repetitions`, `-n / --n-gen`, `-c / --ctx-size`, `-t / --threads`,
`-m / --model`, `--n-gpu-layers`, `--no-warmup` — so users moving between
the two tools read the same vocabulary.

It accepts a **local model path** (a UniRT bundle directory, or a `.gguf`
file). Download remote repositories with the Python `unirt pull` command
before copying the resolved model directory to the target device. Runs on
Windows, Android, and Linux — the same binary feeds UniRT Bench.

## Build

Gated on the `UNIRT_BENCHMARK` cmake option, which the snapdragon presets in
[`sdk/CMakePresets.json`](../CMakePresets.json) enable for both `debug` and
`release`. The recipes below match [`notes/build.md`](../../notes/build.md).

### Windows ARM64 (Snapdragon)

> [!NOTE]
> The Hexagon toolchain has a 250-character path limit. Shorten the source
> path with `subst` before building if the repo lives under a long path:
> `subst G: C:\path\to\unirt; cd G:\sdk`

```powershell
cd sdk
cmake --preset arm64-windows-snapdragon-release -B build
cmake --build build -j --target unirt-bench
# → build\benchmark\unirt-bench.exe
cmake --install build --prefix pkg-unirt   # optional → pkg-unirt\bin\unirt-bench.exe
```

### Linux (cross-compile from x86_64)

Build inside the Snapdragon Linux toolchain container per
[`notes/build.md`](../../notes/build.md):

```bash
docker run --rm -u $(id -u):$(id -g) \
  --volume $(pwd):/workspace --workdir /workspace/sdk \
  --platform linux/amd64 \
  <linux-toolchain-image> \
  bash -c 'cmake --preset arm64-linux-snapdragon-release -B build-linux . \
    && cmake --build build-linux -j --target unirt-bench \
    && cmake --install build-linux --prefix pkg-unirt'
# → pkg-unirt/bin/unirt-bench
```

### Android (cross-compile from Linux)

```bash
docker run --rm -u $(id -u):$(id -g) \
  --volume $(pwd):/workspace --workdir /workspace/sdk \
  --platform linux/amd64 \
  <android-toolchain-image> \
  bash -c 'cmake --preset arm64-android-snapdragon-release -B build-android . \
    && cmake --build build-android -j --target unirt-bench \
    && cmake --install build-android --prefix pkg-unirt'
# → pkg-unirt/bin/unirt-bench
```

## Run

The binary loads `unirt.dll` / `libunirt.so` and the per-plugin backends the
same way the Python binding does — from the installed `pkg-unirt/lib` (and
`lib/llama_cpp`) layout. On Windows run it from the build/install tree so the
DLLs resolve; on Android/Linux export `LD_LIBRARY_PATH=./lib:./lib/llama_cpp`
and `UNIRT_PLUGIN_PATH=./lib` (see [`notes/run.md`](../../notes/run.md)).

```bash
# LLM, llama_cpp — point -m at a .gguf file
unirt-bench \
  --plugin llama_cpp --device hybrid \
  -m /path/to/Qwen3-0.6B-Q4_0.gguf

# LLM, the NPU engine — the bundle dir is the "model path"
unirt-bench \
  --plugin npu-engine --device npu \
  -m /path/to/npu-bundle/Qwen3-4B-Instruct-2507/

# VLM, llama_cpp/libmtmd — model and projector are separate GGUF files
unirt-bench \
  --plugin llama_cpp --device auto \
  -m /path/to/LFM2-VL-450M-Q4_0.gguf \
  --mmproj-path /path/to/mmproj-LFM2-VL-450M-Q8_0.gguf \
  --image /path/to/sample.jpg --accuracy -n 96

# VLM, the NPU engine — the vision encoder is baked into the bundle (no mmproj), so
# pass --vlm to force VLM mode plus one or more --image
unirt-bench \
  --plugin npu-engine --device npu --vlm \
  -m /path/to/npu-bundle/Qwen2.5-VL-7B-Instruct/ \
  --image /path/to/sample.jpg

# GPU (llama_cpp) — the gpu alias selects the GPU device and offloads all layers by
# default (-1); pass --n-gpu-layers to offload only some
unirt-bench \
  --plugin llama_cpp --device gpu \
  -m /path/to/Qwen3-4B-Q4_K_M.gguf

# Customise: prompt, sample count, output files
unirt-bench \
  --plugin llama_cpp --device hybrid \
  -m .../Qwen3-1.7B-Q4_0.gguf \
  --warmup 1 -r 3 \
  -n 128 --temperature 0.0 --seed 42 \
  --output-json results/qwen3-1.7b-hybrid.json \
  --cell-id Qwen3-1.7B-llama_cpp-hybrid

# Accuracy mode: single run, print the generated text (eyeball output quality,
# not speed). Pair with --prompt-file so the model sees a real prompt.
unirt-bench \
  --plugin llama_cpp --device hybrid \
  -m .../Qwen3-1.7B-Q4_0.gguf \
  --accuracy --prompt-file prompt.txt -n 128
```

On Windows the same invocations work with `.exe` and backslash paths, e.g.:

```powershell
build\benchmark\unirt-bench.exe --plugin npu-engine --device npu `
  -m $env:USERPROFILE\.cache\unirt\models\<org>\Qwen3-4B-Instruct-2507
```

Run `unirt-bench --help` for the full flag list.

## Defaults

- `n_gen=128`, `temperature=0.0`, `seed=42`
- `--warmup 1`, `-r 5` (5 measured runs after 1 warmup; pass `--no-warmup`
  to skip warmup)
- `--accuracy` pins a single run (`--warmup 0 -r 1`) and prints the generated
  text to stdout (`[gen ] ...`); use it to sanity-check output quality rather
  than timing. Pair with `--prompt-file`, since the default random-ids prefill
  yields meaningless text.
- llama_cpp gets a `[warmup=i]` / `[run=i]` suffix appended to the prompt
  so the KV cache is busted between runs
- for `--plugin npu-engine`, `prompt_tokens` and `prefill_tps` are reported over the
  padded prompt length `ceil(prompt_tokens / 128) * 128`: the the NPU engine engine pads
  input_ids to a 128-token prefill chunk, so the padded count reflects the work
  actually done (#1194). llama_cpp does no such padding and is reported as-is

## Per-cell JSON shape

```json
{
  "schema_version": "2",
  "cell_id": "Qwen3-0.6B-llama_cpp-cpu",
  "plugin": "llama_cpp",
  "device": "cpu",
  "device_id": null,
  "model_path": ".../Qwen_Qwen3-0.6B-Q4_0.gguf",
  "model_size_bytes": 368705536,
  "params": { "warmup": 1, "repetitions": 3, "n_gen": 128, ... },
  "runs": [ { "run_idx": 0, "ttft_us": 49758, "prefill_tps": 102.1, ... }, ... ],
  "agg": {
    "ttft_ms":     {"median": 49.8, "min": 47.4, "max": 52.1, "mean": 49.7, "stdev": 2.4},
    "prefill_tps": {"median": 102.1, "min": 98.0, "max": 110.3, "mean": 103.4, "stdev": 6.2},
    "decode_tps":  {"median": 60.9, "min": 58.1, "max": 62.5, "mean": 60.5, "stdev": 2.3},
    "gen_tokens":  {"median": 128},
    "prompt_tokens":{"median": 42}
  }
}
```

## Markdown row shape

`--output-md` (and the QDC bench report) produce a llama-bench-aligned table:

```
| Model     | Size    | Backend    | Device | ngl | Test       | TTFT (ms)   | Prefill (tok/s) | Decode (tok/s) |
|-----------|--------:|------------|--------|----:|------------|------------:|----------------:|---------------:|
| Qwen3-0.6B| 351 MiB | llama_cpp  | cpu    |   - | pp42+tg128 | 49.8 ± 2.4  |  102.1 ± 6.2    |  60.9 ± 2.3    |
```

## Matrix-style runs

Run the C binary in **matrix mode** so a single `unirt_init` covers the
whole sweep — Hexagon FastRPC sessions and other plugin init costs are
then amortised across cells:

```bash
cat > matrix.tsv <<EOF
# cell_id<TAB>plugin<TAB>device<TAB>model_path[<TAB>tokenizer_path][<TAB>mmproj_path]
Qwen3-0.6B-llama_cpp-cpu	llama_cpp	cpu	/data/local/tmp/.cache/unirt/models/bartowski/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf
Qwen3-0.6B-llama_cpp-npu	llama_cpp	npu	/data/local/tmp/.cache/unirt/models/bartowski/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf
Qwen3-4B-npu-engine-npu	npu-engine	npu	/data/local/tmp/.cache/unirt/models/<org>/Qwen3-4B-Instruct-2507
EOF

unirt-bench --matrix-file matrix.tsv --output-json-dir results/
```

For a one-cell-per-process invocation (cold-start each time, useful as
the reference for a customer-facing single-call workload), pass
`--plugin / --device / -m` directly without `--matrix-file`.

Column 4 is always a local path. Pre-pull repositories on the host with
`unirt pull owner/repo --quant Q4_0`, then use `unirt ls` to inspect the
cached variants before copying them to the benchmark device.

```bash
cat > matrix.tsv <<EOF
# cell_id<TAB>plugin<TAB>device<TAB>model_path
Qwen3-0.6B-cpu	llama_cpp	cpu	/data/models/Qwen_Qwen3-0.6B-Q4_0.gguf
Qwen3-4B-npu-engine	npu-engine	npu	/data/models/qwen3_4b
EOF
unirt-bench --matrix-file matrix.tsv --output-json-dir results/
```
