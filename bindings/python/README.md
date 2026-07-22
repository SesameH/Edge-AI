# unirt (Python binding)

Python binding for the UniRT SDK — run LLMs locally through a single API with
interchangeable backends:

| runtime     | models                                               | hardware                    |
|-------------|------------------------------------------------------|-----------------------------|
| `llama_cpp` | GGUF                                                 | CPU / Metal / Vulkan / CUDA |
| `mlx`       | HF safetensors (validated SmolLM2-style Llama/ByteLevel-BPE layout; dense or MLX-quantized) | Apple Silicon Metal GPU |
| `onnxruntime` | ONNX encoder embeddings                         | CPU / Apple Core ML     |

The bundled llama_cpp runtime supports GGUF VLMs through libmtmd when an
mmproj is present. MLX remains text-only and fails explicitly for VLM models.
MLX also requires a usable Metal device; if none is visible, model loading
fails before native model allocation.

## Install

```sh
pip install unirt
```

macOS arm64 wheels ship every native library — no toolchain, no build step.

## Quickstart (CLI)

```sh
unirt chat bartowski/SmolLM2-135M-Instruct-GGUF   # download + interactive chat
unirt pull <hf-repo>                              # download only
unirt ls                                          # cached models
unirt devices                                     # plugins + devices
```

## Usage

```python
from unirt.auto import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'bartowski/SmolLM2-135M-Instruct-GGUF',
    precision='Q4_K_M',
    device_map='llama_cpp',
)
out = model.generate(prompt, max_new_tokens=128, temperature=0.7)
print(out.text)
model.close()
```

Embedding repositories select an ONNX variant and tokenizer sidecars without
downloading the PyTorch checkpoint:

```python
from unirt import AutoModelForEmbedding

with AutoModelForEmbedding.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2',
    device_map='cpu',  # or 'coreml' on Apple Silicon
) as model:
    vectors = model.encode(['a cat on a mat', 'a kitten on a rug'])
    print(len(vectors), len(vectors[0]))  # 2, 384
```

Repository ids are inspected and downloaded with `huggingface_hub`. GGUF
repositories download only the selected quantization (including all of its
shards) plus tokenizer/config sidecars. The default cache is
`~/.cache/unirt`; set `UNIRT_DATADIR` to move it and `UNIRT_HFTOKEN` for gated
or private repositories.

Generation is stateless by default (`n_past=0` clears prior KV state before
prefilling the supplied prompt). To continue from a known cached prefix, pass
the exact prefix length through `n_past`; invalid values are rejected rather
than silently duplicating context.

## OpenAI-compatible server

```sh
python3 -m unirt.server --model bartowski/SmolLM2-135M-Instruct-GGUF \
  --backend llama_cpp --port 8080
```

Then point any OpenAI client (or plain curl) at
`http://localhost:8080/v1/chat/completions` — streaming SSE included, and
GGUF VLMs accept image content parts when loaded with an mmproj.

The native library is discovered automatically from `<repo>/sdk/pkg-unirt/lib`
(dev layout) or the packaged wheel; set `UNIRT_LIB_PATH` / `UNIRT_PLUGIN_PATH`
to override. See the repository README for build instructions, the interactive
chat example, and the OpenAI-compatible server (`python3 -m unirt.server`).
