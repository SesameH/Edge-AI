# unirt (Python binding)

Python binding for the UniRT SDK — run LLMs locally through a single API with
interchangeable backends:

| runtime       | models                                                | hardware                | in the published wheels |
|---------------|-------------------------------------------------------|-------------------------|-------------------------|
| `llama_cpp`   | GGUF (LLM, VLM, embeddings, rerank)                   | CPU, Metal on macOS     | yes                     |
| `mlx`         | HF safetensors (SmolLM2-style Llama/ByteLevel-BPE layout; dense or MLX-quantized) | Apple Silicon Metal GPU | no — build from source  |
| `onnxruntime` | ONNX encoder embeddings                               | CPU, Apple Core ML      | no — build from source  |

Run `unirt devices` to list what your install actually has. The wheels ship the
`llama_cpp` runtime only, built for a generic CPU baseline plus Metal on macOS —
no CUDA and no Vulkan. The other two runtimes exist in the source tree and
require building the SDK yourself.

The bundled llama_cpp runtime supports GGUF VLMs through libmtmd when an
mmproj is present. MLX remains text-only and fails explicitly for VLM models.
MLX also requires a usable Metal device; if none is visible, model loading
fails before native model allocation.

## Install

```sh
pip install unirt
```

Wheels ship the native libraries — no toolchain, no build step — for macOS 14+
arm64, Linux x86_64 and arm64 (`manylinux_2_31`, glibc 2.31+), and Windows 10+
x86_64 and arm64. Python 3.10+.

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

## Structured output

Constrain decoding so the reply is guaranteed to parse — the grammar masks
invalid tokens at every step, which makes even small models reliable JSON
emitters (llama_cpp backend; MLX rejects these options):

```python
schema = {'type': 'object',
          'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
          'required': ['city', 'country']}
out = model.generate('Facts about the capital of France as JSON.',
                     json_schema=schema)      # dict or serialized JSON string
data = json.loads(out.text)                    # always parses

model.generate(prompt, json_mode=True)         # any syntactically valid JSON
model.generate(prompt, grammar='root ::= ...')  # raw GBNF
```

The server accepts the OpenAI `response_format` field with types
`json_object` and `json_schema`. Note a `length` finish can still truncate
mid-object — budget `max_tokens` accordingly.

## OpenAI-compatible server

```sh
python3 -m unirt.server --model bartowski/SmolLM2-135M-Instruct-GGUF \
  --backend llama_cpp --port 8080
```

Then point any OpenAI client (or plain curl) at
`http://localhost:8080/v1/chat/completions` — streaming SSE included, and
GGUF VLMs accept image content parts when loaded with an mmproj.

`--embedding-model <encoder>` adds `/v1/embeddings` and `--rerank-model
<cross-encoder>` adds `/v1/rerank`; either may be given without `--model`, so a
retrieval sidecar needs no chat model at all.

The native library is closed-source and ships prebuilt. A wheel already bundles
it under `unirt/lib/`; a source checkout is discovered at
`<repo>/sdk/pkg-unirt/lib` after building the SDK, and installing from source
without building means populating `unirt/lib/` yourself from a Release's
native-libs archive before `pip install .`. `UNIRT_LIB_PATH` /
`UNIRT_PLUGIN_PATH` override the search.
