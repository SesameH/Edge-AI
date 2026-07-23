// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause
//
// UniRT public C API. The ABI declared here is frozen: struct layouts, enum
// values and symbol names must not change without bumping the SDK major
// version. See NOTICE for third-party dependency attribution.

#pragma once

/**
 * @file unirt.h
 * @brief UniRT — a runtime-agnostic C interface for on-device generation
 *        and embedding inference.
 *
 * The API is a thin, stable shell over interchangeable backend plugins
 * (llama.cpp, MLX, ...). A typical session:
 *
 *   1. unirt_init() once per process.
 *   2. Discover backends and devices: unirt_get_plugin_list(),
 *      unirt_get_device_list(), unirt_resolve_device().
 *   3. Create an LLM, VLM, or embedding model handle.
 *   4. unirt_llm_apply_chat_template() + unirt_llm_generate() per turn,
 *      streaming tokens through the caller's callback.
 *   5. Destroy handles, then unirt_deinit().
 *
 * Conventions used throughout:
 *   - Functions returning int32_t yield UNIRT_SUCCESS (0) or a negative
 *     unirt_ErrorCode.
 *   - Strings are null-terminated UTF-8.
 *   - Any pointer documented as "caller must free" is released with
 *     unirt_free(); everything else is owned by the library or plugin.
 *   - Handles are opaque and single-threaded: never drive one handle from
 *     two threads at once. Distinct handles are independent.
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef UNIRT_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef UNIRT_BUILD
#define UNIRT_API __declspec(dllexport)
#else
#define UNIRT_API __declspec(dllimport)
#endif
#else
#define UNIRT_API __attribute__((visibility("default")))
#endif
#else
#define UNIRT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/*  Error codes                                                               */
/* ========================================================================== */

/**
 * Status codes returned by every fallible API call. Zero is success; all
 * failures are negative. Codes are grouped in bands of one hundred so a
 * binding can classify by `code / 100`:
 *   -10xx runtime & arguments, -11xx hub & network, -12xx model/plugin
 *   loading, -13xx LLM, -14xx VLM, -15xx embedding.
 */
typedef enum {
    UNIRT_SUCCESS = 0, /**< No error */

    /* -- Runtime, lifecycle, arguments (-10xx) -- */

    UNIRT_ERROR_COMMON_UNKNOWN             = -1000, /**< Unclassified failure */
    UNIRT_ERROR_COMMON_INVALID_INPUT       = -1001, /**< NULL/malformed argument or bad handle */
    UNIRT_ERROR_COMMON_INVALID_DEVICE      = -1002, /**< Device alias not one of cpu/gpu/npu/hybrid */
    UNIRT_ERROR_COMMON_MEMORY_ALLOCATION   = -1003, /**< Out of memory */
    UNIRT_ERROR_COMMON_FILE_NOT_FOUND      = -1004, /**< Missing or unreadable file */
    UNIRT_ERROR_COMMON_NOT_INITIALIZED     = -1005, /**< unirt_init() has not been called */
    UNIRT_ERROR_COMMON_ALREADY_INITIALIZED = -1006, /**< Second unirt_init() without unirt_deinit() */
    UNIRT_ERROR_COMMON_CANCELLED           = -1007, /**< Caller aborted the operation */
    UNIRT_ERROR_COMMON_NOT_SUPPORTED       = -1008, /**< Feature absent from this build or plugin */
    UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED = -1009, /**< Plugin rejects this particular parameter */
    UNIRT_ERROR_COMMON_BUSY                = -1010, /**< Model handles still open; close them first */

    /* -- Model hub & network (-11xx) -- */

    UNIRT_ERROR_COMMON_NETWORK             = -1100, /**< Transport-level failure (timeout, DNS, proxy, ...) */
    UNIRT_ERROR_COMMON_AUTH                = -1101, /**< Hub demands credentials (HTTP 401/403) */
    UNIRT_ERROR_COMMON_HUB_MODEL_NOT_FOUND = -1102, /**< No such model on the remote hub (HTTP 404) */
    UNIRT_ERROR_COMMON_RATE_LIMITED        = -1103, /**< Hub throttled the request (HTTP 429) */
    UNIRT_ERROR_COMMON_HUB_SERVER          = -1104, /**< Hub-side failure (HTTP 5xx) */
    UNIRT_ERROR_COMMON_MANIFEST_PARSE      = -1105, /**< Manifest / index document did not parse */

    /* -- Model & plugin loading (-12xx) -- */

    UNIRT_ERROR_COMMON_MODEL_LOAD     = -1200, /**< Model failed to load */
    UNIRT_ERROR_COMMON_MODEL_INVALID  = -1201, /**< File is not a model this plugin understands */
    UNIRT_ERROR_COMMON_PLUGIN_LOAD    = -1210, /**< Plugin library failed to load */
    UNIRT_ERROR_COMMON_PLUGIN_INVALID = -1211, /**< Library lacks the plugin entry points / ABI */

    /* -- LLM: tokenization and text generation (-13xx) -- */

    UNIRT_ERROR_LLM_TOKENIZATION_FAILED         = -1300, /**< Tokenizer could not encode the input */
    UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH = -1301, /**< Tokenized input exceeds the context window */
    UNIRT_ERROR_LLM_GENERATION_FAILED           = -1310, /**< Decode loop failed */
    UNIRT_ERROR_LLM_GENERATION_PROMPT_TOO_LONG  = -1311, /**< Prompt alone overflows the context window */

    /* -- VLM: multimodal inputs and generation (-14xx) -- */

    UNIRT_ERROR_VLM_IMAGE_LOAD        = -1400, /**< Image file could not be read/decoded */
    UNIRT_ERROR_VLM_IMAGE_FORMAT      = -1401, /**< Image encoding not supported */
    UNIRT_ERROR_VLM_AUDIO_LOAD        = -1410, /**< Audio file could not be read/decoded */
    UNIRT_ERROR_VLM_AUDIO_FORMAT      = -1411, /**< Audio encoding not supported */
    UNIRT_ERROR_VLM_GENERATION_FAILED = -1420, /**< Multimodal decode loop failed */

    /* -- Embedding: encoder inference and output validation (-15xx) -- */

    UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED = -1500, /**< Encoder graph execution failed */
    UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID   = -1501, /**< Model output cannot be pooled as embeddings */

} unirt_ErrorCode;

/**
 * Map an error code to a short human-readable description.
 * Returns a static string; never NULL, never freed by the caller.
 * Thread-safe.
 */
UNIRT_API const char* unirt_get_error_message(const unirt_ErrorCode error_code);

/**
 * Detail text for the most recent failure the runtime recorded on the
 * calling thread — best effort, meant for diagnostics, not control flow.
 * Empty string when the last call on this thread succeeded or failed
 * without leaving detail. The pointer stays valid until the next UniRT
 * call on the same thread; never NULL, never freed by the caller.
 * Thread-safe (thread-local storage).
 */
UNIRT_API const char* unirt_last_error_message(void);

/* ========================================================================== */
/*  Shared scalar types and callbacks                                         */
/* ========================================================================== */

/** A plugin identifier such as "llama_cpp", "mlx", or "onnxruntime" (plain UTF-8 pointer).
 *  Valid ids are exactly those reported by unirt_get_plugin_list(). */
typedef const char* unirt_PluginId;

/** A filesystem path (plain UTF-8 pointer). */
typedef const char* unirt_Path;

/** Severity levels for the logging callback, ordered least to most severe. */
typedef enum {
    UNIRT_LOG_LEVEL_TRACE, /* Finest-grained tracing */
    UNIRT_LOG_LEVEL_DEBUG, /* Developer diagnostics */
    UNIRT_LOG_LEVEL_INFO,  /* Normal operational messages */
    UNIRT_LOG_LEVEL_WARN,  /* Recoverable problems */
    UNIRT_LOG_LEVEL_ERROR  /* Failures */
} unirt_LogLevel;

/** Receives one log line at a time. Install with unirt_set_log(). */
typedef void (*unirt_log_callback)(unirt_LogLevel, const char*);

/**
 * Streaming-generation callback. Invoked once per detokenized text piece
 * (always a complete UTF-8 sequence — the bridge re-joins code points split
 * across token boundaries). `user_data` is the pointer supplied on the
 * generate input. Return true to keep generating, false to stop early
 * (the run then finishes with stop_reason "user").
 */
typedef bool (*unirt_token_callback)(const char* token, void* user_data);

/* ========================================================================== */
/*  Runtime lifecycle                                                         */
/* ========================================================================== */

/** The C plugin table, defined in plugin/plugin_abi.h. Only plugins and
 *  embedders that statically link a backend ever name this type. */
typedef struct unirt_PluginTable unirt_PluginTable;

/** The two entry points a plugin exports, as function pointers, for
 *  platforms where shared libraries cannot be scanned (iOS forbids dlopen
 *  of arbitrary paths; some embedders prefer one static binary). */
typedef unirt_PluginId (*unirt_plugin_id_func)(void);
typedef unirt_PluginTable* (*unirt_plugin_open_func)(void);

/**
 * @brief Register a statically linked backend, bypassing the shared-library
 *        scan. May be called before or after unirt_init(); the backend joins
 *        any dynamically discovered plugins under the id it reports.
 *
 * @param identity[in]:    The plugin's unirt_plugin_id function.
 * @param open_plugin[in]: The plugin's unirt_plugin_open function.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe.
 */
UNIRT_API int32_t unirt_register_plugin(
    unirt_plugin_id_func identity, unirt_plugin_open_func open_plugin);

/**
 * @brief Bring up the runtime: scan the plugin directory and prepare the
 *        registry. Call once before any other API except unirt_set_log().
 *
 * @return UNIRT_SUCCESS, or UNIRT_ERROR_COMMON_ALREADY_INITIALIZED on a
 *         repeated call.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_init(void);

/**
 * @brief Tear the runtime down and unload plugins.
 *
 * Fails with UNIRT_ERROR_COMMON_BUSY while any model handle is still
 * alive — destroy handles first so plugin code is never unmapped under a
 * live object.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_deinit(void);

/**
 * @brief Route library logging to a custom sink. Install before unirt_init()
 *        to capture startup messages; passing NULL restores the default
 *        stderr sink.
 *
 * @param callback[in]: The sink to install.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe.
 */
UNIRT_API int32_t unirt_set_log(unirt_log_callback callback);

/**
 * @brief Release memory that a UniRT call handed to the caller (any pointer
 *        documented as "caller must free"). NULL is a no-op.
 *
 * @param ptr[in]: The pointer to release.
 *
 * @thread_safety: Thread-safe for distinct pointers.
 */
UNIRT_API void unirt_free(void* ptr);

/**
 * @brief The SDK's own version string.
 *
 * @return Static null-terminated UTF-8 string; do not free.
 *
 * @thread_safety: Thread-safe.
 */
UNIRT_API const char* unirt_version(void);

/* ========================================================================== */
/*  Backend and device discovery                                              */
/* ========================================================================== */

/**
 * @brief Version string of a registered plugin (e.g. the llama.cpp build
 *        commit, MLX release, or ONNX Runtime release). Plugins own
 *        their versions; the SDK core does not link any backend runtime.
 *
 * @param plugin_id[in]: Backend id; required.
 *
 * @return Null-terminated UTF-8 string owned by the plugin, or NULL when
 *         `plugin_id` is NULL or unknown.
 *
 * @thread_safety: Thread-safe.
 */
UNIRT_API const char* unirt_get_plugin_version(unirt_PluginId plugin_id);

/** Result of unirt_get_plugin_list(). */
typedef struct {
    unirt_PluginId* plugin_ids;   /**< Array of plugin ids (UTF-8) (caller must free with unirt_free) */
    int32_t          plugin_count; /**< Number of entries in plugin_ids */
} unirt_GetPluginListOutput;

/**
 * @brief Enumerate the plugins the runtime discovered (or that were
 *        registered in-process).
 *
 * @param output[out] Receives the id array and count; free the array with
 *                    unirt_free().
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_get_plugin_list(unirt_GetPluginListOutput* output);

/** Bit flags describing which model kinds a plugin can host. */
#define UNIRT_MODALITY_LLM 0x1u       /**< Text generation */
#define UNIRT_MODALITY_VLM 0x2u       /**< Multimodal generation */
#define UNIRT_MODALITY_EMBEDDING 0x4u /**< Embedding encoding */

/**
 * @brief Which model kinds a plugin supports, as UNIRT_MODALITY_* bits,
 *        without instantiating any model. Zero means the plugin did not
 *        declare its capabilities (older plugin binaries).
 *
 * @param plugin_id[in]:        Backend id; required.
 * @param out_modalities[out]:  Receives the bitmask; zeroed on failure.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe.
 */
UNIRT_API int32_t unirt_get_plugin_modalities(unirt_PluginId plugin_id, uint32_t* out_modalities);

/** Selects the plugin whose devices to enumerate. */
typedef struct {
    unirt_PluginId plugin_id; /**< Backend id */
} unirt_GetDeviceListInput;

/** Result of unirt_get_device_list(): parallel id/name arrays. */
typedef struct {
    // ids are plugin-native, e.g. "Metal", "CPU", "HTP0"
    const char** device_ids;   /**< Array of device ids (caller must free with unirt_free when not null) */
    const char** device_names; /**< Array of display names (caller must free with unirt_free when not null) */
    int32_t      device_count; /**< Number of entries in each array */
} unirt_GetDeviceListOutput;

/**
 * @brief Enumerate the compute devices a plugin can run on. device_ids[i]
 *        is the string to pass as `device_id` at model-create time;
 *        device_names[i] is its human-readable label.
 *
 * @param input[in]   Which plugin to ask.
 * @param output[out] Receives both arrays and the count.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_get_device_list(const unirt_GetDeviceListInput* input, unirt_GetDeviceListOutput* output);

/**
 * Input for unirt_resolve_device().
 *
 * `mode` is the user-facing alias: "cpu", "gpu", "npu", "hybrid", or the
 * "pick the plugin default" spellings — NULL, "" and "auto". Matching is
 * case-insensitive and ignores surrounding whitespace.
 *
 * `ngl_default` is the caller's preferred `n_gpu_layers`. It flows through
 * unchanged for gpu / npu / hybrid on llama_cpp-style plugins; negative
 * means "offload all layers", so callers express both "everything" and
 * "unset" as -1. `cpu` forces it to 0, as do NPU-only plugins.
 */
typedef struct {
    unirt_PluginId plugin_id;   /**< Which backend to consult (required) */
    const char*     mode;        /**< User-facing alias; NULL / "" / "auto" → plugin default */
    int32_t         ngl_default; /**< Fallback n_gpu_layers when the alias implies none */
} unirt_ResolveDeviceInput;

/**
 * Output of unirt_resolve_device().
 *
 * `device_id` is the concrete plugin-native string to place on the create
 * input (e.g. "HTP0", "GPUOpenCL", "NPU"). NULL means "leave device_id
 * unset and let the plugin choose" (how the cpu and hybrid aliases resolve
 * for llama_cpp). Copy it before freeing the output.
 *
 * `ngl` is the resolved `n_gpu_layers`: 0 for `cpu`; otherwise
 * `ngl_default` passed through (negative still means "all layers").
 *
 * `warning` is set when the alias had to be coerced — say, a plugin that
 * only exposes an NPU maps cpu/gpu/hybrid onto it. Coercion is not an
 * error: surface the warning and proceed.
 *
 * Both `device_id` and `warning` are heap-allocated; unirt_free() each
 * non-NULL pointer.
 */
typedef struct {
    char*   device_id; /**< Chosen device id, or NULL for the backend default; release with unirt_free */
    int32_t ngl;       /**< Effective n_gpu_layers after resolution */
    char*   warning;   /**< Present when the alias had to be coerced; release with unirt_free */
} unirt_ResolveDeviceOutput;

/**
 * @brief Translate a user-facing device alias (cpu / gpu / npu / hybrid /
 *        auto) into the concrete (device_id, n_gpu_layers) pair a plugin
 *        expects. This function is the single alias table for the whole
 *        SDK — language bindings call it rather than duplicating the
 *        mapping.
 *
 * @param input[in]   Non-NULL input struct.
 * @param output[out] Non-NULL output struct; on success its string fields
 *                    may be heap-allocated (free with unirt_free).
 *
 * @return UNIRT_SUCCESS (including coerced aliases, which populate
 *         `warning`); UNIRT_ERROR_COMMON_INVALID_INPUT when input / output /
 *         plugin_id is NULL; UNIRT_ERROR_COMMON_INVALID_DEVICE when `mode`
 *         is a non-empty, non-"auto" string outside the alias table.
 *
 * @thread_safety: Thread-safe; touches no shared state.
 */
UNIRT_API int32_t unirt_resolve_device(const unirt_ResolveDeviceInput* input, unirt_ResolveDeviceOutput* output);

/* ========================================================================== */
/*  Generation parameters shared by LLM and VLM                               */
/* ========================================================================== */

/** Per-run performance metrics, filled into every generate output. */
typedef struct {
    int64_t ttft;        /* Time to first token (us) */
    int64_t prompt_time; /* Prefill wall time (us) */
    int64_t decode_time; /* Decode wall time (us) */

    int64_t prompt_tokens;    /* Tokens consumed from the prompt */
    int64_t generated_tokens; /* Tokens produced */
    int64_t audio_duration;   /* Duration of audio inputs (us); 0 when none */

    double prefill_speed;    /* Prefill throughput (tokens/sec) */
    double decoding_speed;   /* Decode throughput (tokens/sec) */
    double real_time_factor; /* RTF for audio (1.0 = real time; >1 = faster) */

    const char* stop_reason; /* "eos", "length", "user", "stop_sequence", "context_length" */
} unirt_ProfileData;

/**
 * Sampling controls. Zero-initialize, then set what you need; a temperature
 * of 0 (or below) selects greedy decoding. At most one of grammar_path /
 * grammar_string / enable_json / json_schema may be active — they all
 * constrain output through the same grammar slot.
 */
typedef struct {
    float       temperature;        /* Softmax temperature; <=0 = greedy */
    float       top_p;              /* Nucleus sampling mass (0.0-1.0) */
    int32_t     top_k;              /* Keep only the k most likely tokens; 0 = off */
    float       min_p;              /* Drop tokens below this fraction of the top probability */
    float       repetition_penalty; /* Multiplicative penalty on already-seen tokens */
    float       presence_penalty;   /* Flat penalty once a token has appeared */
    float       frequency_penalty;  /* Penalty scaled by how often a token appeared */
    int32_t     seed;               /* RNG seed; -1 = nondeterministic */
    unirt_Path grammar_path;       /* Constrain output with a grammar file (optional) */
    const char* grammar_string;     /* Constrain output with an inline grammar (optional) */
    bool        enable_json;        /* Constrain output to JSON */
    /* Constrain output to a JSON Schema (UTF-8 JSON text, optional). The
     * plugin compiles it to a grammar; generated text is guaranteed to
     * parse and to validate against the schema's supported subset.
     * Appended for 0.2: native libraries and bindings ship in lock-step
     * (wheel/AAR/XCFramework bundle both sides), so tail growth is safe. */
    const char* json_schema;
} unirt_SamplerConfig;

/**
 * Per-request generation settings, shared by unirt_llm_generate() and
 * unirt_vlm_generate(). Zero-initialized defaults are sensible; the struct
 * is read for the duration of the call only.
 */
typedef struct {
    int32_t               max_tokens;     /* Generation budget; <=0 = plugin default */
    const char**          stop;           /* Stop sequences (checked against decoded text) */
    int32_t               stop_count;     /* Number of stop sequences */
    /* How much cached KV state to keep before prefilling prompt_utf8:
     * 0 = automatic — the plugin reuses the longest prefix shared with the
     * previous transcript (multi-turn chat gets fast TTFT for free);
     * > 0 = explicit rewind to exactly this many retained tokens. Values
     * beyond the cached history are invalid. The bundled llama_cpp VLM
     * accepts only 0: retained multimodal position state cannot be
     * expressed safely through this ABI. */
    int32_t               n_past;
    unirt_SamplerConfig* sampler_config; /* Sampling controls; NULL = plugin defaults */
    // --- Multimodal inputs (VLM-capable plugins only) ---
    unirt_Path* image_paths;      /* Image files referenced by the prompt (NULL if none) */
    int32_t      image_count;      /* Number of images */
    int32_t      image_max_length; /* Cap on the longest image edge; 0 = no resize */
    unirt_Path* audio_paths;      /* Audio files referenced by the prompt (NULL if none) */
    int32_t      audio_count;      /* Number of audio files */
    // --- Behaviour when the context window fills up ---
    /* When enabled, supporting plugins evict the oldest tokens (keeping
     * sliding_window_n_keep anchored at the head) instead of failing with a
     * context-length error. Plugins that cannot shift their context reject
     * the option. */
    bool    sliding_window;
    int32_t sliding_window_n_keep; /* Anchored head tokens when sliding (0 = plugin default of 4) */
} unirt_GenerationConfig;

/**
 * Model-load settings, embedded by value in the LLM/VLM create inputs.
 * Zero means "plugin default" for every numeric field; pointer fields are
 * optional and may be NULL.
 */
typedef struct {
    int32_t n_ctx;            // context window in tokens; 0 = read from the model
    int32_t n_threads;        // decode threads
    int32_t n_threads_batch;  // prefill/batch threads
    int32_t n_batch;          // logical batch ceiling per decode submission
    int32_t n_ubatch;         // physical micro-batch ceiling
    int32_t n_seq_max;        // max concurrent sequences (recurrent-model states)
    int32_t n_gpu_layers;     // layers offloaded to the device; 0 = all on CPU

    // Prompt-formatting extras.
    unirt_Path chat_template_path;     // external file overriding the model's chat template (optional)
    const char* chat_template_content;  // inline chat template override (optional)
    const char* grammar_str;            // default grammar applied to every request
} unirt_ModelConfig;

/* ========================================================================== */
/*  LLM — text-only language models                                           */
/* ========================================================================== */

typedef struct unirt_LLM unirt_LLM; /* Opaque LLM handle */

/* --------------------  Lifecycle  ---------------------------------------- */

typedef struct {
    unirt_Path        model_path;     /** Model weights (file or directory, plugin-dependent) */
    unirt_Path        tokenizer_path; /** Tokenizer file; NULL when bundled with the model */
    unirt_ModelConfig config;         /** Load-time settings */
    unirt_PluginId    plugin_id;      /** Backend to load the model with */
    const char*        device_id;      /** Concrete device id; NULL = plugin default */
} unirt_LlmCreateInput;

/**
 * @brief Load a language model and return a live handle.
 *
 * @param input[in]:      Model location, backend and load settings.
 * @param out_handle[out]: Receives the handle; release with unirt_llm_destroy().
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_llm_create(const unirt_LlmCreateInput* input, unirt_LLM** out_handle);

/**
 * @brief Unload the model and release everything owned by the handle.
 *
 * @param handle[in]: Text-model handle being torn down.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_llm_destroy(unirt_LLM* handle);

/**
 * @brief Drop all conversational state — KV cache, cached transcript,
 *        sampler state — as if the model were freshly loaded.
 *
 * @param handle[in]: Text-model handle whose state is dropped.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_llm_reset(unirt_LLM* handle);

/* --------------------  KV-cache persistence  ------------------------------ */

/** Where to write the KV-cache snapshot. */
typedef struct {
    unirt_Path path; /** Destination file */
} unirt_KvCacheSaveInput;

/** Reserved result struct for KV-cache save; pass NULL today. */
typedef struct {
    void* reserved; /** Reserved for future use, safe to set as NULL */
} unirt_KvCacheSaveOutput;

/** Where to read a KV-cache snapshot from. */
typedef struct {
    unirt_Path path; /** Source file */
} unirt_KvCacheLoadInput;

/** Reserved result struct for KV-cache load; pass NULL today. */
typedef struct {
    void* reserved; /** Reserved for future use, safe to set as NULL */
} unirt_KvCacheLoadOutput;

/**
 * @brief Snapshot the current KV cache to disk, so a later
 *        unirt_llm_load_kv_cache() can resume the conversation without
 *        re-prefilling.
 *
 * @param handle[in]:  LLM handle
 * @param input[in]:   Destination path
 * @param output[out]: Reserved; NULL is fine
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 */
UNIRT_API int32_t unirt_llm_save_kv_cache(
    unirt_LLM* handle, const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput* output);

/**
 * @brief Restore a KV cache previously written by unirt_llm_save_kv_cache().
 *        The snapshot must come from the same model.
 *
 * @param handle[in]:  LLM handle
 * @param input[in]:   Source path
 * @param output[out]: Reserved; NULL is fine
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 */
UNIRT_API int32_t unirt_llm_load_kv_cache(
    unirt_LLM* handle, const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput* output);

/* --------------------  Chat templating  ----------------------------------- */

/** One turn of a text-only conversation. */
typedef struct {
    const char* role;    /* "user", "assistant" or "system" */
    const char* content; /* Turn text, UTF-8 */
} unirt_LlmChatMessage;

/** Conversation to render into the model's prompt format. */
typedef struct {
    unirt_LlmChatMessage* messages;              /** Turns, oldest first */
    int32_t                message_count;         /** Number of turns */
    const char*            tools;                 /** Tool definitions as JSON (optional, may be NULL) */
    bool                   enable_thinking;       /** Ask the template for reasoning markup */
    bool                   add_generation_prompt; /** Append the assistant-turn opener */
} unirt_LlmApplyChatTemplateInput;

/** Rendered prompt text. */
typedef struct {
    char* formatted_text; /** Prompt string (caller must free with unirt_free) */
} unirt_LlmApplyChatTemplateOutput;

/**
 * @brief Render a conversation through the model's chat template (or the
 *        override supplied at load time), producing the prompt string to
 *        pass to unirt_llm_generate().
 *
 * @param handle[in]:  LLM handle
 * @param input[in]:   Conversation and template flags
 * @param output[out]: Rendered prompt (free with unirt_free)
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 */
UNIRT_API int32_t unirt_llm_apply_chat_template(
    unirt_LLM* handle, const unirt_LlmApplyChatTemplateInput* input, unirt_LlmApplyChatTemplateOutput* output);

/* --------------------  Generation  ---------------------------------------- */

/** Input for one streaming generation run. */
typedef struct {
    const char*                    prompt_utf8; /** Full rendered prompt (the whole transcript) */
    const unirt_GenerationConfig* config;      /** Per-request settings; NULL = defaults */
    unirt_token_callback          on_token;    /** Streaming callback; NULL for blocking-only use */
    void*                          user_data;   /** Opaque pointer echoed to on_token */

    /** Exactly one input form is used per call:
     *  - input_ids non-NULL with input_ids_count > 0 → the ids are fed
     *    directly and prompt_utf8 is ignored.
     *  - otherwise prompt_utf8 must be non-NULL.
     *  - neither → UNIRT_ERROR_COMMON_INVALID_INPUT.
     *
     *  With input_ids the caller owns special-token placement (BOS/EOS);
     *  nothing is inserted automatically. */
    const int32_t* input_ids;       /** Pre-tokenized prompt (optional, may be NULL) */
    int32_t        input_ids_count; /** Number of ids in input_ids */
} unirt_LlmGenerateInput;

/** Result of one generation run. */
typedef struct {
    char*              full_text;    /** Concatenated output text (caller must free with unirt_free) */
    unirt_ProfileData profile_data; /** Timing and token counts for this run */
} unirt_LlmGenerateOutput;

/**
 * @brief Run one generation: prefill the prompt (reusing cached KV where the
 *        transcript prefix matches), then decode until EOS, the token
 *        budget, a stop sequence, or the callback returns false. Tokens
 *        stream through on_token as they decode; the complete text and
 *        profile land in `output`.
 *
 * @param handle[in]:  LLM handle
 * @param input[in]:   Prompt and per-request settings
 * @param output[out]: Full text and metrics
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 */
UNIRT_API int32_t unirt_llm_generate(
    unirt_LLM* handle, const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output);

/* --------------------  Introspection  ------------------------------------- */

/** Static vocabulary facts about the loaded model. The bridge zeroes the
 *  struct before the plugin fills it, so unreported fields read as 0. */
typedef struct {
    int32_t vocab_size; /** Vocabulary size (>=1 when reported). */
    int32_t bos_token;  /** BOS token id, or -1 if the model has none. */
    int32_t add_bos;    /** 1 = prepend BOS when feeding raw input_ids. */
    int32_t reserved0;  /** Reserved, must be 0. */
} unirt_LlmModelInfo;

/**
 * @brief Vocabulary metadata for callers that build raw input_ids
 *        themselves (benchmarks, custom tokenizers) and need vocab_size /
 *        BOS handling without a tokenizer round-trip. Cheap and
 *        side-effect-free.
 *
 * @param handle[in]: Open text-model handle.
 * @param output[out]: Zeroed, then populated by the plugin.
 *
 * @return unirt_ErrorCode:
 *   - UNIRT_SUCCESS                          when the call completes.
 *   - UNIRT_ERROR_COMMON_NOT_INITIALIZED     for a NULL handle.
 *   - UNIRT_ERROR_COMMON_INVALID_INPUT       for a missing output pointer.
 *   - UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED when the plugin cannot report
 *     this metadata; callers that require vocab_size must treat it as fatal.
 */
UNIRT_API int32_t unirt_llm_get_model_info(unirt_LLM* handle, unirt_LlmModelInfo* output);

/** Memory footprint of the loaded model. Byte fields are -1 when the plugin
 *  cannot measure them; the bridge zero-initializes the rest. */
typedef struct {
    int64_t model_bytes;       /** Bytes held by the weights. */
    int64_t kv_cache_bytes;    /** Bytes held by KV cache / decode state. */
    int64_t device_peak_bytes; /** Peak device (GPU/NPU) allocation since load. */
    int64_t process_rss_bytes; /** Whole-process resident set; filled by the bridge. */
    const char* device_name;   /** Active compute device; static string owned by the plugin. */
} unirt_LlmRuntimeStats;

/**
 * @brief Memory usage of the loaded model — weights, KV cache, device peak
 *        and process RSS. Cheap; fine to call between generations. Note
 *        `process_rss_bytes` spans the whole process, so it is not
 *        per-model when several models are loaded.
 *
 * @return UNIRT_SUCCESS, or UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED when the
 *         plugin can report nothing at all.
 */
UNIRT_API int32_t unirt_llm_get_runtime_stats(unirt_LLM* handle, unirt_LlmRuntimeStats* output);

/* ========================================================================== */
/*  VLM — multimodal (vision / audio) language models                         */
/* ========================================================================== */

/** One piece of a multimodal message. */
typedef struct {
    const char* type;  // "text", "image", "audio", ... (null-terminated UTF-8)
    const char* text;  // the payload: text content, a path/URL, or a special token
} unirt_VlmContent;

/** One turn of a multimodal conversation. */
typedef struct {
    const char*        role;           // "user", "assistant", "system", ...
    unirt_VlmContent* contents;       // pieces of this turn (may be NULL)
    int64_t            content_count;  // number of entries in `contents`
} unirt_VlmChatMessage;

typedef struct unirt_VLM unirt_VLM; /* Opaque VLM handle */

/* --------------------  Lifecycle  ----------------------------------------- */

typedef struct {
    unirt_Path        model_path;     /** Language-model weights */
    unirt_Path        mmproj_path;    /** Multimodal projector weights */
    unirt_ModelConfig config;         /** Load-time settings */
    unirt_PluginId    plugin_id;      /** Backend to load the model with */
    const char*        device_id;      /** Concrete device id; NULL = plugin default */
    unirt_Path        tokenizer_path; /** Tokenizer file; NULL when bundled with the model */
} unirt_VlmCreateInput;

/**
 * @brief Load a multimodal model (language weights + projector) and return
 *        a live handle.
 *
 * @param input[in]:      Model locations, backend and load settings.
 * @param out_handle[out]: Receives the handle; release with unirt_vlm_destroy().
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_vlm_create(const unirt_VlmCreateInput* input, unirt_VLM** out_handle);

/**
 * @brief Unload the model and release everything owned by the handle.
 *
 * @param handle[in]: Multimodal handle being torn down.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_vlm_destroy(unirt_VLM* handle);

/**
 * @brief Drop all conversational state (KV cache, sampler state) as if the
 *        model were freshly loaded.
 *
 * @param handle[in]: Multimodal handle whose state is dropped.
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
UNIRT_API int32_t unirt_vlm_reset(unirt_VLM* handle);

/* --------------------  Chat templating  ----------------------------------- */

/** Multimodal conversation to render into the model's prompt format. */
typedef struct {
    unirt_VlmChatMessage* messages;        /** Turns, oldest first */
    int32_t                message_count;   /** Number of turns */
    const char*            tools;           /** Tool definitions as JSON (optional, may be NULL) */
    bool                   enable_thinking; /** Ask the template for reasoning markup */

    // deepseek-ocr
    bool grounding; /** Insert the grounding token (OCR models) */
} unirt_VlmApplyChatTemplateInput;

/** Rendered prompt text. */
typedef struct {
    char* formatted_text; /** Prompt string (caller must free with unirt_free) */
} unirt_VlmApplyChatTemplateOutput;

/**
 * @brief Render a multimodal conversation through the model's chat template,
 *        producing the prompt string (with media placeholders) to pass to
 *        unirt_vlm_generate().
 *
 * @param handle[in]: Open multimodal handle
 * @param input[in]:   Conversation and template flags
 * @param output[out]: Rendered prompt (free with unirt_free)
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 */
UNIRT_API int32_t unirt_vlm_apply_chat_template(
    unirt_VLM* handle, const unirt_VlmApplyChatTemplateInput* input, unirt_VlmApplyChatTemplateOutput* output);

/* --------------------  Capability query  ----------------------------------- */

/** Input modalities the loaded projector supports. */
typedef struct {
    bool supports_vision; /** Model accepts image inputs */
    bool supports_audio;  /** Model accepts audio inputs */
} unirt_VlmCapabilities;

/**
 * @brief Which media the loaded model can actually consume. Plugins without
 *        a modality probe report both flags false; the bundled llama_cpp
 *        plugin reflects the loaded mmproj's capabilities, and MLX is
 *        text-only today.
 *
 * @param handle[in]: Open multimodal handle
 * @param output[out]: Capability flags
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 */
UNIRT_API int32_t unirt_vlm_get_capabilities(unirt_VLM* handle, unirt_VlmCapabilities* output);

/* --------------------  Generation  ----------------------------------------- */

/** Input for one streaming multimodal generation run. Media files travel in
 *  unirt_GenerationConfig (image_paths / audio_paths). */
typedef struct {
    const char*                    prompt_utf8; /** Full rendered prompt (the whole transcript) */
    const unirt_GenerationConfig* config;      /** Per-request settings; NULL = defaults */
    unirt_token_callback          on_token;    /** Streaming callback; NULL for blocking-only use */
    void*                          user_data;   /** Opaque pointer echoed to on_token */
} unirt_VlmGenerateInput;

/** Result of one multimodal generation run. */
typedef struct {
    char*              full_text;    /** Concatenated output text (caller must free with unirt_free) */
    unirt_ProfileData profile_data; /** Timing and token counts for this run */
} unirt_VlmGenerateOutput;

/**
 * @brief Run one multimodal generation: encode the media referenced by the
 *        config, prefill the prompt, then decode with streaming callbacks —
 *        the multimodal counterpart of unirt_llm_generate().
 *
 * @param handle[in]: Open multimodal handle
 * @param input[in]:   Prompt and per-request settings
 * @param output[out]: Full text and metrics
 *
 * @return UNIRT_SUCCESS on success, negative on failure.
 */
UNIRT_API int32_t unirt_vlm_generate(
    unirt_VLM* handle, const unirt_VlmGenerateInput* input, unirt_VlmGenerateOutput* output);

/** Memory footprint of the loaded model. Byte fields are -1 when the plugin
 *  cannot measure them; the bridge zero-initializes the rest. Same shape as
 *  unirt_LlmRuntimeStats — kept as its own type since VLM's model_bytes may
 *  need to cover the projector/encoder too, not just the text weights. */
typedef struct {
    int64_t model_bytes;       /** Bytes held by the weights (+ projector, if any). */
    int64_t kv_cache_bytes;    /** Bytes held by KV cache / decode state. */
    int64_t device_peak_bytes; /** Peak device (GPU/NPU) allocation since load. */
    int64_t process_rss_bytes; /** Whole-process resident set; filled by the bridge. */
    const char* device_name;   /** Active compute device; static string owned by the plugin. */
} unirt_VlmRuntimeStats;

/**
 * @brief Memory usage of the loaded multimodal model — the VLM counterpart
 *        of unirt_llm_get_runtime_stats(). Cheap; fine to call between
 *        generations.
 *
 * @return UNIRT_SUCCESS, or UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED when the
 *         plugin can report nothing at all.
 */
UNIRT_API int32_t unirt_vlm_get_runtime_stats(unirt_VLM* handle, unirt_VlmRuntimeStats* output);

/* ========================================================================== */
/*  Embedding — pre-tokenized encoder models                                 */
/* ========================================================================== */

typedef struct unirt_Embedding unirt_Embedding; /* Opaque embedding handle */

/** Pooling applied when the selected ONNX output is token-level [B,S,H].
 *  A sentence-level [B,H] output bypasses pooling. */
typedef enum {
    UNIRT_EMBEDDING_POOLING_MODEL_DEFAULT = 0, /**< Mean for token-level output */
    UNIRT_EMBEDDING_POOLING_CLS           = 1, /**< Token at sequence index zero */
    UNIRT_EMBEDDING_POOLING_MEAN          = 2, /**< Attention-mask weighted mean */
    UNIRT_EMBEDDING_POOLING_LAST_TOKEN    = 3  /**< Last unmasked token */
} unirt_EmbeddingPooling;

/** Load-time settings for an embedding encoder. Tokenization deliberately
 *  stays outside the plugin so WordPiece, BPE and SentencePiece callers can
 *  all use the same native inference ABI. */
typedef struct {
    unirt_Path              model_path;  /**< ONNX model file */
    unirt_PluginId          plugin_id;   /**< Backend, normally "onnxruntime" */
    const char*             device_id;   /**< Concrete device id; NULL = plugin default */
    unirt_EmbeddingPooling pooling;     /**< Pooling for token-level output */
    bool                    normalize;   /**< L2-normalize each result vector */
    const char*             output_name; /**< Graph output to use; NULL = automatic */
} unirt_EmbeddingCreateInput;

/** One rectangular token batch. All arrays are row-major [batch_size,
 *  sequence_length] and borrowed for the call. attention_mask may be NULL
 *  (all tokens visible). token_type_ids may be NULL (all zeros when the
 *  model declares that input). */
typedef struct {
    const int64_t* input_ids;
    const int64_t* attention_mask;
    const int64_t* token_type_ids;
    int32_t        batch_size;
    int32_t        sequence_length;
} unirt_EmbeddingEncodeInput;

/** Contiguous row-major [embedding_count, embedding_dimension] float32
 *  result. Release embeddings with unirt_free(). */
typedef struct {
    float*  embeddings;
    int32_t embedding_count;
    int32_t embedding_dimension;
} unirt_EmbeddingEncodeOutput;

/** Memory footprint of a loaded encoder. Unknown byte fields are -1. */
typedef struct {
    int64_t     model_bytes;
    int64_t     device_peak_bytes;
    int64_t     process_rss_bytes;
    const char* device_name; /**< Static string owned by the plugin */
} unirt_EmbeddingRuntimeStats;

UNIRT_API int32_t unirt_embedding_create(
    const unirt_EmbeddingCreateInput* input, unirt_Embedding** out_handle);
UNIRT_API int32_t unirt_embedding_destroy(unirt_Embedding* handle);
UNIRT_API int32_t unirt_embedding_encode(
    unirt_Embedding* handle, const unirt_EmbeddingEncodeInput* input,
    unirt_EmbeddingEncodeOutput* output);
UNIRT_API int32_t unirt_embedding_get_runtime_stats(
    unirt_Embedding* handle, unirt_EmbeddingRuntimeStats* output);

/** One query scored against N candidate documents. Unlike
 *  unirt_EmbeddingEncodeInput, this takes raw UTF-8 text rather than
 *  pre-tokenized ids: cross-encoder reranking needs the model's own
 *  tokenizer and (when present) a model-specific "rerank" prompt template
 *  to assemble the query+document pair correctly, neither of which the
 *  pre-tokenized ABI above can express. Supported only by backends whose
 *  loaded model has a classifier head (GGUF rerankers via llama_cpp);
 *  others report UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED. */
typedef struct {
    const char*        query_utf8;
    const char* const* documents_utf8;
    int32_t            document_count;
} unirt_EmbeddingRerankInput;

/** One relevance score per document, same order as the request. Release
 *  scores with unirt_free(). */
typedef struct {
    float*  scores;
    int32_t score_count;
} unirt_EmbeddingRerankOutput;

/**
 * @brief Score a query against each candidate document with the loaded
 *        model's cross-encoder/classifier head (llama.cpp's
 *        LLAMA_POOLING_TYPE_RANK) — higher is more relevant; the caller
 *        sorts.
 *
 * @return UNIRT_SUCCESS, UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED when the
 *         loaded model has no classifier head, or a common validation code.
 */
UNIRT_API int32_t unirt_embedding_rerank(
    unirt_Embedding* handle, const unirt_EmbeddingRerankInput* input,
    unirt_EmbeddingRerankOutput* output);

#ifdef __cplusplus
} /* extern "C" */
#endif
