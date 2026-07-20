// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * Shared machinery for the C-API bridge layer.
 *
 * Every exported unirt_* function follows the same discipline:
 *   1. validate caller-visible arguments without touching plugin code,
 *   2. resolve the opaque handle through the handle registry,
 *   3. forward to the plugin virtual call inside a uniform exception wall.
 *
 * The templates here implement steps 2 and 3 once, so an entrypoint is a
 * few lines of validation plus a lambda, and adding a new API cannot forget
 * the exception wall.
 */

#include <memory>
#include <utility>

#include "unirt.h"
#include "handle_registry.h"
#include "logging.h"
#include "profile.h"
#include "registry.h"
#include "utils.h"

namespace unirt::bridge {

/** Record detail for unirt_last_error_message() and pass the code through:
 *  `return fail(UNIRT_ERROR_..., "what went wrong");` */
inline int32_t fail(int32_t code, std::string message) noexcept {
    set_last_error(std::move(message));
    return code;
}

/** Run `body` behind the uniform exception wall. `failure` is the code for
 *  unclassified failures of this entrypoint. Clears the thread's failure
 *  detail on entry and records one for every escaped exception. */
template <typename Body>
int32_t shielded(const char* entrypoint, int32_t failure, Body&& body) noexcept {
    clear_last_error();
    try {
        return std::forward<Body>(body)();
    } catch (const PluginNotFoundException&) {
        UNIRT_LOG_ERROR("{}: plugin not found", entrypoint);
        return fail(
            UNIRT_ERROR_COMMON_PLUGIN_INVALID,
            std::string(entrypoint) + ": no plugin with this id is registered");
    } catch (const PluginLoadException&) {
        UNIRT_LOG_ERROR("{}: plugin failed to load", entrypoint);
        return fail(
            UNIRT_ERROR_COMMON_PLUGIN_LOAD,
            std::string(entrypoint) + ": the plugin exists but failed to load (see log)");
    } catch (const std::bad_alloc&) {
        UNIRT_LOG_ERROR("{}: out of memory", entrypoint);
        return fail(UNIRT_ERROR_COMMON_MEMORY_ALLOCATION, std::string(entrypoint) + ": out of memory");
    } catch (const std::exception& e) {
        UNIRT_LOG_ERROR("{}: {}", entrypoint, e.what());
        return fail(failure, std::string(entrypoint) + ": " + e.what());
    } catch (...) {
        UNIRT_LOG_ERROR("{}: non-standard exception", entrypoint);
        return fail(failure, std::string(entrypoint) + ": non-standard exception");
    }
}

/** Handle traits tie the opaque C handle type to its plugin interface and
 *  the handle-registry accessors, so the forwarding templates below work for
 *  both modalities. */
template <typename Interface>
struct HandleTraits;

template <>
struct HandleTraits<LlmBackend> {
    using CHandle = unirt_LLM;
    static LlmBackend* find(const CHandle* h) noexcept { return find_llm_handle(h); }
    static CHandle* track(LlmBackend* backend) { return register_llm_handle(backend); }
    static LlmBackend* untrack(CHandle* h) noexcept { return remove_llm_handle(h); }
};

template <>
struct HandleTraits<VlmBackend> {
    using CHandle = unirt_VLM;
    static VlmBackend* find(const CHandle* h) noexcept { return find_vlm_handle(h); }
    static CHandle* track(VlmBackend* backend) { return register_vlm_handle(backend); }
    static VlmBackend* untrack(CHandle* h) noexcept { return remove_vlm_handle(h); }
};

template <>
struct HandleTraits<EmbeddingBackend> {
    using CHandle = unirt_Embedding;
    static EmbeddingBackend* find(const CHandle* h) noexcept { return find_embedding_handle(h); }
    static CHandle* track(EmbeddingBackend* backend) { return register_embedding_handle(backend); }
    static EmbeddingBackend* untrack(CHandle* h) noexcept { return remove_embedding_handle(h); }
};

/** Resolve a handle and forward one member call:
 *  with_backend<LlmBackend>("llm.reset", h, UNKNOWN, [](LlmBackend& m) { return m.reset(); }) */
template <typename Interface, typename Call>
int32_t with_backend(
    const char* entrypoint, typename HandleTraits<Interface>::CHandle* handle, int32_t failure,
    Call&& call) noexcept {
    return shielded(entrypoint, failure, [&]() -> int32_t {
        if (!handle) {
            return fail(UNIRT_ERROR_COMMON_NOT_INITIALIZED, std::string(entrypoint) + ": handle is NULL");
        }
        Interface* backend = HandleTraits<Interface>::find(handle);
        if (!backend) {
            return fail(
                UNIRT_ERROR_COMMON_INVALID_INPUT,
                std::string(entrypoint) + ": handle is stale, destroyed, or of the wrong model kind");
        }
        return call(*backend);
    });
}

/** Construct a backend through the plugin directory, hand ownership to the
 *  handle registry on success. */
template <typename Interface, typename CreateInput>
int32_t open_backend(
    const char* entrypoint, const CreateInput* input,
    typename HandleTraits<Interface>::CHandle** out_handle) noexcept {
    return shielded(entrypoint, UNIRT_ERROR_COMMON_MODEL_LOAD, [&]() -> int32_t {
        std::unique_ptr<Interface> backend(
            PluginDirectory::instance().get<Interface>(input->plugin_id));
        if (!backend) {
            return fail(
                UNIRT_ERROR_COMMON_NOT_SUPPORTED,
                std::string(entrypoint) + ": plugin '" + input->plugin_id +
                    "' does not implement this model kind");
        }
        const int32_t rc = backend->create(input);
        if (rc != UNIRT_SUCCESS) return rc;
        *out_handle = HandleTraits<Interface>::track(backend.get());
        backend.release();
        return UNIRT_SUCCESS;
    });
}

/** Tear a backend down and forget its handle. */
template <typename Interface>
int32_t close_backend(
    const char* entrypoint, typename HandleTraits<Interface>::CHandle* handle) noexcept {
    return shielded(entrypoint, UNIRT_ERROR_COMMON_UNKNOWN, [&]() -> int32_t {
        if (!handle) {
            return fail(UNIRT_ERROR_COMMON_NOT_INITIALIZED, std::string(entrypoint) + ": handle is NULL");
        }
        Interface* backend = HandleTraits<Interface>::untrack(handle);
        if (!backend) {
            return fail(
                UNIRT_ERROR_COMMON_INVALID_INPUT,
                std::string(entrypoint) + ": handle is stale, already destroyed, or of the wrong model kind");
        }
        delete backend;
        return UNIRT_SUCCESS;
    });
}

/** Forward a generate call, splicing a UTF-8 reassembly stage between the
 *  plugin's raw byte pieces and the caller's callback, and deriving the
 *  speed figures afterwards. Works for both modalities because their
 *  generate inputs share the on_token/user_data field shape. */
template <typename Interface, typename GenInput, typename GenOutput>
int32_t run_generation(Interface& backend, const GenInput* input, GenOutput* output) {
    int32_t rc;
    if (input->on_token) {
        StreamJoiner joiner(input->on_token, input->user_data);
        GenInput     spliced = *input;
        spliced.on_token     = StreamJoiner::trampoline();
        spliced.user_data    = &joiner;
        rc = backend.generate(&spliced, output);
        if (rc == UNIRT_SUCCESS) joiner.finish();
    } else {
        rc = backend.generate(input, output);
    }
    finalize_profile(output->profile_data);
    return rc;
}

/** Field checks shared by both modalities' create paths. */
inline bool model_config_sane(const unirt_ModelConfig& config) noexcept {
    return config.n_ctx >= 0 && config.n_threads >= 0 && config.n_threads_batch >= 0 &&
           config.n_batch >= 0 && config.n_ubatch >= 0 && config.n_seq_max >= 0 &&
           config.n_gpu_layers >= -1;
}

/** Field checks shared by both modalities' generation configs. Media counts
 *  are validated here; whether media is *allowed* stays with the caller. */
inline bool generation_config_sane(const unirt_GenerationConfig& config) noexcept {
    if (config.max_tokens < 0 || config.stop_count < 0 || config.image_count < 0 ||
        config.audio_count < 0 || config.n_past < 0 || config.sliding_window_n_keep < 0) {
        return false;
    }
    if ((config.stop_count > 0 && !config.stop) ||
        (config.image_count > 0 && !config.image_paths) ||
        (config.audio_count > 0 && !config.audio_paths)) {
        return false;
    }
    for (int32_t i = 0; i < config.stop_count; ++i) {
        if (!config.stop[i]) return false;
    }
    return true;
}

}  // namespace unirt::bridge
