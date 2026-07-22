// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// JNI glue between ai.unirt.Native (Kotlin) and the UniRT C API. Kept to a
// mechanical pattern: unpack Java arguments, call one unirt_* entrypoint,
// wrap the result. Streaming tokens re-enter Kotlin through the callback on
// the calling thread, so the JNIEnv stays valid for the whole generate call.

#include <jni.h>

#include <string>
#include <vector>

#include "unirt.h"

namespace {

/** UTF-8 copy of a jstring; empty for null. */
std::string as_utf8(JNIEnv* env, jstring text) {
    if (!text) return {};
    const char* raw = env->GetStringUTFChars(text, nullptr);
    if (!raw) return {};
    std::string copied(raw);
    env->ReleaseStringUTFChars(text, raw);
    return copied;
}

struct TokenRelay {
    JNIEnv*   env;
    jobject   callback;
    jmethodID on_token;
    bool      java_threw = false;
};

bool relay_token(const char* piece, void* user_data) {
    auto* relay = static_cast<TokenRelay*>(user_data);
    if (!relay || relay->java_threw) return false;
    jstring text = relay->env->NewStringUTF(piece ? piece : "");
    if (!text) {
        relay->java_threw = true;
        return false;
    }
    const jboolean keep_going =
        relay->env->CallBooleanMethod(relay->callback, relay->on_token, text);
    relay->env->DeleteLocalRef(text);
    if (relay->env->ExceptionCheck()) {
        relay->java_threw = true;
        return false;
    }
    return keep_going == JNI_TRUE;
}

jlong as_handle(unirt_LLM* handle) { return reinterpret_cast<jlong>(handle); }
unirt_LLM* as_llm(jlong handle) { return reinterpret_cast<unirt_LLM*>(handle); }

jlong as_handle(unirt_VLM* handle) { return reinterpret_cast<jlong>(handle); }
unirt_VLM* as_vlm(jlong handle) { return reinterpret_cast<unirt_VLM*>(handle); }

/** Builds an ai.unirt.LlmGenerateResult(text, GenerationProfile(...)) from
 *  one generate() call's output. Returns null (with a pending Java
 *  exception) if either Kotlin class/constructor can't be resolved. */
jobject to_generate_result(JNIEnv* env, const char* text, const unirt_ProfileData& profile) {
    jclass profile_class = env->FindClass("ai/unirt/GenerationProfile");
    if (!profile_class) return nullptr;
    jmethodID profile_ctor = env->GetMethodID(profile_class, "<init>", "(JJJJJDDLjava/lang/String;)V");
    if (!profile_ctor) return nullptr;

    jstring stop_reason = env->NewStringUTF(profile.stop_reason ? profile.stop_reason : "");
    jobject profile_obj = env->NewObject(
        profile_class, profile_ctor, static_cast<jlong>(profile.ttft),
        static_cast<jlong>(profile.prompt_time), static_cast<jlong>(profile.decode_time),
        static_cast<jlong>(profile.prompt_tokens), static_cast<jlong>(profile.generated_tokens),
        static_cast<jdouble>(profile.prefill_speed), static_cast<jdouble>(profile.decoding_speed),
        stop_reason);
    env->DeleteLocalRef(stop_reason);
    if (!profile_obj) return nullptr;

    jclass result_class = env->FindClass("ai/unirt/LlmGenerateResult");
    if (!result_class) return nullptr;
    jmethodID result_ctor =
        env->GetMethodID(result_class, "<init>", "(Ljava/lang/String;Lai/unirt/GenerationProfile;)V");
    if (!result_ctor) return nullptr;

    jstring text_str = env->NewStringUTF(text ? text : "");
    jobject result = env->NewObject(result_class, result_ctor, text_str, profile_obj);
    env->DeleteLocalRef(text_str);
    env->DeleteLocalRef(profile_obj);
    return result;
}

/** Builds an ai.unirt.RuntimeStats from either modality's stats struct —
 *  unirt_LlmRuntimeStats and unirt_VlmRuntimeStats are deliberately the
 *  same shape (see unirt.h), so one builder serves both. */
jobject to_runtime_stats(
    JNIEnv* env, int64_t model_bytes, int64_t kv_cache_bytes, int64_t device_peak_bytes,
    int64_t process_rss_bytes, const char* device_name) {
    jclass stats_class = env->FindClass("ai/unirt/RuntimeStats");
    if (!stats_class) return nullptr;
    jmethodID ctor = env->GetMethodID(stats_class, "<init>", "(JJJJLjava/lang/String;)V");
    if (!ctor) return nullptr;
    jstring name = env->NewStringUTF(device_name ? device_name : "");
    jobject stats = env->NewObject(
        stats_class, ctor, static_cast<jlong>(model_bytes), static_cast<jlong>(kv_cache_bytes),
        static_cast<jlong>(device_peak_bytes), static_cast<jlong>(process_rss_bytes), name);
    env->DeleteLocalRef(name);
    return stats;
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL Java_ai_unirt_Native_init(JNIEnv*, jclass) { return unirt_init(); }

JNIEXPORT jint JNICALL Java_ai_unirt_Native_deinit(JNIEnv*, jclass) { return unirt_deinit(); }

JNIEXPORT jstring JNICALL Java_ai_unirt_Native_version(JNIEnv* env, jclass) {
    return env->NewStringUTF(unirt_version());
}

JNIEXPORT jstring JNICALL Java_ai_unirt_Native_lastError(JNIEnv* env, jclass) {
    return env->NewStringUTF(unirt_last_error_message());
}

JNIEXPORT jstring JNICALL Java_ai_unirt_Native_errorMessage(JNIEnv* env, jclass, jint code) {
    return env->NewStringUTF(unirt_get_error_message(static_cast<unirt_ErrorCode>(code)));
}

JNIEXPORT jobjectArray JNICALL Java_ai_unirt_Native_pluginList(JNIEnv* env, jclass) {
    unirt_GetPluginListOutput output{};
    jclass string_class = env->FindClass("java/lang/String");
    if (unirt_get_plugin_list(&output) != UNIRT_SUCCESS || output.plugin_count <= 0) {
        return env->NewObjectArray(0, string_class, nullptr);
    }
    jobjectArray listed = env->NewObjectArray(output.plugin_count, string_class, nullptr);
    for (int32_t i = 0; listed && i < output.plugin_count; ++i) {
        jstring id = env->NewStringUTF(output.plugin_ids[i]);
        env->SetObjectArrayElement(listed, i, id);
        env->DeleteLocalRef(id);
    }
    unirt_free(output.plugin_ids);
    return listed;
}

JNIEXPORT jlong JNICALL Java_ai_unirt_Native_llmCreate(
    JNIEnv* env, jclass, jstring model_path, jstring plugin_id, jstring device_id, jint n_ctx,
    jint n_gpu_layers) {
    const std::string path   = as_utf8(env, model_path);
    const std::string plugin = as_utf8(env, plugin_id);
    const std::string device = as_utf8(env, device_id);

    unirt_LlmCreateInput input{};
    input.model_path          = path.c_str();
    input.plugin_id           = plugin.empty() ? "llama_cpp" : plugin.c_str();
    input.device_id           = device.empty() ? nullptr : device.c_str();
    input.config.n_ctx        = n_ctx;
    input.config.n_gpu_layers = n_gpu_layers;

    unirt_LLM* handle = nullptr;
    if (unirt_llm_create(&input, &handle) != UNIRT_SUCCESS) return 0;
    return as_handle(handle);
}

JNIEXPORT jint JNICALL Java_ai_unirt_Native_llmDestroy(JNIEnv*, jclass, jlong handle) {
    return unirt_llm_destroy(as_llm(handle));
}

JNIEXPORT jint JNICALL Java_ai_unirt_Native_llmReset(JNIEnv*, jclass, jlong handle) {
    return unirt_llm_reset(as_llm(handle));
}

JNIEXPORT jstring JNICALL Java_ai_unirt_Native_llmApplyChatTemplate(
    JNIEnv* env, jclass, jlong handle, jobjectArray roles, jobjectArray contents,
    jboolean add_generation_prompt) {
    const jsize count = roles ? env->GetArrayLength(roles) : 0;
    if (count <= 0 || !contents || env->GetArrayLength(contents) != count) return nullptr;

    std::vector<std::string>          texts(static_cast<size_t>(count) * 2);
    std::vector<unirt_LlmChatMessage> messages(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        auto role    = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto content = static_cast<jstring>(env->GetObjectArrayElement(contents, i));
        texts[static_cast<size_t>(i) * 2]     = as_utf8(env, role);
        texts[static_cast<size_t>(i) * 2 + 1] = as_utf8(env, content);
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
        messages[static_cast<size_t>(i)] = {
            texts[static_cast<size_t>(i) * 2].c_str(),
            texts[static_cast<size_t>(i) * 2 + 1].c_str(),
        };
    }

    unirt_LlmApplyChatTemplateInput input{};
    input.messages              = messages.data();
    input.message_count         = count;
    input.add_generation_prompt = add_generation_prompt == JNI_TRUE;

    unirt_LlmApplyChatTemplateOutput output{};
    if (unirt_llm_apply_chat_template(as_llm(handle), &input, &output) != UNIRT_SUCCESS) {
        return nullptr;
    }
    jstring rendered = env->NewStringUTF(output.formatted_text ? output.formatted_text : "");
    unirt_free(output.formatted_text);
    return rendered;
}

JNIEXPORT jobject JNICALL Java_ai_unirt_Native_llmGenerate(
    JNIEnv* env, jclass, jlong handle, jstring prompt, jint max_tokens, jfloat temperature,
    jfloat top_p, jint top_k, jint seed, jobject on_token) {
    const std::string prompt_text = as_utf8(env, prompt);

    unirt_SamplerConfig sampler{};
    sampler.temperature = temperature;
    sampler.top_p       = top_p;
    sampler.top_k       = top_k;
    sampler.seed        = seed;

    unirt_GenerationConfig config{};
    config.max_tokens     = max_tokens;
    config.sampler_config = &sampler;

    unirt_LlmGenerateInput input{};
    input.prompt_utf8 = prompt_text.c_str();
    input.config      = &config;

    TokenRelay relay{env, on_token, nullptr};
    if (on_token) {
        jclass callback_class = env->GetObjectClass(on_token);
        relay.on_token = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)Z");
        if (!relay.on_token) return nullptr;
        input.on_token  = relay_token;
        input.user_data = &relay;
    }

    unirt_LlmGenerateOutput output{};
    const int32_t status = unirt_llm_generate(as_llm(handle), &input, &output);
    if (relay.java_threw) {
        unirt_free(output.full_text);
        return nullptr;  // the pending Java exception propagates on return
    }
    if (status != UNIRT_SUCCESS) return nullptr;

    jobject result = to_generate_result(env, output.full_text, output.profile_data);
    unirt_free(output.full_text);
    return result;
}

JNIEXPORT jobject JNICALL Java_ai_unirt_Native_llmRuntimeStats(JNIEnv* env, jclass, jlong handle) {
    unirt_LlmRuntimeStats stats{};
    if (unirt_llm_get_runtime_stats(as_llm(handle), &stats) != UNIRT_SUCCESS) return nullptr;
    return to_runtime_stats(
        env, stats.model_bytes, stats.kv_cache_bytes, stats.device_peak_bytes,
        stats.process_rss_bytes, stats.device_name);
}

JNIEXPORT jlong JNICALL Java_ai_unirt_Native_vlmCreate(
    JNIEnv* env, jclass, jstring model_path, jstring mmproj_path, jstring plugin_id,
    jstring device_id, jint n_ctx, jint n_gpu_layers) {
    const std::string path   = as_utf8(env, model_path);
    const std::string mmproj = as_utf8(env, mmproj_path);
    const std::string plugin = as_utf8(env, plugin_id);
    const std::string device = as_utf8(env, device_id);

    unirt_VlmCreateInput input{};
    input.model_path          = path.c_str();
    input.mmproj_path         = mmproj.empty() ? nullptr : mmproj.c_str();
    input.plugin_id           = plugin.empty() ? "llama_cpp" : plugin.c_str();
    input.device_id           = device.empty() ? nullptr : device.c_str();
    input.config.n_ctx        = n_ctx;
    input.config.n_gpu_layers = n_gpu_layers;

    unirt_VLM* handle = nullptr;
    if (unirt_vlm_create(&input, &handle) != UNIRT_SUCCESS) return 0;
    return as_handle(handle);
}

JNIEXPORT jint JNICALL Java_ai_unirt_Native_vlmDestroy(JNIEnv*, jclass, jlong handle) {
    return unirt_vlm_destroy(as_vlm(handle));
}

JNIEXPORT jint JNICALL Java_ai_unirt_Native_vlmReset(JNIEnv*, jclass, jlong handle) {
    return unirt_vlm_reset(as_vlm(handle));
}

JNIEXPORT jobject JNICALL Java_ai_unirt_Native_vlmGetCapabilities(JNIEnv* env, jclass, jlong handle) {
    unirt_VlmCapabilities caps{};
    if (unirt_vlm_get_capabilities(as_vlm(handle), &caps) != UNIRT_SUCCESS) return nullptr;

    jclass caps_class = env->FindClass("ai/unirt/VlmCapabilities");
    if (!caps_class) return nullptr;
    jmethodID ctor = env->GetMethodID(caps_class, "<init>", "(ZZ)V");
    if (!ctor) return nullptr;
    return env->NewObject(
        caps_class, ctor, static_cast<jboolean>(caps.supports_vision),
        static_cast<jboolean>(caps.supports_audio));
}

JNIEXPORT jobject JNICALL Java_ai_unirt_Native_vlmRuntimeStats(JNIEnv* env, jclass, jlong handle) {
    unirt_VlmRuntimeStats stats{};
    if (unirt_vlm_get_runtime_stats(as_vlm(handle), &stats) != UNIRT_SUCCESS) return nullptr;
    return to_runtime_stats(
        env, stats.model_bytes, stats.kv_cache_bytes, stats.device_peak_bytes,
        stats.process_rss_bytes, stats.device_name);
}

// Nested Kotlin arrays (one row of content parts per message) rather than a
// flat parallel-index scheme — VLM turns have a variable number of parts
// (text/image/audio), unlike LLM's one-string-per-turn.
JNIEXPORT jstring JNICALL Java_ai_unirt_Native_vlmApplyChatTemplate(
    JNIEnv* env, jclass, jlong handle, jobjectArray roles, jobjectArray content_types,
    jobjectArray content_texts, jboolean enable_thinking, jboolean grounding) {
    const jsize count = roles ? env->GetArrayLength(roles) : 0;
    if (count <= 0 || !content_types || !content_texts ||
        env->GetArrayLength(content_types) != count || env->GetArrayLength(content_texts) != count) {
        return nullptr;
    }

    std::vector<std::string>                  role_texts(static_cast<size_t>(count));
    std::vector<std::vector<std::string>>      part_types(static_cast<size_t>(count));
    std::vector<std::vector<std::string>>      part_texts(static_cast<size_t>(count));
    std::vector<std::vector<unirt_VlmContent>> contents(static_cast<size_t>(count));
    std::vector<unirt_VlmChatMessage>          messages(static_cast<size_t>(count));

    for (jsize i = 0; i < count; ++i) {
        auto role                   = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        role_texts[static_cast<size_t>(i)] = as_utf8(env, role);
        env->DeleteLocalRef(role);

        auto types_row = static_cast<jobjectArray>(env->GetObjectArrayElement(content_types, i));
        auto texts_row = static_cast<jobjectArray>(env->GetObjectArrayElement(content_texts, i));
        const jsize part_count = types_row ? env->GetArrayLength(types_row) : 0;
        if (!texts_row || env->GetArrayLength(texts_row) != part_count) {
            if (types_row) env->DeleteLocalRef(types_row);
            if (texts_row) env->DeleteLocalRef(texts_row);
            return nullptr;
        }

        auto& types = part_types[static_cast<size_t>(i)];
        auto& texts = part_texts[static_cast<size_t>(i)];
        types.resize(static_cast<size_t>(part_count));
        texts.resize(static_cast<size_t>(part_count));
        for (jsize j = 0; j < part_count; ++j) {
            auto type_str = static_cast<jstring>(env->GetObjectArrayElement(types_row, j));
            auto text_str = static_cast<jstring>(env->GetObjectArrayElement(texts_row, j));
            types[static_cast<size_t>(j)] = as_utf8(env, type_str);
            texts[static_cast<size_t>(j)] = as_utf8(env, text_str);
            env->DeleteLocalRef(type_str);
            env->DeleteLocalRef(text_str);
        }
        env->DeleteLocalRef(types_row);
        env->DeleteLocalRef(texts_row);

        auto& parts = contents[static_cast<size_t>(i)];
        parts.resize(static_cast<size_t>(part_count));
        for (jsize j = 0; j < part_count; ++j) {
            parts[static_cast<size_t>(j)] = {types[static_cast<size_t>(j)].c_str(),
                                              texts[static_cast<size_t>(j)].c_str()};
        }
        messages[static_cast<size_t>(i)] = {
            role_texts[static_cast<size_t>(i)].c_str(), parts.data(),
            static_cast<int64_t>(part_count)};
    }

    unirt_VlmApplyChatTemplateInput input{};
    input.messages        = messages.data();
    input.message_count   = count;
    input.enable_thinking = enable_thinking == JNI_TRUE;
    input.grounding       = grounding == JNI_TRUE;

    unirt_VlmApplyChatTemplateOutput output{};
    if (unirt_vlm_apply_chat_template(as_vlm(handle), &input, &output) != UNIRT_SUCCESS) {
        return nullptr;
    }
    jstring rendered = env->NewStringUTF(output.formatted_text ? output.formatted_text : "");
    unirt_free(output.formatted_text);
    return rendered;
}

JNIEXPORT jobject JNICALL Java_ai_unirt_Native_vlmGenerate(
    JNIEnv* env, jclass, jlong handle, jstring prompt, jint max_tokens, jfloat temperature,
    jfloat top_p, jint top_k, jint seed, jobjectArray image_paths, jobjectArray audio_paths,
    jint image_max_length, jobject on_token) {
    const std::string prompt_text = as_utf8(env, prompt);

    unirt_SamplerConfig sampler{};
    sampler.temperature = temperature;
    sampler.top_p       = top_p;
    sampler.top_k       = top_k;
    sampler.seed        = seed;

    const jsize image_count = image_paths ? env->GetArrayLength(image_paths) : 0;
    const jsize audio_count = audio_paths ? env->GetArrayLength(audio_paths) : 0;
    std::vector<std::string> image_strs(static_cast<size_t>(image_count));
    std::vector<std::string> audio_strs(static_cast<size_t>(audio_count));
    std::vector<unirt_Path>  image_ptrs(static_cast<size_t>(image_count));
    std::vector<unirt_Path>  audio_ptrs(static_cast<size_t>(audio_count));
    for (jsize i = 0; i < image_count; ++i) {
        auto path = static_cast<jstring>(env->GetObjectArrayElement(image_paths, i));
        image_strs[static_cast<size_t>(i)] = as_utf8(env, path);
        image_ptrs[static_cast<size_t>(i)] = image_strs[static_cast<size_t>(i)].c_str();
        env->DeleteLocalRef(path);
    }
    for (jsize i = 0; i < audio_count; ++i) {
        auto path = static_cast<jstring>(env->GetObjectArrayElement(audio_paths, i));
        audio_strs[static_cast<size_t>(i)] = as_utf8(env, path);
        audio_ptrs[static_cast<size_t>(i)] = audio_strs[static_cast<size_t>(i)].c_str();
        env->DeleteLocalRef(path);
    }

    unirt_GenerationConfig config{};
    config.max_tokens       = max_tokens;
    config.sampler_config   = &sampler;
    config.image_paths      = image_count > 0 ? image_ptrs.data() : nullptr;
    config.image_count      = static_cast<int32_t>(image_count);
    config.image_max_length = image_max_length;
    config.audio_paths      = audio_count > 0 ? audio_ptrs.data() : nullptr;
    config.audio_count      = static_cast<int32_t>(audio_count);

    unirt_VlmGenerateInput input{};
    input.prompt_utf8 = prompt_text.c_str();
    input.config      = &config;

    TokenRelay relay{env, on_token, nullptr};
    if (on_token) {
        jclass callback_class = env->GetObjectClass(on_token);
        relay.on_token = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)Z");
        if (!relay.on_token) return nullptr;
        input.on_token  = relay_token;
        input.user_data = &relay;
    }

    unirt_VlmGenerateOutput output{};
    const int32_t status = unirt_vlm_generate(as_vlm(handle), &input, &output);
    if (relay.java_threw) {
        unirt_free(output.full_text);
        return nullptr;  // the pending Java exception propagates on return
    }
    if (status != UNIRT_SUCCESS) return nullptr;

    jobject result = to_generate_result(env, output.full_text, output.profile_data);
    unirt_free(output.full_text);
    return result;
}

}  // extern "C"
