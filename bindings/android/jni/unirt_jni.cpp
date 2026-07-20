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

JNIEXPORT jstring JNICALL Java_ai_unirt_Native_llmGenerate(
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

    jstring text = env->NewStringUTF(
        output.full_text ? static_cast<const char*>(output.full_text) : "");
    unirt_free(output.full_text);
    return text;
}

}  // extern "C"
