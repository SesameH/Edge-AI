// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

/*
 * unirt-bench: a portable benchmark built exclusively on the public UniRT C
 * interface.  It deliberately owns no backend-specific inference code.
 */

#include <unirt.h>

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#endif
#else
#include <dirent.h>
#endif

#define MAX_MEDIA 16
#define STOP_REASON_CAPACITY 48

typedef struct {
    const char* plugin;
    const char* device;
    const char* device_id;
    const char* model_path;
    const char* tokenizer_path;
    const char* mmproj_path;
    const char* images[MAX_MEDIA];
    int32_t     image_count;
    const char* audios[MAX_MEDIA];
    int32_t     audio_count;
    bool        force_vlm;

    int32_t n_prompt;
    int32_t n_generate;
    int32_t n_ctx;
    int32_t n_threads;
    int32_t n_gpu_layers;
    int32_t warmups;
    int32_t repetitions;
    int32_t seed;
    float   temperature;
    bool    reset_between_runs;
    bool    accuracy;

    char*       prompt;
    const char* cell_id;
    const char* output_json;
    const char* output_md;
    const char* matrix_file;
    const char* output_json_dir;
} bench_options;

typedef struct {
    int32_t run_index;
    int64_t ttft_us;
    int64_t prompt_us;
    int64_t decode_us;
    int64_t prompt_tokens;
    int64_t generated_tokens;
    double  prefill_tps;
    double  decode_tps;
    char    stop_reason[STOP_REASON_CAPACITY];
} bench_run;

typedef struct {
    double median;
    double minimum;
    double maximum;
    double mean;
    double deviation;
} distribution;

typedef struct {
    distribution ttft_ms;
    distribution prefill_tps;
    distribution decode_tps;
    double       prompt_tokens_median;
    double       generated_tokens_median;
} bench_summary;

static void print_usage(const char* program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --plugin ID --device MODE -m MODEL [options]\n"
            "  %s --matrix-file FILE [--output-json-dir DIR] [options]\n\n"
            "Core options:\n"
            "  --plugin ID              backend plugin id\n"
            "  --device MODE            auto, cpu, gpu, npu, or hybrid\n"
            "  --device-id ID           concrete backend device id\n"
            "  -m, --model PATH         local model file or directory\n"
            "  --tokenizer-path PATH    tokenizer override\n"
            "  --mmproj-path PATH       projector path; enables VLM mode\n"
            "  --vlm                    force VLM mode without a projector path\n"
            "  --image PATH             image input; repeatable\n"
            "  --audio PATH             audio input; repeatable\n\n"
            "Workload options:\n"
            "  -p, --n-prompt N         random-token prefill length (default 512)\n"
            "  -n, --n-gen N            generation budget (default 128)\n"
            "  -r, --repetitions N      measured runs (default 5)\n"
            "  --warmup N               warmup runs (default 1)\n"
            "  --no-warmup              disable warmup\n"
            "  --prompt-file PATH       benchmark a UTF-8 text prompt\n"
            "  --temperature F          sampling temperature (default 0)\n"
            "  --seed N                 sampler and prompt seed (default 42)\n"
            "  -c, --ctx-size N         context length; 0 uses model default\n"
            "  -t, --threads N          decode threads; 0 uses backend default\n"
            "  -ngl, --n-gpu-layers N   layer offload override\n"
            "  --no-reset-between-runs  retain backend state between runs\n"
            "  --accuracy               one measured run and print its text\n\n"
            "Output options:\n"
            "  --cell-id ID              report label\n"
            "  --output-json PATH        write a JSON report\n"
            "  --output-md PATH          append a Markdown result row\n"
            "  --matrix-file PATH        tab-separated benchmark cells\n"
            "  --output-json-dir DIR     per-cell JSON directory in matrix mode\n",
            program,
            program);
}

static void fail_message(const char* message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(1);
}

static void require_status(int32_t status, const char* operation) {
    if (status == UNIRT_SUCCESS) return;
    fprintf(stderr,
            "ERROR: %s: %s (%d)\n",
            operation,
            unirt_get_error_message((unirt_ErrorCode)status),
            status);
    exit(1);
}

static const char* take_value(int argc, char** argv, int* index) {
    if (*index + 1 >= argc) {
        fprintf(stderr, "ERROR: %s requires a value\n", argv[*index]);
        exit(2);
    }
    *index += 1;
    return argv[*index];
}

static int32_t parse_i32(const char* value, const char* flag, int32_t minimum) {
    char* end = NULL;
    errno     = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || !end || *end || parsed < minimum || parsed > INT32_MAX) {
        fprintf(stderr, "ERROR: invalid value for %s: %s\n", flag, value);
        exit(2);
    }
    return (int32_t)parsed;
}

static float parse_float(const char* value, const char* flag) {
    char* end = NULL;
    errno     = 0;
    float parsed = strtof(value, &end);
    if (errno || !end || *end || !isfinite(parsed)) {
        fprintf(stderr, "ERROR: invalid value for %s: %s\n", flag, value);
        exit(2);
    }
    return parsed;
}

static char* read_text_file(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: cannot open prompt file: %s\n", path);
        exit(1);
    }
    if (fseek(file, 0, SEEK_END) != 0) fail_message("cannot seek prompt file");
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) fail_message("cannot size prompt file");
    char* text = (char*)malloc((size_t)length + 1);
    if (!text) fail_message("out of memory while reading prompt file");
    size_t read_count = fread(text, 1, (size_t)length, file);
    fclose(file);
    if (read_count != (size_t)length) {
        free(text);
        fail_message("short read from prompt file");
    }
    text[length] = '\0';
    return text;
}

static void default_options(bench_options* options) {
    memset(options, 0, sizeof(*options));
    options->device             = "auto";
    options->n_prompt           = 512;
    options->n_generate         = 128;
    options->n_gpu_layers       = -1;
    options->warmups            = 1;
    options->repetitions        = 5;
    options->seed               = 42;
    options->temperature        = 0.0f;
    options->reset_between_runs = true;
}

static void parse_arguments(int argc, char** argv, bench_options* options) {
    default_options(options);
    for (int index = 1; index < argc; ++index) {
        const char* flag = argv[index];
        if (!strcmp(flag, "-h") || !strcmp(flag, "--help")) {
            print_usage(argv[0]);
            exit(0);
        } else if (!strcmp(flag, "--plugin")) {
            options->plugin = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--device")) {
            options->device = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--device-id")) {
            options->device_id = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "-m") || !strcmp(flag, "--model")) {
            options->model_path = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--tokenizer-path")) {
            options->tokenizer_path = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--mmproj-path")) {
            options->mmproj_path = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--vlm")) {
            options->force_vlm = true;
        } else if (!strcmp(flag, "--image")) {
            if (options->image_count == MAX_MEDIA) fail_message("too many --image arguments");
            options->images[options->image_count++] = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--audio")) {
            if (options->audio_count == MAX_MEDIA) fail_message("too many --audio arguments");
            options->audios[options->audio_count++] = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "-p") || !strcmp(flag, "--n-prompt")) {
            options->n_prompt = parse_i32(take_value(argc, argv, &index), flag, 1);
        } else if (!strcmp(flag, "-n") || !strcmp(flag, "--n-gen")) {
            options->n_generate = parse_i32(take_value(argc, argv, &index), flag, 1);
        } else if (!strcmp(flag, "-r") || !strcmp(flag, "--repetitions")) {
            options->repetitions = parse_i32(take_value(argc, argv, &index), flag, 1);
        } else if (!strcmp(flag, "--warmup")) {
            options->warmups = parse_i32(take_value(argc, argv, &index), flag, 0);
        } else if (!strcmp(flag, "--no-warmup")) {
            options->warmups = 0;
        } else if (!strcmp(flag, "--prompt-file")) {
            options->prompt = read_text_file(take_value(argc, argv, &index));
        } else if (!strcmp(flag, "--temperature")) {
            options->temperature = parse_float(take_value(argc, argv, &index), flag);
        } else if (!strcmp(flag, "--seed")) {
            options->seed = parse_i32(take_value(argc, argv, &index), flag, -1);
        } else if (!strcmp(flag, "-c") || !strcmp(flag, "--ctx-size")) {
            options->n_ctx = parse_i32(take_value(argc, argv, &index), flag, 0);
        } else if (!strcmp(flag, "-t") || !strcmp(flag, "--threads")) {
            options->n_threads = parse_i32(take_value(argc, argv, &index), flag, 0);
        } else if (!strcmp(flag, "-ngl") || !strcmp(flag, "--n-gpu-layers")) {
            options->n_gpu_layers = parse_i32(take_value(argc, argv, &index), flag, -1);
        } else if (!strcmp(flag, "--no-reset-between-runs")) {
            options->reset_between_runs = false;
        } else if (!strcmp(flag, "--accuracy")) {
            options->accuracy    = true;
            options->warmups     = 0;
            options->repetitions = 1;
        } else if (!strcmp(flag, "--cell-id")) {
            options->cell_id = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--output-json")) {
            options->output_json = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--output-md")) {
            options->output_md = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--matrix-file")) {
            options->matrix_file = take_value(argc, argv, &index);
        } else if (!strcmp(flag, "--output-json-dir")) {
            options->output_json_dir = take_value(argc, argv, &index);
        } else {
            fprintf(stderr, "ERROR: unknown option: %s\n", flag);
            print_usage(argv[0]);
            exit(2);
        }
    }

    if (!options->matrix_file && (!options->plugin || !options->model_path)) {
        fail_message("single-cell mode requires --plugin and --model");
    }
}

static bool is_local_path(const char* value) {
    if (!value || !*value) return false;
    if (value[0] == '/' || value[0] == '.' || value[0] == '\\') return true;
#ifdef _WIN32
    if (strlen(value) > 2 && value[1] == ':') return true;
#endif
    return strchr(value, '/') == NULL;
}

static bool metadata_says_vlm(const char* model_path) {
    struct stat status;
    char        directory[1024];
    if (!model_path || strlen(model_path) >= sizeof(directory)) return false;
    strcpy(directory, model_path);
    if (stat(directory, &status) == 0 && S_ISREG(status.st_mode)) {
        char* slash = strrchr(directory, '/');
#ifdef _WIN32
        char* backslash = strrchr(directory, '\\');
        if (backslash && (!slash || backslash > slash)) slash = backslash;
#endif
        if (!slash) return false;
        *slash = '\0';
    }

    char metadata[1152];
    if (snprintf(metadata, sizeof(metadata), "%s/metadata.json", directory) >= (int)sizeof(metadata)) return false;
    FILE* file = fopen(metadata, "rb");
    if (!file) return false;
    char buffer[8193];
    size_t count = fread(buffer, 1, sizeof(buffer) - 1, file);
    fclose(file);
    buffer[count] = '\0';
    const char* key = strstr(buffer, "\"supports_vision\"");
    if (!key) return false;
    const char* colon = strchr(key, ':');
    if (!colon) return false;
    do {
        colon++;
    } while (*colon == ' ' || *colon == '\t' || *colon == '\r' || *colon == '\n');
    return !strncmp(colon, "true", 4);
}

static bool stream_token(const char* token, void* user_data) {
    (void)token;
    (void)user_data;
    return true;
}

static void fill_sampler(const bench_options* options, unirt_SamplerConfig* sampler) {
    memset(sampler, 0, sizeof(*sampler));
    sampler->temperature        = options->temperature;
    sampler->top_p              = 1.0f;
    sampler->repetition_penalty = 1.0f;
    sampler->seed               = options->seed;
}

static void fill_generation(const bench_options* options,
                            unirt_SamplerConfig* sampler,
                            bool with_media,
                            unirt_GenerationConfig* config) {
    memset(config, 0, sizeof(*config));
    config->max_tokens     = options->n_generate;
    config->sampler_config = sampler;
    if (!with_media) return;
    config->image_paths = (unirt_Path*)options->images;
    config->image_count = options->image_count;
    config->audio_paths = (unirt_Path*)options->audios;
    config->audio_count = options->audio_count;
}

static void fill_model_config(const bench_options* options, int32_t ngl, unirt_ModelConfig* config) {
    memset(config, 0, sizeof(*config));
    config->n_ctx        = options->n_ctx;
    config->n_threads    = options->n_threads;
    config->n_gpu_layers = ngl;
}

static void copy_profile(int32_t index, const unirt_ProfileData* profile, bench_run* run) {
    memset(run, 0, sizeof(*run));
    run->run_index       = index;
    run->ttft_us         = profile->ttft;
    run->prompt_us       = profile->prompt_time;
    run->decode_us       = profile->decode_time;
    run->prompt_tokens   = profile->prompt_tokens;
    run->generated_tokens = profile->generated_tokens;
    run->prefill_tps     = profile->prefill_speed;
    run->decode_tps      = profile->decoding_speed;
    if (profile->stop_reason) {
        snprintf(run->stop_reason, sizeof(run->stop_reason), "%s", profile->stop_reason);
    }
}

static int32_t* make_random_prompt(unirt_LLM* model, const bench_options* options) {
    unirt_LlmModelInfo info;
    memset(&info, 0, sizeof(info));
    int32_t status = unirt_llm_get_model_info(model, &info);
    if (status != UNIRT_SUCCESS || info.vocab_size < 1) {
        fprintf(stderr,
                "ERROR: backend cannot provide vocabulary metadata; use --prompt-file "
                "for text-prompt mode (%d)\n",
                status);
        return NULL;
    }
    int32_t* tokens = (int32_t*)malloc(sizeof(int32_t) * (size_t)options->n_prompt);
    if (!tokens) return NULL;
    srand((unsigned int)options->seed);
    for (int32_t index = 0; index < options->n_prompt; ++index) {
        tokens[index] = rand() % info.vocab_size;
    }
    if (info.add_bos && info.bos_token >= 0) tokens[0] = info.bos_token;
    return tokens;
}

static void run_llm(const bench_options* options, const char* device_id, int32_t ngl, bench_run* measured) {
    unirt_LlmCreateInput create;
    memset(&create, 0, sizeof(create));
    create.model_path     = options->model_path;
    create.tokenizer_path = options->tokenizer_path;
    create.plugin_id      = options->plugin;
    create.device_id      = device_id;
    fill_model_config(options, ngl, &create.config);

    unirt_LLM* model = NULL;
    require_status(unirt_llm_create(&create, &model), "unirt_llm_create");

    int32_t* random_tokens = NULL;
    if (!options->prompt) {
        random_tokens = make_random_prompt(model, options);
        if (!random_tokens) {
            unirt_llm_destroy(model);
            fail_message("unable to construct random prompt");
        }
    }

    unirt_SamplerConfig sampler;
    unirt_GenerationConfig generation;
    fill_sampler(options, &sampler);
    fill_generation(options, &sampler, false, &generation);

    int32_t total = options->warmups + options->repetitions;
    for (int32_t iteration = 0; iteration < total; ++iteration) {
        bool warmup = iteration < options->warmups;
        if (options->reset_between_runs) require_status(unirt_llm_reset(model), "unirt_llm_reset");

        unirt_LlmGenerateInput input;
        unirt_LlmGenerateOutput output;
        memset(&input, 0, sizeof(input));
        memset(&output, 0, sizeof(output));
        input.config   = &generation;
        input.on_token = stream_token;
        if (options->prompt) {
            input.prompt_utf8 = options->prompt;
        } else {
            input.input_ids       = random_tokens;
            input.input_ids_count = options->n_prompt;
        }
        require_status(unirt_llm_generate(model, &input, &output), "unirt_llm_generate");
        if (!warmup) {
            int32_t measured_index = iteration - options->warmups;
            copy_profile(measured_index, &output.profile_data, &measured[measured_index]);
            if (options->accuracy && output.full_text) printf("[gen ] %s\n", output.full_text);
        }
        unirt_free(output.full_text);
    }

    free(random_tokens);
    require_status(unirt_llm_destroy(model), "unirt_llm_destroy");
}

static char* format_vlm_prompt(unirt_VLM* model, const bench_options* options) {
    int32_t content_count = options->image_count + options->audio_count + 1;
    unirt_VlmContent* contents =
        (unirt_VlmContent*)calloc((size_t)content_count, sizeof(unirt_VlmContent));
    if (!contents) return NULL;

    int32_t cursor = 0;
    for (int32_t index = 0; index < options->image_count; ++index) {
        contents[cursor].type = "image";
        contents[cursor].text = options->images[index];
        cursor++;
    }
    for (int32_t index = 0; index < options->audio_count; ++index) {
        contents[cursor].type = "audio";
        contents[cursor].text = options->audios[index];
        cursor++;
    }
    contents[cursor].type = "text";
    contents[cursor].text = options->prompt ? options->prompt : "Describe the provided media.";

    unirt_VlmChatMessage message;
    memset(&message, 0, sizeof(message));
    message.role          = "user";
    message.contents      = contents;
    message.content_count = content_count;

    unirt_VlmApplyChatTemplateInput input;
    unirt_VlmApplyChatTemplateOutput output;
    memset(&input, 0, sizeof(input));
    memset(&output, 0, sizeof(output));
    input.messages      = &message;
    input.message_count = 1;
    int32_t status = unirt_vlm_apply_chat_template(model, &input, &output);
    free(contents);
    if (status != UNIRT_SUCCESS) {
        fprintf(stderr, "ERROR: VLM chat template: %s (%d)\n", unirt_get_error_message((unirt_ErrorCode)status), status);
        return NULL;
    }
    return output.formatted_text;
}

static void verify_vlm_media(unirt_VLM* model, const bench_options* options) {
    unirt_VlmCapabilities capabilities;
    memset(&capabilities, 0, sizeof(capabilities));
    require_status(unirt_vlm_get_capabilities(model, &capabilities), "unirt_vlm_get_capabilities");
    if (options->image_count && !capabilities.supports_vision) {
        fail_message("loaded VLM does not support image input");
    }
    if (options->audio_count && !capabilities.supports_audio) {
        fail_message("loaded VLM does not support audio input");
    }
}

static void run_vlm(const bench_options* options, const char* device_id, int32_t ngl, bench_run* measured) {
    unirt_VlmCreateInput create;
    memset(&create, 0, sizeof(create));
    create.model_path     = options->model_path;
    create.mmproj_path    = options->mmproj_path;
    create.tokenizer_path = options->tokenizer_path;
    create.plugin_id      = options->plugin;
    create.device_id      = device_id;
    fill_model_config(options, ngl, &create.config);

    unirt_VLM* model = NULL;
    require_status(unirt_vlm_create(&create, &model), "unirt_vlm_create");
    verify_vlm_media(model, options);

    unirt_SamplerConfig sampler;
    unirt_GenerationConfig generation;
    fill_sampler(options, &sampler);
    fill_generation(options, &sampler, true, &generation);

    int32_t total = options->warmups + options->repetitions;
    for (int32_t iteration = 0; iteration < total; ++iteration) {
        bool warmup = iteration < options->warmups;
        if (iteration || options->reset_between_runs) {
            require_status(unirt_vlm_reset(model), "unirt_vlm_reset");
        }
        char* prompt = format_vlm_prompt(model, options);
        if (!prompt) {
            unirt_vlm_destroy(model);
            fail_message("unable to format VLM prompt");
        }

        unirt_VlmGenerateInput input;
        unirt_VlmGenerateOutput output;
        memset(&input, 0, sizeof(input));
        memset(&output, 0, sizeof(output));
        input.prompt_utf8 = prompt;
        input.config      = &generation;
        input.on_token    = stream_token;
        require_status(unirt_vlm_generate(model, &input, &output), "unirt_vlm_generate");
        if (!warmup) {
            int32_t measured_index = iteration - options->warmups;
            copy_profile(measured_index, &output.profile_data, &measured[measured_index]);
            if (options->accuracy && output.full_text) printf("[gen ] %s\n", output.full_text);
        }
        unirt_free(output.full_text);
        unirt_free(prompt);
    }
    require_status(unirt_vlm_destroy(model), "unirt_vlm_destroy");
}

static int compare_double(const void* left, const void* right) {
    double a = *(const double*)left;
    double b = *(const double*)right;
    return (a > b) - (a < b);
}

static distribution describe(double* values, int32_t count) {
    distribution result;
    memset(&result, 0, sizeof(result));
    if (count <= 0) return result;
    qsort(values, (size_t)count, sizeof(double), compare_double);
    result.minimum = values[0];
    result.maximum = values[count - 1];
    result.median  = count % 2 ? values[count / 2]
                               : (values[count / 2 - 1] + values[count / 2]) / 2.0;
    for (int32_t index = 0; index < count; ++index) result.mean += values[index];
    result.mean /= count;
    for (int32_t index = 0; index < count; ++index) {
        double delta = values[index] - result.mean;
        result.deviation += delta * delta;
    }
    result.deviation = sqrt(result.deviation / count);
    return result;
}

typedef double (*run_value)(const bench_run* run);

static double value_ttft(const bench_run* run) { return (double)run->ttft_us / 1000.0; }
static double value_prefill(const bench_run* run) { return run->prefill_tps; }
static double value_decode(const bench_run* run) { return run->decode_tps; }
static double value_prompt_tokens(const bench_run* run) { return (double)run->prompt_tokens; }
static double value_generated_tokens(const bench_run* run) { return (double)run->generated_tokens; }

static distribution describe_runs(const bench_run* runs, int32_t count, run_value getter) {
    double* values = (double*)malloc(sizeof(double) * (size_t)count);
    if (!values) fail_message("out of memory while aggregating benchmark");
    for (int32_t index = 0; index < count; ++index) values[index] = getter(&runs[index]);
    distribution result = describe(values, count);
    free(values);
    return result;
}

static bench_summary summarize(const bench_run* runs, int32_t count) {
    bench_summary result;
    memset(&result, 0, sizeof(result));
    result.ttft_ms    = describe_runs(runs, count, value_ttft);
    result.prefill_tps = describe_runs(runs, count, value_prefill);
    result.decode_tps = describe_runs(runs, count, value_decode);
    result.prompt_tokens_median = describe_runs(runs, count, value_prompt_tokens).median;
    result.generated_tokens_median = describe_runs(runs, count, value_generated_tokens).median;
    return result;
}

static void write_json_string(FILE* file, const char* value) {
    if (!value) {
        fputs("null", file);
        return;
    }
    fputc('"', file);
    for (const unsigned char* cursor = (const unsigned char*)value; *cursor; ++cursor) {
        switch (*cursor) {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\b': fputs("\\b", file); break;
            case '\f': fputs("\\f", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            default:
                if (*cursor < 0x20) fprintf(file, "\\u%04x", *cursor);
                else fputc(*cursor, file);
        }
    }
    fputc('"', file);
}

static int64_t path_size(const char* path) {
    struct stat status;
    if (!path || stat(path, &status) != 0) return -1;
    if (S_ISREG(status.st_mode)) return (int64_t)status.st_size;
    if (!S_ISDIR(status.st_mode)) return 0;
    int64_t total = 0;
#ifdef _WIN32
    char pattern[1200];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    WIN32_FIND_DATAA data;
    HANDLE handle = FindFirstFileA(pattern, &data);
    if (handle == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!strcmp(data.cFileName, ".") || !strcmp(data.cFileName, "..")) continue;
        char child[1200];
        snprintf(child, sizeof(child), "%s\\%s", path, data.cFileName);
        int64_t size = path_size(child);
        if (size > 0) total += size;
    } while (FindNextFileA(handle, &data));
    FindClose(handle);
#else
    DIR* directory = opendir(path);
    if (!directory) return 0;
    struct dirent* entry;
    while ((entry = readdir(directory)) != NULL) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char child[1200];
        if (snprintf(child, sizeof(child), "%s/%s", path, entry->d_name) >= (int)sizeof(child)) continue;
        int64_t size = path_size(child);
        if (size > 0) total += size;
    }
    closedir(directory);
#endif
    return total;
}

static void write_distribution(FILE* file, const distribution* value) {
    fprintf(file,
            "{\"median\":%.6f,\"min\":%.6f,\"max\":%.6f,\"mean\":%.6f,\"stdev\":%.6f}",
            value->median,
            value->minimum,
            value->maximum,
            value->mean,
            value->deviation);
}

static void write_json_report(const bench_options* options,
                              const char* device_id,
                              int32_t ngl,
                              const bench_run* runs,
                              const bench_summary* summary) {
    FILE* file = fopen(options->output_json, "wb");
    if (!file) fail_message("cannot open JSON output");
    fputs("{\n  \"schema_version\":\"3\",\n  \"cell_id\":", file);
    write_json_string(file, options->cell_id ? options->cell_id : "cell");
    fputs(",\n  \"plugin\":", file);
    write_json_string(file, options->plugin);
    fputs(",\n  \"device\":", file);
    write_json_string(file, options->device);
    fputs(",\n  \"device_id\":", file);
    write_json_string(file, device_id);
    fputs(",\n  \"model_path\":", file);
    write_json_string(file, options->model_path);
    fprintf(file,
            ",\n  \"model_size_bytes\":%lld,\n"
            "  \"params\":{\"warmup\":%d,\"repetitions\":%d,\"n_prompt\":%d,"
            "\"n_gen\":%d,\"temperature\":%.6f,\"seed\":%d,\"n_ctx\":%d,"
            "\"n_threads\":%d,\"n_gpu_layers\":%d},\n  \"runs\":[\n",
            (long long)path_size(options->model_path),
            options->warmups,
            options->repetitions,
            options->n_prompt,
            options->n_generate,
            options->temperature,
            options->seed,
            options->n_ctx,
            options->n_threads,
            ngl);
    for (int32_t index = 0; index < options->repetitions; ++index) {
        const bench_run* run = &runs[index];
        fprintf(file,
                "    {\"run_idx\":%d,\"ttft_us\":%lld,\"prompt_time_us\":%lld,"
                "\"decode_time_us\":%lld,\"prompt_tokens\":%lld,\"gen_tokens\":%lld,"
                "\"prefill_tps\":%.6f,\"decode_tps\":%.6f,\"stop_reason\":",
                run->run_index,
                (long long)run->ttft_us,
                (long long)run->prompt_us,
                (long long)run->decode_us,
                (long long)run->prompt_tokens,
                (long long)run->generated_tokens,
                run->prefill_tps,
                run->decode_tps);
        write_json_string(file, run->stop_reason[0] ? run->stop_reason : NULL);
        fprintf(file, "}%s\n", index + 1 == options->repetitions ? "" : ",");
    }
    fputs("  ],\n  \"agg\":{\n    \"ttft_ms\":", file);
    write_distribution(file, &summary->ttft_ms);
    fputs(",\n    \"prefill_tps\":", file);
    write_distribution(file, &summary->prefill_tps);
    fputs(",\n    \"decode_tps\":", file);
    write_distribution(file, &summary->decode_tps);
    fprintf(file,
            ",\n    \"prompt_tokens\":{\"median\":%.6f},"
            "\n    \"gen_tokens\":{\"median\":%.6f}\n  }\n}\n",
            summary->prompt_tokens_median,
            summary->generated_tokens_median);
    fclose(file);
}

static void format_size(int64_t bytes, char* output, size_t capacity) {
    if (bytes < 0) snprintf(output, capacity, "-");
    else if (bytes < 1024) snprintf(output, capacity, "%lld B", (long long)bytes);
    else if (bytes < 1024 * 1024) snprintf(output, capacity, "%.1f KiB", bytes / 1024.0);
    else if (bytes < (int64_t)1024 * 1024 * 1024) snprintf(output, capacity, "%.1f MiB", bytes / 1048576.0);
    else snprintf(output, capacity, "%.2f GiB", bytes / 1073741824.0);
}

static void write_markdown(const bench_options* options, int32_t ngl, const bench_summary* summary) {
    struct stat status;
    bool new_file = stat(options->output_md, &status) != 0 || status.st_size == 0;
    FILE* file = fopen(options->output_md, "ab");
    if (!file) fail_message("cannot open Markdown output");
    if (new_file) {
        fputs("| Model | Size | Backend | Device | ngl | Test | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |\n"
              "|---|---:|---|---|---:|---|---:|---:|---:|\n",
              file);
    }
    char size[32];
    format_size(path_size(options->model_path), size, sizeof(size));
    fprintf(file,
            "| %s | %s | %s | %s | %d | pp%.0f+tg%.0f | %.1f ± %.1f | %.1f ± %.1f | %.1f ± %.1f |\n",
            options->cell_id ? options->cell_id : "cell",
            size,
            options->plugin,
            options->device,
            ngl,
            summary->prompt_tokens_median,
            summary->generated_tokens_median,
            summary->ttft_ms.median,
            summary->ttft_ms.deviation,
            summary->prefill_tps.median,
            summary->prefill_tps.deviation,
            summary->decode_tps.median,
            summary->decode_tps.deviation);
    fclose(file);
}

static int run_cell(bench_options* options) {
    if (!is_local_path(options->model_path)) {
        fprintf(stderr, "ERROR: model must be a local path: %s\n", options->model_path);
        return 1;
    }
    if (!options->force_vlm && !options->mmproj_path && metadata_says_vlm(options->model_path)) {
        options->force_vlm = true;
    }

    unirt_ResolveDeviceInput resolve_input;
    unirt_ResolveDeviceOutput resolve_output;
    memset(&resolve_input, 0, sizeof(resolve_input));
    memset(&resolve_output, 0, sizeof(resolve_output));
    resolve_input.plugin_id   = options->plugin;
    resolve_input.mode        = options->device;
    resolve_input.ngl_default = options->n_gpu_layers;
    int32_t status = unirt_resolve_device(&resolve_input, &resolve_output);
    if (status != UNIRT_SUCCESS) {
        fprintf(stderr, "ERROR: device resolution failed: %s (%d)\n", unirt_get_error_message((unirt_ErrorCode)status), status);
        return 1;
    }
    if (resolve_output.warning) fprintf(stderr, "[warn] %s\n", resolve_output.warning);
    const char* device_id = options->device_id ? options->device_id : resolve_output.device_id;
    int32_t ngl = options->n_gpu_layers >= 0 ? options->n_gpu_layers : resolve_output.ngl;

    bench_run* runs = (bench_run*)calloc((size_t)options->repetitions, sizeof(bench_run));
    if (!runs) fail_message("out of memory allocating run results");
    bool vlm = options->force_vlm || options->mmproj_path != NULL;
    if (vlm) run_vlm(options, device_id, ngl, runs);
    else run_llm(options, device_id, ngl, runs);

    bench_summary summary = summarize(runs, options->repetitions);
    printf("[ok  ] %s plugin=%s device=%s%s%s ttft=%.1fms prefill=%.1ftps decode=%.1ftps gen=%.0f tok\n",
           options->cell_id ? options->cell_id : "cell",
           options->plugin,
           options->device,
           device_id ? " id=" : "",
           device_id ? device_id : "",
           summary.ttft_ms.median,
           summary.prefill_tps.median,
           summary.decode_tps.median,
           summary.generated_tokens_median);
    if (options->output_json) write_json_report(options, device_id, ngl, runs, &summary);
    if (options->output_md) write_markdown(options, ngl, &summary);

    free(runs);
    unirt_free(resolve_output.device_id);
    unirt_free(resolve_output.warning);
    return 0;
}

static char* trim_line(char* line) {
    size_t length = strlen(line);
    while (length && (line[length - 1] == '\n' || line[length - 1] == '\r' ||
                      line[length - 1] == ' ' || line[length - 1] == '\t')) {
        line[--length] = '\0';
    }
    return line;
}

static int run_matrix(const bench_options* base) {
    FILE* file = fopen(base->matrix_file, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: cannot open matrix file: %s\n", base->matrix_file);
        return 1;
    }
    int errors = 0;
    int line_number = 0;
    char line[4096];
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        trim_line(line);
        if (!line[0] || line[0] == '#') continue;
        char* fields[8] = {0};
        int field_count = 1;
        fields[0] = line;
        for (char* cursor = line; *cursor && field_count < 8; ++cursor) {
            if (*cursor == '\t') {
                *cursor = '\0';
                fields[field_count++] = cursor + 1;
            }
        }
        if (field_count < 4) {
            fprintf(stderr, "ERROR: matrix line %d needs at least four tab-separated fields\n", line_number);
            errors++;
            continue;
        }
        bench_options cell = *base;
        cell.cell_id        = fields[0];
        cell.plugin         = fields[1];
        cell.device         = fields[2];
        cell.model_path     = fields[3];
        cell.tokenizer_path = field_count > 4 && fields[4][0] ? fields[4] : NULL;
        cell.mmproj_path    = field_count > 5 && fields[5][0] ? fields[5] : NULL;
        cell.image_count    = 0;
        cell.audio_count    = 0;
        cell.force_vlm      = field_count > 7 && fields[7][0];
        cell.output_md      = NULL;
        if (field_count > 6 && fields[6][0]) {
            char* cursor = fields[6];
            while (cursor && cell.image_count < MAX_MEDIA) {
                char* comma = strchr(cursor, ',');
                if (comma) *comma = '\0';
                cell.images[cell.image_count++] = cursor;
                cursor = comma ? comma + 1 : NULL;
            }
        }
        char json_path[1400];
        if (base->output_json_dir) {
            snprintf(json_path, sizeof(json_path), "%s/%s.json", base->output_json_dir, cell.cell_id);
            cell.output_json = json_path;
        } else {
            cell.output_json = NULL;
        }
        printf("[run ] %s\n", cell.cell_id);
        if (run_cell(&cell) != 0) errors++;
    }
    fclose(file);
    return errors;
}

int main(int argc, char** argv) {
    bench_options options;
    parse_arguments(argc, argv, &options);
    require_status(unirt_init(), "unirt_init");
    int result = options.matrix_file ? run_matrix(&options) : run_cell(&options);
    free(options.prompt);
    require_status(unirt_deinit(), "unirt_deinit");
    return result ? 1 : 0;
}
