// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "vendor/json.hpp"

namespace {

bool expect_tokens(
    const unirt::mlx_plugin::BpeTokenizer& tokenizer, const std::string& text,
    const std::vector<int32_t>& expected) {
    const auto actual = tokenizer.encode(text);
    if (actual == expected) return true;
    std::cerr << "token mismatch for: " << text << "\nexpected:";
    for (const auto token : expected) std::cerr << ' ' << token;
    std::cerr << "\nactual:";
    for (const auto token : actual) std::cerr << ' ' << token;
    std::cerr << '\n';
    return false;
}

std::string to_utf8(uint32_t cp) {
    std::string value;
    if (cp < 0x80) {
        value += static_cast<char>(cp);
    } else if (cp < 0x800) {
        value += static_cast<char>(0xC0 | (cp >> 6));
        value += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        value += static_cast<char>(0xE0 | (cp >> 12));
        value += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        value += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return value;
}

std::vector<std::string> byte_symbols() {
    std::vector<int> bytes;
    for (int value = '!'; value <= '~'; ++value) bytes.push_back(value);
    for (int value = 0xA1; value <= 0xAC; ++value) bytes.push_back(value);
    for (int value = 0xAE; value <= 0xFF; ++value) bytes.push_back(value);
    std::vector<int> codepoints = bytes;
    int next = 0;
    for (int value = 0; value < 256; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
            bytes.push_back(value);
            codepoints.push_back(256 + next++);
        }
    }
    std::vector<std::string> result(256);
    for (size_t i = 0; i < bytes.size(); ++i) {
        result[static_cast<size_t>(bytes[i])] = to_utf8(static_cast<uint32_t>(codepoints[i]));
    }
    return result;
}

std::filesystem::path write_synthetic_fixture() {
    using nlohmann::json;
    const auto symbols = byte_symbols();
    json vocab = json::object();
    for (int id = 0; id < 256; ++id) vocab[symbols[static_cast<size_t>(id)]] = id;
    vocab["12"] = 256;
    vocab["123"] = 257;
    vocab["'m"] = 258;
    vocab[symbols[32] + symbols[32]] = 259;

    json fixture = {
        {"normalizer", nullptr},
        {"post_processor", nullptr},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "Digits"}, {"individual_digits", true}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", true},
             {"use_regex", true}}}}}},
        {"decoder", {{"type", "ByteLevel"}}},
        {"added_tokens", json::array()},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", nullptr},
          {"end_of_word_suffix", nullptr},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", false},
          {"vocab", vocab},
          {"merges",
           {symbols['1'] + " " + symbols['2'], "12 " + symbols['3'],
            symbols['\''] + " " + symbols['m'], symbols[32] + " " + symbols[32]}}}}};

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("unirt-mlx-tokenizer-test-" + std::to_string(unique) + ".json");
    std::ofstream stream(path);
    stream << fixture;
    return path;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto synthetic_path = write_synthetic_fixture();
        unirt::mlx_plugin::BpeTokenizer tokenizer;
        tokenizer.load(synthetic_path.string());
        bool ok = true;
        ok &= expect_tokens(tokenizer, "123", {'1', '2', '3'});
        ok &= expect_tokens(tokenizer, "I'm", {'I', 258});
        ok &= expect_tokens(tokenizer, "  a", {' ', ' ', 'a'});
        std::filesystem::remove(synthetic_path);

        if (argc == 2 && std::filesystem::is_regular_file(argv[1])) {
            tokenizer.load(argv[1]);
            ok &= expect_tokens(tokenizer, "123", {33, 34, 35});
            ok &= expect_tokens(
                tokenizer, "I'm we're don't", {57, 5248, 392, 2316, 1326, 982});
            ok &= expect_tokens(tokenizer, "hello  world\n", {28120, 216, 905, 198});
            ok &= expect_tokens(
                tokenizer, "你好，世界🙂",
                {18645, 250, 48392, 138, 12831, 7906, 240, 178, 239, 230, 10813, 38887});
        }
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "tokenizer validation failed: " << error.what() << '\n';
        return 1;
    }
}
