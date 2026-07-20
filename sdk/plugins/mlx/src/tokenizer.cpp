// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "vendor/json.hpp"

namespace unirt::mlx_plugin {

namespace {

// GPT-2 byte-level encoding: maps each raw byte to a printable unicode
// codepoint so tokens are valid UTF-8 strings inside tokenizer.json.
const std::unordered_map<uint8_t, std::string>& byte_encoder() {
    static const auto* enc = [] {
        auto* m = new std::unordered_map<uint8_t, std::string>();
        auto  to_utf8 = [](uint32_t cp) {
            std::string s;
            if (cp < 0x80) {
                s += static_cast<char>(cp);
            } else if (cp < 0x800) {
                s += static_cast<char>(0xC0 | (cp >> 6));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                s += static_cast<char>(0xE0 | (cp >> 12));
                s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            }
            return s;
        };
        // Printable bytes map to themselves; the rest get 256+n codepoints.
        auto printable = [](int b) {
            return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        };
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = printable(b) ? static_cast<uint32_t>(b) : 256 + n++;
            (*m)[static_cast<uint8_t>(b)] = to_utf8(cp);
        }
        return m;
    }();
    return *enc;
}

const std::unordered_map<std::string, uint8_t>& byte_decoder() {
    static const auto* dec = [] {
        auto* m = new std::unordered_map<std::string, uint8_t>();
        for (const auto& [b, s] : byte_encoder()) (*m)[s] = b;
        return m;
    }();
    return *dec;
}

bool is_ascii_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

enum class CharClass { letter, number, space, punctuation };

size_t utf8_length(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

uint32_t decode_codepoint(const std::string& text, size_t offset, size_t& length) {
    const unsigned char first = static_cast<unsigned char>(text[offset]);
    length = std::min(utf8_length(first), text.size() - offset);
    if (length == 1) return first;
    uint32_t cp = first & ((1u << (7 - length)) - 1u);
    for (size_t i = 1; i < length; ++i) {
        const unsigned char next = static_cast<unsigned char>(text[offset + i]);
        if ((next & 0xC0) != 0x80) {
            length = 1;
            return first;
        }
        cp = (cp << 6) | (next & 0x3F);
    }
    return cp;
}

bool in_range(uint32_t value, uint32_t first, uint32_t last) {
    return value >= first && value <= last;
}

CharClass char_class(const std::string& text, size_t offset, size_t& length) {
    const unsigned char byte = static_cast<unsigned char>(text[offset]);
    if (byte < 0x80) {
        length = 1;
        if (is_ascii_letter(byte)) return CharClass::letter;
        if (is_ascii_digit(byte)) return CharClass::number;
        if (is_space(byte)) return CharClass::space;
        return CharClass::punctuation;
    }

    const uint32_t cp = decode_codepoint(text, offset, length);
    if (cp == 0x00A0 || in_range(cp, 0x2000, 0x200A) || cp == 0x2028 || cp == 0x2029 ||
        cp == 0x202F || cp == 0x205F || cp == 0x3000) {
        return CharClass::space;
    }
    if (in_range(cp, 0x0660, 0x0669) || in_range(cp, 0x06F0, 0x06F9) ||
        in_range(cp, 0x0966, 0x096F) || in_range(cp, 0xFF10, 0xFF19)) {
        return CharClass::number;
    }
    if (in_range(cp, 0x2000, 0x206F) || in_range(cp, 0x20A0, 0x27BF) ||
        in_range(cp, 0x2E00, 0x2E7F) || in_range(cp, 0x3000, 0x303F) ||
        in_range(cp, 0xFE10, 0xFE1F) || in_range(cp, 0xFE30, 0xFE6F) ||
        in_range(cp, 0xFF00, 0xFF0F) || in_range(cp, 0xFF1A, 0xFF20) ||
        in_range(cp, 0xFF3B, 0xFF40) || in_range(cp, 0xFF5B, 0xFF65) ||
        in_range(cp, 0x1F000, 0x1FAFF)) {
        return CharClass::punctuation;
    }
    return CharClass::letter;
}

size_t contraction_length(const std::string& text, size_t offset) {
    if (offset >= text.size() || text[offset] != '\'') return 0;
    static const char* suffixes[] = {"re", "ve", "ll", "s", "d", "m", "t"};
    for (const char* suffix : suffixes) {
        const size_t length = std::char_traits<char>::length(suffix);
        if (offset + 1 + length > text.size()) continue;
        bool match = true;
        for (size_t i = 0; i < length; ++i) {
            const unsigned char value = static_cast<unsigned char>(text[offset + 1 + i]);
            if (static_cast<char>(std::tolower(value)) != suffix[i]) {
                match = false;
                break;
            }
        }
        if (match) return length + 1;
    }
    return 0;
}

}  // namespace

void BpeTokenizer::load(const std::string& tokenizer_json_path) {
    vocab_.clear();
    id_to_token_.clear();
    merge_rank_.clear();
    special_tokens_.clear();
    special_ids_.clear();
    bos_id_ = -1;
    eos_id_ = -1;
    individual_digits_ = false;

    std::error_code size_error;
    const auto tokenizer_size = std::filesystem::file_size(tokenizer_json_path, size_error);
    if (size_error || tokenizer_size > 512 * 1024 * 1024) {
        throw std::runtime_error("tokenizer.json is missing or unreasonably large");
    }
    std::ifstream f(tokenizer_json_path);
    if (!f) throw std::runtime_error("cannot open " + tokenizer_json_path);
    nlohmann::json j;
    f >> j;

    if (j.contains("normalizer") && !j["normalizer"].is_null()) {
        throw std::runtime_error("tokenizer normalizers are not implemented by the MLX plugin");
    }
    if (j.contains("post_processor") && !j["post_processor"].is_null()) {
        throw std::runtime_error("tokenizer post-processors are not implemented by the MLX plugin");
    }
    const auto validate_byte_level = [](const nlohmann::json& value) {
        if (!value.is_object() || value.value("type", "") != "ByteLevel" ||
            value.value("add_prefix_space", false) || !value.value("use_regex", true)) {
            throw std::runtime_error("unsupported ByteLevel pre-tokenizer configuration");
        }
    };
    const auto& pre = j.at("pre_tokenizer");
    if (pre.value("type", "") == "ByteLevel") {
        validate_byte_level(pre);
    } else if (pre.value("type", "") == "Sequence") {
        const auto& items = pre.at("pretokenizers");
        if (!items.is_array() || items.size() != 2 || items[0].value("type", "") != "Digits" ||
            !items[0].value("individual_digits", false)) {
            throw std::runtime_error("unsupported pre-tokenizer sequence");
        }
        individual_digits_ = true;
        validate_byte_level(items[1]);
    } else {
        throw std::runtime_error("MLX tokenizer requires a ByteLevel pre-tokenizer");
    }
    if (!j.contains("decoder") || j["decoder"].value("type", "") != "ByteLevel") {
        throw std::runtime_error("MLX tokenizer requires a ByteLevel decoder");
    }

    const auto& model = j.at("model");
    if (model.at("type").get<std::string>() != "BPE") {
        throw std::runtime_error("tokenizer model type is not BPE");
    }
    if (!model.value("dropout", nlohmann::json(nullptr)).is_null() ||
        !model.value("unk_token", nlohmann::json(nullptr)).is_null() ||
        !model.value("continuing_subword_prefix", nlohmann::json(nullptr)).is_null() ||
        !model.value("end_of_word_suffix", nlohmann::json(nullptr)).is_null() ||
        model.value("byte_fallback", false) || model.value("ignore_merges", false)) {
        throw std::runtime_error("unsupported BPE tokenizer options");
    }

    const auto& vocab = model.at("vocab");
    if (!vocab.is_object() || vocab.empty()) {
        throw std::runtime_error("tokenizer BPE vocabulary is empty or invalid");
    }
    int32_t     max_id = -1;
    std::unordered_set<int32_t> ids;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const int64_t raw_id = it.value().get<int64_t>();
        if (raw_id < 0 || raw_id > 10'000'000 || raw_id > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("tokenizer vocabulary id is out of range");
        }
        const int32_t id = static_cast<int32_t>(raw_id);
        if (!ids.insert(id).second) throw std::runtime_error("duplicate tokenizer vocabulary id");
        max_id = std::max(max_id, id);
    }
    if (ids.size() != static_cast<size_t>(max_id) + 1) {
        throw std::runtime_error("tokenizer base vocabulary ids must be dense");
    }
    id_to_token_.assign(static_cast<size_t>(max_id) + 1, "");
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int32_t id  = it.value().get<int32_t>();
        vocab_[it.key()]            = id;
        id_to_token_[static_cast<size_t>(id)] = it.key();
    }

    const auto& merges = model.at("merges");
    if (!merges.is_array()) throw std::runtime_error("tokenizer BPE merges must be an array");
    int32_t     rank   = 0;
    for (const auto& m : merges) {
        // merges appear either as "a b" strings or ["a","b"] pairs
        std::string key = m.is_string() ? m.get<std::string>()
                                        : m.at(0).get<std::string>() + " " + m.at(1).get<std::string>();
        if (!merge_rank_.emplace(std::move(key), rank).second) {
            throw std::runtime_error("duplicate tokenizer BPE merge");
        }
        if (rank == std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("too many tokenizer BPE merges");
        }
        ++rank;
    }

    if (j.contains("added_tokens")) {
        for (const auto& t : j["added_tokens"]) {
            std::string content = t.at("content").get<std::string>();
            int32_t     id      = t.at("id").get<int32_t>();
            if (!t.value("special", false) || t.value("single_word", false) ||
                t.value("lstrip", false) || t.value("rstrip", false) ||
                t.value("normalized", false)) {
                throw std::runtime_error("unsupported added-token matching options");
            }
            if (content.empty() || id < 0 || id > 10'000'000) {
                throw std::runtime_error("invalid added token");
            }
            auto existing_content = vocab_.find(content);
            if (existing_content != vocab_.end() && existing_content->second != id) {
                throw std::runtime_error("added token content collides with the base vocabulary");
            }
            if (static_cast<size_t>(id) < id_to_token_.size() &&
                !id_to_token_[static_cast<size_t>(id)].empty() &&
                id_to_token_[static_cast<size_t>(id)] != content) {
                throw std::runtime_error("added token id collides with the base vocabulary");
            }
            if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(id + 1, "");
            id_to_token_[static_cast<size_t>(id)] = content;
            vocab_[content]                       = id;
            special_ids_[id]                      = true;
            special_tokens_.emplace_back(std::move(content), id);
            // Longer specials first so <|im_start|> wins over shorter overlaps.
        }
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    auto find_id = [&](const char* s) -> int32_t {
        auto it = vocab_.find(s);
        return it == vocab_.end() ? -1 : it->second;
    };
    // SmolLM2/ChatML convention; fall back to GPT-2 style endoftext.
    bos_id_ = find_id("<|im_start|>");
    eos_id_ = find_id("<|im_end|>");
    if (eos_id_ < 0) eos_id_ = find_id("<|endoftext|>");

}

std::vector<int32_t> BpeTokenizer::bpe_segment(const std::string& piece) const {
    // Split the byte-encoded piece into UTF-8 "symbols", then merge greedily
    // by rank until no merge applies.
    std::vector<std::string> parts;
    for (size_t i = 0; i < piece.size();) {
        size_t        len = 1;
        unsigned char c   = piece[i];
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        parts.push_back(piece.substr(i, len));
        i += len;
    }

    while (parts.size() > 1) {
        int32_t best_rank = INT32_MAX;
        size_t  best_i    = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            auto it = merge_rank_.find(parts[i] + " " + parts[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i    = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        parts[best_i] = parts[best_i] + parts[best_i + 1];
        parts.erase(parts.begin() + best_i + 1);
    }

    std::vector<int32_t> out;
    out.reserve(parts.size());
    for (const auto& p : parts) {
        auto it = vocab_.find(p);
        if (it != vocab_.end()) {
            out.push_back(it->second);
        } else {
            // Unknown merged symbol: fall back to its individual byte-level
            // characters, each of which is guaranteed a vocab entry.
            for (size_t i = 0; i < p.size();) {
                size_t        len = 1;
                unsigned char c   = p[i];
                if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
                auto bit = vocab_.find(p.substr(i, len));
                if (bit == vocab_.end()) {
                    throw std::runtime_error("tokenizer cannot encode a byte-level symbol");
                }
                out.push_back(bit->second);
                i += len;
            }
        }
    }
    return out;
}

std::vector<int32_t> BpeTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;

    // 1) split out special tokens verbatim
    std::vector<std::pair<std::string, int32_t>> segments;  // (text, special_id or -1)
    size_t pos = 0;
    while (pos < text.size()) {
        size_t  next_special = std::string::npos;
        int32_t special_id   = -1;
        size_t  special_len  = 0;
        for (const auto& [tok, id] : special_tokens_) {
            size_t p = text.find(tok, pos);
            if (p != std::string::npos && (next_special == std::string::npos || p < next_special)) {
                next_special = p;
                special_id   = id;
                special_len  = tok.size();
            }
        }
        if (next_special == std::string::npos) {
            segments.emplace_back(text.substr(pos), -1);
            break;
        }
        if (next_special > pos) segments.emplace_back(text.substr(pos, next_special - pos), -1);
        segments.emplace_back("", special_id);
        pos = next_special + special_len;
    }

    // 2) per text segment: simplified GPT-2 pre-tokenization, then BPE
    for (const auto& [seg, sid] : segments) {
        if (sid >= 0) {
            ids.push_back(sid);
            continue;
        }
        auto append_piece = [&](const std::string& word) {
            std::string byte_level;
            for (unsigned char b : word) byte_level += byte_encoder().at(b);
            auto seg_ids = bpe_segment(byte_level);
            ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
        };

        size_t i = 0;
        while (i < seg.size()) {
            size_t start = i;
            const size_t contraction = contraction_length(seg, i);
            if (contraction > 0) {
                i += contraction;
                append_piece(seg.substr(start, i - start));
                continue;
            }

            // GPT-2's regex attaches only the final ASCII space to a following
            // word/punctuation run. Digits split by the preceding Digits
            // pre-tokenizer keep all preceding spaces separate.
            if (seg[i] == ' ') {
                size_t end = i;
                while (end < seg.size() && seg[end] == ' ') ++end;
                size_t next_len = 0;
                const bool attach =
                    end < seg.size() &&
                    !(individual_digits_ && char_class(seg, end, next_len) == CharClass::number) &&
                    contraction_length(seg, end) == 0;
                if (!attach) {
                    i = end;
                    append_piece(seg.substr(start, i - start));
                    continue;
                }
                if (end - start > 1) {
                    i = end - 1;
                    append_piece(seg.substr(start, i - start));
                    continue;
                }
                i = end;
            }

            size_t char_len = 0;
            const CharClass cls = char_class(seg, i, char_len);
            if (cls == CharClass::space) {
                do {
                    i += char_len;
                    if (i >= seg.size()) break;
                } while (char_class(seg, i, char_len) == CharClass::space);
            } else if (cls == CharClass::number) {
                int count = 0;
                do {
                    i += char_len;
                    ++count;
                    if (individual_digits_ || count == 3 || i >= seg.size()) break;
                } while (char_class(seg, i, char_len) == CharClass::number);
            } else {
                do {
                    i += char_len;
                    if (i >= seg.size()) break;
                } while (char_class(seg, i, char_len) == cls);
            }
            append_piece(seg.substr(start, i - start));
        }
    }
    return ids;
}

std::string BpeTokenizer::decode_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "";
    const std::string& tok = id_to_token_[static_cast<size_t>(id)];
    if (special_ids_.count(id)) return tok;
    // map byte-level codepoints back to raw bytes
    std::string out;
    const auto& dec = byte_decoder();
    for (size_t i = 0; i < tok.size();) {
        size_t        len = 1;
        unsigned char c   = tok[i];
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        auto it = dec.find(tok.substr(i, len));
        if (it != dec.end()) out += static_cast<char>(it->second);
        i += len;
    }
    return out;
}

std::string BpeTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += decode_piece(id);
    return out;
}

}  // namespace unirt::mlx_plugin
