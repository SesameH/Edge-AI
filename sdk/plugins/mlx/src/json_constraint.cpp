// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "json_constraint.h"

#include <algorithm>

#include "vendor/json.hpp"

namespace unirt::mlx_plugin {
namespace {

// Ordered, not the default sorted map: properties must be emitted in the
// order the schema declares them. Sorted keys would put a tool call's
// "arguments" before its "name", forcing the model to invent arguments before
// it has committed to which tool they belong to. llama_cpp's compiler parses
// with the same type for the same reason.
using json = nlohmann::ordered_json;

constexpr int32_t kMaxNodes = 4096;
constexpr int32_t kMaxDepth = 64;

bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool is_hex(char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

}  // namespace

struct JsonConstraint::Node {
    enum Kind : uint8_t { Object, Array, String, Number, Literal, AnyOf } kind = Object;

    // Object: declared properties, in emission order. `keys` already carries
    // the quotes and the trailing colon, so matching a key is a plain literal
    // compare against the byte stream.
    std::vector<std::string> keys;
    std::vector<int32_t>     values;
    std::vector<uint8_t>     required;
    bool                     free_form = false;

    // Array
    int32_t item      = -1;
    int32_t min_items = 0;
    int32_t max_items = -1;

    bool integer_only = false;  // Number
    std::string literal;        // Literal
    std::vector<int32_t> options;  // AnyOf
};

struct JsonConstraint::Frame {
    int32_t node  = 0;
    uint8_t phase = 0;
    int32_t i     = 0;  // property index, item count, or nothing
    int32_t k     = 0;  // offset inside a key or literal

    bool operator==(const Frame& other) const {
        return node == other.node && phase == other.phase && i == other.i && k == other.k;
    }
    bool operator<(const Frame& other) const {
        if (node != other.node) return node < other.node;
        if (phase != other.phase) return phase < other.phase;
        if (i != other.i) return i < other.i;
        return k < other.k;
    }
};

namespace {

using Node = JsonConstraint::Node;

// A number frame may end here, so the parent can take over the next byte.
bool number_can_stop(uint8_t phase) {
    return phase == 2 || phase == 3 || phase == 5 || phase == 8;
}

class Builder {
   public:
    explicit Builder(std::vector<Node>& nodes) : nodes_(nodes) {}

    int32_t build(const json& schema, int32_t depth, std::string& error) {
        if (depth > kMaxDepth) {
            error = "schema nests deeper than " + std::to_string(kMaxDepth) + " levels";
            return -1;
        }
        if (static_cast<int32_t>(nodes_.size()) > kMaxNodes) {
            error = "schema expands to more than " + std::to_string(kMaxNodes) + " nodes";
            return -1;
        }
        if (!schema.is_object()) {
            // `true` is the permissive schema; anything else is nonsense.
            if (schema.is_boolean() && schema.get<bool>()) return any_value(error);
            error = "each schema must be a JSON object";
            return -1;
        }
        if (schema.contains("$ref")) {
            // Silently ignoring a $ref would widen the language to "any JSON",
            // which is exactly the surprise a caller using $ref cannot afford.
            error = "$ref is not supported by the MLX schema compiler";
            return -1;
        }
        if (schema.contains("const")) return literal(schema["const"].dump());
        if (schema.contains("enum")) {
            const json& values = schema["enum"];
            if (!values.is_array() || values.empty()) {
                error = "enum must be a non-empty array";
                return -1;
            }
            std::vector<int32_t> options;
            options.reserve(values.size());
            for (const json& value : values) options.push_back(literal(value.dump()));
            return any_of(options);
        }
        for (const char* key : {"anyOf", "oneOf"}) {
            if (!schema.contains(key)) continue;
            const json& branches = schema[key];
            if (!branches.is_array() || branches.empty()) {
                error = std::string(key) + " must be a non-empty array";
                return -1;
            }
            std::vector<int32_t> options;
            options.reserve(branches.size());
            for (const json& branch : branches) {
                const int32_t child = build(branch, depth + 1, error);
                if (child < 0) return -1;
                options.push_back(child);
            }
            return any_of(options);
        }

        if (!schema.contains("type")) {
            // No type but declared properties still pins the shape.
            if (schema.contains("properties")) return object(schema, depth, error);
            return any_value(error);
        }
        const json& type = schema["type"];
        if (type.is_array()) {
            std::vector<int32_t> options;
            for (const json& one : type) {
                json narrowed = schema;
                narrowed["type"] = one;
                const int32_t child = build(narrowed, depth + 1, error);
                if (child < 0) return -1;
                options.push_back(child);
            }
            return any_of(options);
        }
        if (!type.is_string()) {
            error = "type must be a string or an array of strings";
            return -1;
        }
        const std::string name = type.get<std::string>();
        if (name == "object") return object(schema, depth, error);
        if (name == "array") return array(schema, depth, error);
        if (name == "string") return simple(Node::String);
        if (name == "number" || name == "integer") {
            const int32_t id = simple(Node::Number);
            nodes_[static_cast<size_t>(id)].integer_only = (name == "integer");
            return id;
        }
        if (name == "boolean") return any_of({literal("true"), literal("false")});
        if (name == "null") return literal("null");
        error = "unsupported schema type: " + name;
        return -1;
    }

    // Any JSON value; built once and shared, since it is self-recursive: the
    // placeholder id is published before the branches are filled in so that
    // arrays and free-form objects can point back at it.
    int32_t any_value(std::string& error) {
        (void)error;
        if (any_ >= 0) return any_;
        const int32_t placeholder = add(Node::AnyOf);
        any_                      = placeholder;

        const int32_t string_id   = simple(Node::String);
        const int32_t free_object = free_form_object(string_id, any_);
        const int32_t any_array   = add(Node::Array);
        nodes_[static_cast<size_t>(any_array)].item = any_;

        nodes_[static_cast<size_t>(placeholder)].options = {
            free_object, any_array, string_id, simple(Node::Number),
            literal("true"), literal("false"), literal("null"),
        };
        return any_;
    }

    // Arbitrary string keys, arbitrary values. `values` carries the key node
    // and the value node, in that order.
    int32_t free_form_object(int32_t key_node, int32_t value_node) {
        const int32_t id = add(Node::Object);
        Node&         node = nodes_[static_cast<size_t>(id)];
        node.free_form     = true;
        node.values        = {key_node, value_node};
        return id;
    }

   private:
    int32_t add(Node::Kind kind) {
        nodes_.push_back(Node{});
        nodes_.back().kind = kind;
        return static_cast<int32_t>(nodes_.size()) - 1;
    }

    int32_t simple(Node::Kind kind) { return add(kind); }

    int32_t literal(const std::string& text) {
        const int32_t id = add(Node::Literal);
        nodes_[static_cast<size_t>(id)].literal = text;
        return id;
    }

    int32_t any_of(std::vector<int32_t> options) {
        if (options.size() == 1) return options[0];
        const int32_t id = add(Node::AnyOf);
        nodes_[static_cast<size_t>(id)].options = std::move(options);
        return id;
    }

    int32_t object(const json& schema, int32_t depth, std::string& error) {
        const json* properties = schema.contains("properties") ? &schema["properties"] : nullptr;
        if (properties && !properties->is_object()) {
            error = "properties must be an object";
            return -1;
        }
        if (!properties || properties->empty()) {
            const int32_t any = any_value(error);
            if (any < 0) return -1;
            return free_form_object(simple(Node::String), any);
        }

        std::vector<std::string> required;
        if (schema.contains("required")) {
            const json& listed = schema["required"];
            if (!listed.is_array()) {
                error = "required must be an array";
                return -1;
            }
            for (const json& name : listed) {
                if (name.is_string()) required.push_back(name.get<std::string>());
            }
        }

        // Children are built before the node is added, because building them
        // reallocates `nodes_` and would dangle a reference held across it.
        std::vector<std::string> keys;
        std::vector<int32_t>     values;
        std::vector<uint8_t>     flags;
        for (auto it = properties->begin(); it != properties->end(); ++it) {
            const int32_t child = build(it.value(), depth + 1, error);
            if (child < 0) return -1;
            keys.push_back(json(it.key()).dump() + ":");
            values.push_back(child);
            flags.push_back(std::find(required.begin(), required.end(), it.key()) !=
                            required.end());
        }
        const int32_t id = add(Node::Object);
        Node&         node = nodes_[static_cast<size_t>(id)];
        node.keys          = std::move(keys);
        node.values        = std::move(values);
        node.required      = std::move(flags);
        return id;
    }

    int32_t array(const json& schema, int32_t depth, std::string& error) {
        int32_t item = -1;
        if (schema.contains("items")) {
            const json& items = schema["items"];
            // Tuple-typed `items` (an array) would need per-position state;
            // treat the positions as alternatives rather than reject outright.
            if (items.is_array()) {
                std::vector<int32_t> options;
                for (const json& entry : items) {
                    const int32_t child = build(entry, depth + 1, error);
                    if (child < 0) return -1;
                    options.push_back(child);
                }
                item = options.empty() ? any_value(error) : any_of(options);
            } else {
                item = build(items, depth + 1, error);
            }
            if (item < 0) return -1;
        } else {
            item = any_value(error);
            if (item < 0) return -1;
        }
        const int32_t id = add(Node::Array);
        Node&         node = nodes_[static_cast<size_t>(id)];
        node.item          = item;
        node.min_items     = schema.value("minItems", 0);
        node.max_items     = schema.value("maxItems", -1);
        if (node.min_items < 0) node.min_items = 0;
        return id;
    }

    std::vector<Node>& nodes_;
    int32_t            any_ = -1;
};

}  // namespace

JsonConstraint::JsonConstraint()  = default;
JsonConstraint::~JsonConstraint() = default;

std::unique_ptr<JsonConstraint> JsonConstraint::compile(const std::string& schema_json,
                                                        std::string&       error) {
    auto    constraint = std::unique_ptr<JsonConstraint>(new JsonConstraint());
    Builder builder(constraint->nodes_);
    int32_t root = -1;
    if (schema_json.empty()) {
        root = builder.any_value(error);
    } else {
        json parsed;
        try {
            parsed = json::parse(schema_json);
        } catch (const std::exception& failure) {
            error = std::string("schema is not valid JSON: ") + failure.what();
            return nullptr;
        }
        root = builder.build(parsed, 0, error);
    }
    if (root < 0) {
        if (error.empty()) error = "schema could not be compiled";
        return nullptr;
    }
    constraint->states_.push_back(State{Frame{root, 0, 0, 0}});
    return constraint;
}

void JsonConstraint::step_frame(const State& state, char byte, States& into) const {
    if (state.empty()) return;  // the value is finished; nothing may follow
    const Frame& top  = state.back();
    const Node&  node = nodes_[static_cast<size_t>(top.node)];

    auto with_top = [&](uint8_t phase, int32_t i, int32_t k) {
        State next = state;
        next.back() = Frame{top.node, phase, i, k};
        into.push_back(std::move(next));
    };
    auto pop = [&]() {
        State next = state;
        next.pop_back();
        into.push_back(std::move(next));
    };
    // Push a child value and immediately let it consume this byte.
    auto descend = [&](uint8_t parent_phase, int32_t parent_i, int32_t child) {
        State next  = state;
        next.back() = Frame{top.node, parent_phase, parent_i, 0};
        next.push_back(Frame{child, 0, 0, 0});
        step_frame(next, byte, into);
    };

    switch (node.kind) {
        case Node::AnyOf: {
            // The choice itself consumes nothing: replace the frame with each
            // branch and let the byte decide which branches survive.
            for (int32_t option : node.options) {
                State next  = state;
                next.back() = Frame{option, 0, 0, 0};
                step_frame(next, byte, into);
            }
            return;
        }

        case Node::Literal: {
            const std::string& text = node.literal;
            if (static_cast<size_t>(top.k) >= text.size()) return;
            if (text[static_cast<size_t>(top.k)] != byte) return;
            if (static_cast<size_t>(top.k) + 1 == text.size()) {
                pop();
            } else {
                with_top(0, top.i, top.k + 1);
            }
            return;
        }

        case Node::String: {
            switch (top.phase) {
                case 0:
                    if (byte == '"') with_top(1, 0, 0);
                    return;
                case 1:
                    if (byte == '"') { pop(); return; }
                    if (byte == '\\') { with_top(2, 0, 0); return; }
                    // Control characters must be escaped; everything else,
                    // including UTF-8 continuation bytes, is literal content.
                    if (static_cast<unsigned char>(byte) >= 0x20) with_top(1, 0, 0);
                    return;
                case 2:
                    if (byte == 'u') { with_top(3, 0, 0); return; }
                    if (std::string("\"\\/bfnrt").find(byte) != std::string::npos) {
                        with_top(1, 0, 0);
                    }
                    return;
                default:
                    if (!is_hex(byte)) return;
                    if (top.k + 1 == 4) with_top(1, 0, 0);
                    else with_top(3, 0, top.k + 1);
                    return;
            }
        }

        case Node::Number: {
            switch (top.phase) {
                case 0:
                    if (byte == '-') with_top(1, 0, 0);
                    else if (byte == '0') with_top(3, 0, 0);
                    else if (is_digit(byte)) with_top(2, 0, 0);
                    return;
                case 1:
                    if (byte == '0') with_top(3, 0, 0);
                    else if (is_digit(byte)) with_top(2, 0, 0);
                    return;
                case 2:
                case 3:
                    if (top.phase == 2 && is_digit(byte)) with_top(2, 0, 0);
                    if (!node.integer_only) {
                        if (byte == '.') with_top(4, 0, 0);
                        if (byte == 'e' || byte == 'E') with_top(6, 0, 0);
                    }
                    break;
                case 4:
                    if (is_digit(byte)) with_top(5, 0, 0);
                    return;
                case 5:
                    if (is_digit(byte)) with_top(5, 0, 0);
                    if (byte == 'e' || byte == 'E') with_top(6, 0, 0);
                    break;
                case 6:
                    if (byte == '+' || byte == '-') with_top(7, 0, 0);
                    else if (is_digit(byte)) with_top(8, 0, 0);
                    return;
                case 7:
                    if (is_digit(byte)) with_top(8, 0, 0);
                    return;
                default:
                    if (is_digit(byte)) with_top(8, 0, 0);
                    break;
            }
            // A number has no closing delimiter: when the digits so far are a
            // complete number, the enclosing frame may take this byte instead.
            if (number_can_stop(top.phase)) {
                State next = state;
                next.pop_back();
                step_frame(next, byte, into);
            }
            return;
        }

        case Node::Array: {
            switch (top.phase) {
                case 0:
                    if (byte == '[') with_top(1, 0, 0);
                    return;
                case 1: {
                    if (byte == ']' && top.i >= node.min_items) { pop(); return; }
                    if (node.max_items >= 0 && top.i >= node.max_items) return;
                    descend(2, top.i + 1, node.item);
                    return;
                }
                default:
                    if (byte == ',' && (node.max_items < 0 || top.i < node.max_items)) {
                        with_top(1, top.i, 0);
                    }
                    if (byte == ']' && top.i >= node.min_items) pop();
                    return;
            }
        }

        case Node::Object: {
            if (node.free_form) {
                switch (top.phase) {
                    case 0:
                        if (byte == '{') with_top(1, 0, 0);
                        return;
                    case 1:
                        if (byte == '}' && top.i == 0) { pop(); return; }
                        descend(2, top.i + 1, node.values[0]);
                        return;
                    case 2:
                        if (byte == ':') with_top(3, top.i, 0);
                        return;
                    case 3:
                        descend(4, top.i, node.values[1]);
                        return;
                    default:
                        if (byte == ',') with_top(1, top.i, 0);
                        if (byte == '}') pop();
                        return;
                }
            }
            const int32_t count = static_cast<int32_t>(node.keys.size());
            switch (top.phase) {
                case 0:
                    if (byte == '{') with_top(1, 0, 0);
                    return;
                case 1: {
                    // Emit the next declared property, skipping optional ones,
                    // or close if nothing required is left.
                    bool all_optional = true;
                    for (int32_t index = top.i; index < count; ++index) {
                        // Every key starts with a quote, so a quote here keeps
                        // one candidate per still-emittable property alive.
                        if (byte == node.keys[static_cast<size_t>(index)][0]) {
                            with_top(2, index, 1);
                        }
                        if (node.required[static_cast<size_t>(index)]) {
                            all_optional = false;
                            break;
                        }
                    }
                    if (all_optional && byte == '}') pop();
                    return;
                }
                case 2: {
                    const std::string& key = node.keys[static_cast<size_t>(top.i)];
                    if (static_cast<size_t>(top.k) >= key.size()) return;
                    if (key[static_cast<size_t>(top.k)] != byte) return;
                    if (static_cast<size_t>(top.k) + 1 == key.size()) {
                        // Key and colon consumed; the value starts here.
                        State next  = state;
                        next.back() = Frame{top.node, 3, top.i, 0};
                        next.push_back(Frame{node.values[static_cast<size_t>(top.i)], 0, 0, 0});
                        into.push_back(std::move(next));
                    } else {
                        with_top(2, top.i, top.k + 1);
                    }
                    return;
                }
                default: {
                    if (byte == ',' && top.i + 1 < count) with_top(1, top.i + 1, 0);
                    if (byte == '}') {
                        bool all_optional = true;
                        for (int32_t index = top.i + 1; index < count; ++index) {
                            if (node.required[static_cast<size_t>(index)]) {
                                all_optional = false;
                                break;
                            }
                        }
                        if (all_optional) pop();
                    }
                    return;
                }
            }
        }
    }
}

bool JsonConstraint::advance(const States& from, char byte, States& into) const {
    into.clear();
    for (const State& state : from) step_frame(state, byte, into);
    std::sort(into.begin(), into.end());
    into.erase(std::unique(into.begin(), into.end()), into.end());
    return !into.empty();
}

bool JsonConstraint::complete(const State& state) const {
    if (state.empty()) return true;
    const Frame& top = state.back();
    if (nodes_[static_cast<size_t>(top.node)].kind != Node::Number) return false;
    if (!number_can_stop(top.phase)) return false;
    return complete(State(state.begin(), state.end() - 1));
}

bool JsonConstraint::allows(const std::string& bytes) const {
    States current = states_;
    States next;
    for (char byte : bytes) {
        if (!advance(current, byte, next)) return false;
        current.swap(next);
    }
    return true;
}

bool JsonConstraint::accept(const std::string& bytes) {
    States current = states_;
    States next;
    for (char byte : bytes) {
        if (!advance(current, byte, next)) return false;
        current.swap(next);
    }
    states_.swap(current);
    return true;
}

std::string JsonConstraint::state_signature() const {
    std::string signature;
    signature.reserve(states_.size() * 16);
    for (const State& state : states_) {
        for (const Frame& frame : state) {
            const int32_t fields[] = {frame.node, static_cast<int32_t>(frame.phase), frame.i,
                                      frame.k};
            signature.append(reinterpret_cast<const char*>(fields), sizeof(fields));
        }
        signature.push_back('\x1f');  // states are sorted, so this is canonical
    }
    return signature;
}

bool JsonConstraint::can_stop() const {
    for (const State& state : states_) {
        if (complete(state)) return true;
    }
    return false;
}

}  // namespace unirt::mlx_plugin
