// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace unirt::mlx_plugin {

// Constrains decoding to JSON that satisfies a schema.
//
// The llama_cpp plugin gets this from llama.cpp's grammar sampler, which is
// welded to that library's vocabulary type and cannot be borrowed. This is the
// MLX-side equivalent: the schema is compiled into a pushdown automaton over
// bytes, and the sampler asks it which tokens may still be appended.
//
// The automaton is nondeterministic only where the schema is: `anyOf`/`oneOf`
// branches and enum alternatives stay live in parallel until an input byte
// separates them (for tool calling, the `"name"` const does that after a few
// tokens). Everything else is a single stack.
//
// Two deliberate narrowings versus a full JSON Schema compiler:
//   * Output is compact -- no whitespace between tokens. Every JSON parser
//     accepts it, and allowing optional whitespace everywhere would multiply
//     live states for nothing.
//   * Declared properties are emitted in schema order, optional ones may be
//     skipped, and `additionalProperties` is not emitted. A schema with no
//     declared properties means a free-form object.
//
// Supported keywords: type (object/array/string/number/integer/boolean/null),
// properties, required, items, minItems, maxItems, enum, const, anyOf, oneOf.
// Anything else is ignored, which widens the language but never narrows it --
// a constrained reply can still fail a strict validator on an unsupported
// keyword, it just cannot be malformed JSON or the wrong shape.
class JsonConstraint {
   public:
    // `schema_json` empty means "any JSON value" (the enable_json path).
    // Returns nullptr and sets `error` when the schema cannot be compiled.
    static std::unique_ptr<JsonConstraint> compile(const std::string& schema_json,
                                                   std::string&       error);

    JsonConstraint();
    ~JsonConstraint();

    // May `bytes` be appended to what has been accepted so far?
    bool allows(const std::string& bytes) const;

    // Commit `bytes`. False means they were not acceptable and state is
    // unchanged; callers should treat that as a sampler bug.
    bool accept(const std::string& bytes);

    // Is the value complete, i.e. may generation stop here?
    bool can_stop() const;

    // Identifies the current automaton state exactly. Two positions with the
    // same signature admit the same tokens, which is what makes caching the
    // token mask worthwhile: every character inside a string leaves the
    // automaton in one and the same state.
    std::string state_signature() const;

    // Compiled grammar node and automaton stack frame. Public only so the
    // schema compiler in the .cpp can name them; both are opaque here.
    struct Node;
    struct Frame;

   private:
    using State  = std::vector<Frame>;
    using States = std::vector<State>;

    bool advance(const States& from, char byte, States& into) const;
    void step_frame(const State& state, char byte, States& into) const;
    bool complete(const State& state) const;

    std::vector<Node> nodes_;
    States            states_;
};

}  // namespace unirt::mlx_plugin
