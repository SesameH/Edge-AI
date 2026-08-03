// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <llama.h>

#include "llama_ptr.h"
#include "weight_cache.h"

namespace unirt::llama_plugin {

/**
 * One llama_context shared by several handles, so their decode steps travel
 * together in a single batch.
 *
 * Why this exists. A slot pool used to be N contexts decoding independently,
 * which is concurrency but not throughput: on a small model each step is a
 * pass over the weights, and doing that pass four times for four tokens costs
 * four times as much as doing it once for four tokens. The pass is bandwidth-
 * bound and the arithmetic per token is trivial, so the second, third and
 * fourth sequence in a batch are close to free -- prefill measures 12-36x the
 * decode rate on the same weights, and that gap is exactly the headroom a
 * batch reclaims.
 *
 * How it works. There is no worker thread and nothing is handed off. Every
 * caller thread stays on its own request for the whole of it -- prompt,
 * sampler, grammar, stop sequences, streaming callback -- and comes here only
 * for the one operation that cannot be done concurrently: llama_decode. The
 * first thread to arrive at an idle engine drives a *round*: it takes every
 * submission queued at that moment, packs them into one llama_batch, decodes,
 * hands each participant back the logits belonging to its own last token, and
 * wakes everyone. Threads that arrive during a round simply join the next one.
 * That is continuous batching: sequences enter and leave between rounds
 * without anything being drained or restarted.
 *
 * What this buys the design. Sampling, grammar state, stop detection and the
 * user's token callback all stay on the caller's thread, where they already
 * were, and none of them can block another sequence -- a slow Python callback
 * costs its own request a round, not the batch. The engine's only shared
 * mutable state is the context, and it is held under one mutex for exactly as
 * long as llama_decode runs.
 *
 * Sequences cannot crowd each other out of the KV cache. The pool is created
 * with capacity x the per-handle window that was asked for, and each handle
 * caps itself at context_size(), so every sequence can reach its own limit
 * even with all of them at theirs.
 */
class BatchEngine;
using SharedEngine = std::shared_ptr<BatchEngine>;

/** What happened to one submission. Mirrors llama_decode's return values,
 *  plus the one case that is ours rather than llama.cpp's. */
enum class DecodeStatus : int32_t {
    ok = 0,
    /** No KV slot for this sequence: evict from it and submit again. */
    no_kv_slot = 1,
    /** The ggml abort callback fired. */
    aborted = 2,
    /** llama_decode failed outright. */
    failed = 3,
    /** The round was rolled back so a batch-mate could be diagnosed alone.
     *  Nothing was decoded for this sequence and nothing was consumed --
     *  submit the identical request again. */
    retry = 4,
};

class BatchEngine {
   public:
    static constexpr int32_t kNoSequence = -1;

    BatchEngine(
        SharedModel model, ContextPtr context, int32_t capacity, int32_t context_size);
    ~BatchEngine();

    BatchEngine(const BatchEngine&)            = delete;
    BatchEngine& operator=(const BatchEngine&) = delete;

    /** The weights, shared with every other engine and handle on this file. */
    const SharedModel& model() const { return model_; }

    /** Tokens one sequence may hold. Not llama_n_ctx: that is the whole pool. */
    int32_t context_size() const { return context_size_; }
    int32_t batch_size() const { return batch_size_; }
    /** The largest batch the backend actually evaluates in one go. A logical
     *  batch bigger than this is split into several of these internally, so it
     *  is the natural unit for a prefill chunk: smaller pays round overhead
     *  for no backend benefit, larger only merges work that was going to be
     *  split anyway, into a round nobody else can join. */
    int32_t ubatch_size() const { return ubatch_size_; }
    int32_t capacity() const { return capacity_; }

    /** True when this engine can only ever have one handle on it, which is
     *  what makes reading the context outside a round safe -- there is nobody
     *  else to race with. Speculative decoding needs it: verification reads
     *  several positions of one batch's logits, and a shared engine hands back
     *  only the row for the caller's own last token. */
    bool exclusive() const { return capacity_ == 1; }

    /** The context, for the one caller that is allowed to touch it without
     *  taking a round: the sole handle of an exclusive engine. Null otherwise,
     *  so a mistake is a crash and not a data race. */
    llama_context* unshared_context() const { return exclusive() ? context_.get() : nullptr; }

    /** Take a sequence id, or kNoSequence when the engine is full. */
    int32_t claim();

    /**
     * Bracket a generation. Between join() and leave() this sequence counts as
     * one the engine should expect a submission from, which is what lets a
     * round wait for its batch-mates instead of decoding the instant the first
     * one arrives -- without it the thread that ran the previous round samples,
     * comes back and finds the engine idle, and the sequences degenerate into
     * taking turns at one token each, which is slower than not sharing at all.
     */
    void join(int32_t sequence);
    void leave(int32_t sequence);

    /**
     * Tell the engine where to find this sequence's transcript, so another
     * sequence can look for a prefix worth borrowing. The vector belongs to
     * the handle and is only ever written by the handle's own thread -- the
     * engine reads it under its lock, and only for a sequence that is not
     * between join() and leave(), which is exactly when nothing is writing.
     */
    void register_transcript(int32_t sequence, const std::vector<llama_token>* transcript);

    /**
     * Take another sequence's cached prefix instead of evaluating it again.
     *
     * Four clients arriving with the same system prompt is the ordinary shape
     * of serving, and each of them prefilling it costs the same work over
     * again -- measured 8.05 s for four against 1.88 s for one, on a
     * 1854-token prompt. With a unified cache llama_memory_seq_cp does not
     * copy anything: it adds this sequence's id to cells that are already
     * there, so the prefix is evaluated once and held once however many
     * sequences read it.
     *
     * `already` is what this sequence could reuse from its own cache; a donor
     * only wins if it beats that. Returns how many leading tokens of `wanted`
     * are now in this sequence's KV, or 0 when nothing was borrowed (in which
     * case the sequence's own cache is untouched).
     */
    size_t borrow_prefix(
        int32_t sequence, const std::vector<llama_token>& wanted, size_t already);

    /**
     * Leading tokens of `sequence` whose cells another sequence may also be
     * pointing at. Context shifting must not touch them: llama.cpp shifts a
     * cell's position for every sequence holding it, so evicting inside a
     * shared region would silently renumber somebody else's conversation.
     */
    int32_t shared_prefix(int32_t sequence) const;

    /** Give one back, dropping whatever it had in the KV cache. */
    void release(int32_t sequence);

    /**
     * Decode `count` tokens of `sequence` starting at position `pos0`,
     * batched with whatever else is waiting.
     *
     * When `logits` is non-null it receives the model's scores for the last
     * token of the submission, valid until this sequence submits again (the
     * row is copied out when more than one handle is live, so another
     * sequence's round cannot overwrite it under the reader).
     */
    DecodeStatus decode(
        int32_t sequence, const llama_token* tokens, int32_t count, llama_pos pos0,
        const float** logits);

    /**
     * Exclusive use of the context for something that is not a decode --
     * trimming a sequence's KV, saving it, measuring it. Blocks until any
     * round in flight has finished; hold it briefly.
     */
    class Access {
       public:
        Access(BatchEngine& engine, std::unique_lock<std::mutex> lock)
            : engine_(engine), lock_(std::move(lock)) {}

        llama_context* context() const { return engine_.context_.get(); }
        llama_memory_t memory() const { return llama_get_memory(engine_.context_.get()); }

        /** This sequence's KV now ends at `length`, so nothing past there can
         *  still be shared with anybody. Call after any trim, or the sequence
         *  keeps refusing to evict a region it no longer holds. */
        void truncated(int32_t sequence, int32_t length) { engine_.forget_shared(sequence, length); }

       private:
        BatchEngine&                 engine_;
        std::unique_lock<std::mutex> lock_;
    };

    Access access();

   private:
    struct Submission {
        int32_t            sequence   = 0;
        const llama_token* tokens     = nullptr;
        int32_t            count      = 0;
        llama_pos          pos0       = 0;
        bool               want_logits = false;
        // Filled in by whichever thread ran the round this joined.
        int32_t      output_index = -1;
        DecodeStatus status       = DecodeStatus::retry;
        const float* logits       = nullptr;
        bool         served       = false;
    };

    // Called with the lock held; releases nothing. Runs one round over the
    // submissions queued right now and marks each of them served.
    void run_round(std::unique_lock<std::mutex>& lock);
    // Lock already held.
    void forget_shared(int32_t sequence, int32_t length);
    // Pack `participants` into the batch and decode. Returns llama_decode's code.
    int32_t attempt(const std::vector<Submission*>& participants);
    void    publish(const std::vector<Submission*>& participants, int32_t code);

    SharedModel model_;
    ContextPtr  context_;
    llama_batch batch_{};
    int32_t     capacity_     = 1;
    int32_t     context_size_ = 0;
    int32_t     batch_size_   = 0;
    int32_t     ubatch_size_  = 0;
    int32_t     vocab_size_   = 0;

    mutable std::mutex       mutex_;
    std::condition_variable  round_;
    std::vector<Submission*> queue_;
    std::vector<bool>        claimed_;
    // Inside a generation: worth waiting for at round time, and off limits as
    // a prefix donor, since its transcript is being written as we look.
    std::vector<bool>        generating_;
    // Each handle's own token transcript, borrowed by pointer. Read only for
    // a sequence that is not generating -- see register_transcript().
    std::vector<const std::vector<llama_token>*> transcripts_;
    // Leading cells of each sequence that some other sequence may also hold.
    std::vector<int32_t> shared_prefix_;
    // Sequences inside a generation, and therefore worth waiting for.
    int32_t active_ = 0;
    // How long the last decode took. A batch-mate that cannot get from
    // "served" to "submitted again" inside that is not worth holding the
    // engine for -- whatever it is doing costs more than the round saves.
    int64_t last_round_us_ = 0;
    // Rounds and the tokens in them, for the occupancy figure in the log:
    // an average near 1.0 means the sequences never met and the sharing is
    // pure overhead.
    int64_t rounds_       = 0;
    int64_t participants_ = 0;
    // Per sequence, and only used when someone else could overwrite the
    // context's own output buffer before this sequence's owner reads it.
    std::vector<std::vector<float>> logit_copies_;
    int32_t                         live_  = 0;
    bool                            busy_  = false;
};

/**
 * Join the engine that already serves this model on these settings, or start
 * one. `out_sequence` receives the claimed sequence id.
 *
 * Handles group by everything that has to match for them to share a context:
 * the file, the device, and the context geometry. The (capacity + 1)th handle
 * on one key does not wait -- it gets an engine of its own, which is exactly
 * the behaviour before batching existed.
 *
 * Returns nullptr when the context could not be created; `factory` is called
 * at most once per new engine, with no lock held on any other engine.
 */
SharedEngine acquire_engine(
    const std::string& key, int32_t capacity, int32_t context_size,
    const std::function<ContextPtr(int32_t seats)>& factory, const SharedModel& model,
    int32_t& out_sequence);

}  // namespace unirt::llama_plugin
