// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "batch_engine.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <utility>

#include "logging.h"

namespace unirt::llama_plugin {

BatchEngine::BatchEngine(
    SharedModel model, ContextPtr context, int32_t capacity, int32_t context_size)
    : model_(std::move(model)), context_(std::move(context)), capacity_(std::max(capacity, 1)) {
    // What the caller asked each sequence to get, but never more than the
    // context can actually address. With a unified cache llama_n_ctx_seq is
    // the whole shared pool, so it is a ceiling here and not the answer.
    context_size_ = std::min(context_size, static_cast<int32_t>(llama_n_ctx_seq(context_.get())));
    batch_size_   = static_cast<int32_t>(llama_n_batch(context_.get()));
    ubatch_size_  = static_cast<int32_t>(llama_n_ubatch(context_.get()));
    vocab_size_   = llama_vocab_n_tokens(llama_model_get_vocab(model_.get()));
    // One token slot per position the batch may carry, one sequence id each:
    // a token belongs to exactly one conversation here.
    batch_ = llama_batch_init(batch_size_, 0, 1);
    claimed_.assign(static_cast<size_t>(capacity_), false);
    generating_.assign(static_cast<size_t>(capacity_), false);
    transcripts_.assign(static_cast<size_t>(capacity_), nullptr);
    shared_prefix_.assign(static_cast<size_t>(capacity_), 0);
    logit_copies_.resize(static_cast<size_t>(capacity_));
}

BatchEngine::~BatchEngine() {
    if (capacity_ > 1 && rounds_ > 0) {
        UNIRT_LOG_DEBUG(
            "llama_cpp: {} rounds, {} sequences per round on average", rounds_,
            static_cast<double>(participants_) / static_cast<double>(rounds_));
    }
    llama_batch_free(batch_);
}

void BatchEngine::join(int32_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_;
    if (sequence >= 0 && sequence < capacity_) generating_[static_cast<size_t>(sequence)] = true;
}

void BatchEngine::leave(int32_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_ > 0) --active_;
    if (sequence >= 0 && sequence < capacity_) generating_[static_cast<size_t>(sequence)] = false;
    // Somebody may be holding a round open waiting for the sequence that just
    // stopped generating.
    round_.notify_all();
}

void BatchEngine::register_transcript(
    int32_t sequence, const std::vector<llama_token>* transcript) {
    if (sequence < 0 || sequence >= capacity_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    transcripts_[static_cast<size_t>(sequence)] = transcript;
}

int32_t BatchEngine::shared_prefix(int32_t sequence) const {
    if (sequence < 0 || sequence >= capacity_) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return shared_prefix_[static_cast<size_t>(sequence)];
}

void BatchEngine::forget_shared(int32_t sequence, int32_t length) {
    if (sequence < 0 || sequence >= capacity_) return;
    auto& shared = shared_prefix_[static_cast<size_t>(sequence)];
    shared = std::min(shared, std::max(length, 0));
}

size_t BatchEngine::borrow_prefix(
    int32_t sequence, const std::vector<llama_token>& wanted, size_t already) {
    if (capacity_ < 2 || sequence < 0 || sequence >= capacity_ || wanted.size() < 2) return 0;

    std::unique_lock<std::mutex> lock(mutex_);
    round_.wait(lock, [this] { return !busy_; });

    // One token has to be left to evaluate, or there are no logits to sample
    // the reply from -- the same cap the handle's own prefix reuse applies.
    const size_t limit = wanted.size() - 1;
    int32_t      donor = kNoSequence;
    size_t       best  = already;
    for (int32_t other = 0; other < capacity_; ++other) {
        const auto index = static_cast<size_t>(other);
        if (other == sequence || !claimed_[index] || generating_[index]) continue;
        const auto* transcript = transcripts_[index];
        if (!transcript) continue;
        const size_t reach = std::min(limit, transcript->size());
        size_t       match = 0;
        while (match < reach && (*transcript)[match] == wanted[match]) ++match;
        if (match > best) {
            best  = match;
            donor = other;
        }
    }
    if (donor == kNoSequence) return 0;

    // Whatever this sequence had cached is worth less than what it is about to
    // point at, and the two cannot be spliced: the borrowed cells carry their
    // own positions from zero.
    auto* memory = llama_get_memory(context_.get());
    llama_memory_seq_rm(memory, sequence, -1, -1);
    llama_memory_seq_cp(memory, donor, sequence, 0, static_cast<llama_pos>(best));

    // Both ends now hold cells the other one can see, and neither may shift
    // them: llama.cpp moves a cell's position for every sequence that holds
    // it, so evicting inside this region would renumber the other's tokens
    // underneath it.
    shared_prefix_[static_cast<size_t>(sequence)] = static_cast<int32_t>(best);
    auto& lent = shared_prefix_[static_cast<size_t>(donor)];
    lent = std::max(lent, static_cast<int32_t>(best));

    UNIRT_LOG_DEBUG(
        "llama_cpp: sequence {} borrowed {} cached tokens from sequence {}", sequence, best,
        donor);
    return best;
}

int32_t BatchEngine::claim() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int32_t sequence = 0; sequence < capacity_; ++sequence) {
        if (claimed_[static_cast<size_t>(sequence)]) continue;
        claimed_[static_cast<size_t>(sequence)] = true;
        ++live_;
        return sequence;
    }
    return kNoSequence;
}

void BatchEngine::release(int32_t sequence) {
    if (sequence < 0 || sequence >= capacity_) return;
    std::unique_lock<std::mutex> lock(mutex_);
    round_.wait(lock, [this] { return !busy_; });
    llama_memory_seq_rm(llama_get_memory(context_.get()), sequence, -1, -1);
    if (claimed_[static_cast<size_t>(sequence)]) {
        claimed_[static_cast<size_t>(sequence)] = false;
        --live_;
    }
    transcripts_[static_cast<size_t>(sequence)]   = nullptr;
    generating_[static_cast<size_t>(sequence)]    = false;
    shared_prefix_[static_cast<size_t>(sequence)] = 0;
    logit_copies_[static_cast<size_t>(sequence)].clear();
    logit_copies_[static_cast<size_t>(sequence)].shrink_to_fit();
}

BatchEngine::Access BatchEngine::access() {
    std::unique_lock<std::mutex> lock(mutex_);
    round_.wait(lock, [this] { return !busy_; });
    return Access(*this, std::move(lock));
}

DecodeStatus BatchEngine::decode(
    int32_t sequence, const llama_token* tokens, int32_t count, llama_pos pos0,
    const float** logits) {
    if (logits) *logits = nullptr;
    if (count <= 0) return DecodeStatus::ok;

    Submission submission;
    submission.sequence    = sequence;
    submission.tokens      = tokens;
    submission.count       = count;
    submission.pos0        = pos0;
    submission.want_logits = logits != nullptr;

    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push_back(&submission);
    // Whoever finds the engine idle drives the next round, batch-mates
    // included. Nobody is designated and nobody hands off. What a driver does
    // wait for is the sequences it knows are coming: they are between rounds,
    // sampling the token they were just handed, which takes a fraction of the
    // decode that batching with them would save. The deadline is the last
    // round's own cost, so a sequence that has gone quiet for longer than a
    // decode stops being waited for and the others carry on without it.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::microseconds(last_round_us_);
    while (!submission.served) {
        if (busy_) {
            round_.wait(lock);
            continue;
        }
        if (static_cast<int32_t>(queue_.size()) >= active_ ||
            std::chrono::steady_clock::now() >= deadline) {
            run_round(lock);
            continue;
        }
        round_.wait_until(lock, deadline);
    }
    if (logits) *logits = submission.logits;
    return submission.status;
}

void BatchEngine::run_round(std::unique_lock<std::mutex>& lock) {
    (void)lock;  // held throughout: llama_decode is the exclusive section.
    busy_ = true;

    std::vector<Submission*> participants;
    int32_t                  packed = 0;
    for (auto entry = queue_.begin(); entry != queue_.end();) {
        Submission* submission = *entry;
        if (submission->count > batch_size_) {
            // Callers chunk to the batch size before submitting, so this is a
            // programming error rather than a load condition.
            UNIRT_LOG_ERROR(
                "llama_cpp: submission of {} tokens exceeds the batch size {}", submission->count,
                batch_size_);
            submission->status = DecodeStatus::failed;
            submission->served = true;
            entry              = queue_.erase(entry);
            continue;
        }
        if (packed + submission->count > batch_size_) {
            // No room this round; it stays queued and joins the next one.
            ++entry;
            continue;
        }
        participants.push_back(submission);
        packed += submission->count;
        entry = queue_.erase(entry);
    }

    if (!participants.empty()) {
        const auto started = std::chrono::steady_clock::now();
        int32_t    code    = attempt(participants);
        if (code == 1 && participants.size() > 1) {
            // "No KV slot" is reported for the batch, not for the sequence that
            // ran out, and telling the innocent ones to evict would drop turns
            // out of conversations that had room. Nothing was decoded (rc == 1
            // is refused before the cache is touched), so hand the rest back
            // unchanged and re-run the first alone, where the answer is
            // unambiguous.
            for (size_t index = 1; index < participants.size(); ++index) {
                participants[index]->status = DecodeStatus::retry;
                participants[index]->served = true;
            }
            participants.resize(1);
            code = attempt(participants);
        }
        last_round_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
        ++rounds_;
        participants_ += static_cast<int64_t>(participants.size());
        publish(participants, code);
    }

    busy_ = false;
    round_.notify_all();
}

int32_t BatchEngine::attempt(const std::vector<Submission*>& participants) {
    batch_.n_tokens = 0;
    for (Submission* submission : participants) {
        for (int32_t index = 0; index < submission->count; ++index) {
            const int32_t slot     = batch_.n_tokens++;
            batch_.token[slot]     = submission->tokens[index];
            batch_.pos[slot]       = submission->pos0 + index;
            batch_.n_seq_id[slot]  = 1;
            batch_.seq_id[slot][0] = submission->sequence;
            // Only the last token of a submission answers a question; the rest
            // are prompt being laid down, and asking for their logits would
            // cost a vocabulary-sized row each.
            batch_.logits[slot] =
                (submission->want_logits && index == submission->count - 1) ? 1 : 0;
        }
        submission->output_index = batch_.n_tokens - 1;
    }
    return llama_decode(context_.get(), batch_);
}

void BatchEngine::publish(const std::vector<Submission*>& participants, int32_t code) {
    DecodeStatus status = DecodeStatus::ok;
    if (code == 1) {
        status = DecodeStatus::no_kv_slot;
    } else if (code == 2) {
        status = DecodeStatus::aborted;
    } else if (code != 0) {
        status = DecodeStatus::failed;
    }

    // With a second handle alive, the next round can start before this one's
    // participants have finished sampling, and llama.cpp's output buffer is
    // one buffer. Copy each row out while it is still ours. On a lone handle
    // there is nobody to race and the pointer is handed back as it is.
    const bool copy = live_ > 1;
    for (Submission* submission : participants) {
        submission->status = status;
        submission->logits = nullptr;
        if (status == DecodeStatus::ok && submission->want_logits) {
            const float* row = llama_get_logits_ith(context_.get(), submission->output_index);
            if (!row) {
                submission->status = DecodeStatus::failed;
            } else if (copy && vocab_size_ > 0) {
                auto& sink = logit_copies_[static_cast<size_t>(submission->sequence)];
                sink.assign(row, row + vocab_size_);
                submission->logits = sink.data();
            } else {
                submission->logits = row;
            }
        }
        submission->served = true;
    }
}

namespace {

struct EngineCache {
    std::mutex                                 mutex;
    std::map<std::string, std::weak_ptr<BatchEngine>> engines;
};

EngineCache& cache() {
    static EngineCache instance;
    return instance;
}

}  // namespace

SharedEngine acquire_engine(
    const std::string& key, int32_t capacity, int32_t context_size,
    const std::function<ContextPtr(int32_t seats)>& factory, const SharedModel& model,
    int32_t& out_sequence) {
    out_sequence = BatchEngine::kNoSequence;
    capacity     = std::max(capacity, 1);

    auto& registry = cache();
    std::lock_guard<std::mutex> lock(registry.mutex);

    auto entry = registry.engines.find(key);
    if (entry != registry.engines.end()) {
        if (SharedEngine existing = entry->second.lock()) {
            out_sequence = existing->claim();
            if (out_sequence != BatchEngine::kNoSequence) return existing;
            // Full. The handle asking now gets a context of its own rather
            // than queueing behind a batch it cannot join, and becomes the
            // engine this key points at so its own batch-mates find it.
        }
        registry.engines.erase(entry);
    }

    // The factory sizes the KV pool for every sequence at its full window, so
    // sharing the cells cannot cost a handle any of the context it asked for.
    ContextPtr context = factory(capacity);
    if (!context) return nullptr;

    auto engine  = std::make_shared<BatchEngine>(
        model, std::move(context), capacity, context_size);
    out_sequence = engine->claim();
    if (out_sequence == BatchEngine::kNoSequence) return nullptr;
    registry.engines.emplace(key, engine);
    if (capacity > 1) {
        UNIRT_LOG_INFO(
            "llama_cpp: batching context for up to {} sequences, {} tokens each", capacity,
            engine->context_size());
    }
    return engine;
}

}  // namespace unirt::llama_plugin
