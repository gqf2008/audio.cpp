#include "engine/community_models/audio8_tts/tokenizer_text.h"

#include "engine/framework/text/chinese_normalization.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::models::audio8_tts {
namespace {

int32_t require_token_id(const engine::tokenizers::LlamaBpeTokenizer & tokenizer, const std::string & token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("Audio8 TTS tokenizer missing token: " + token);
    }
    return *id;
}

}  // namespace

struct Audio8TtsTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const Audio8TtsAssets> input_assets)
        : assets(std::move(input_assets)),
          tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              {},
              {},
              assets->resources.require_file("tokenizer_config"),
              assets->resources.require_file("tokenizer_json"),
              engine::tokenizers::LlamaBpePreTokenizer::Qwen2,
          }),
          im_end(require_token_id(tokenizer, "<|im_end|>")),
          semantic_begin(static_cast<int32_t>(assets->config.semantic_start_token_id)),
          semantic_end(static_cast<int32_t>(assets->config.semantic_end_token_id)) {}

    std::shared_ptr<const Audio8TtsAssets> assets;
    engine::tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t im_end = 0;
    int32_t semantic_begin = 0;
    int32_t semantic_end = 0;
};

Audio8TtsTextTokenizer::Audio8TtsTextTokenizer(std::shared_ptr<const Audio8TtsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Audio8 TTS text tokenizer requires assets");
    }
    impl_ = std::make_shared<Impl>(std::move(assets));
}

namespace {

// Any UTF-8 sequence in the CJK ranges (and the fullwidth forms that follow
// them) is enough to pick the Chinese normalization rules.
bool contains_cjk(const std::string & text) {
    for (size_t i = 0; i < text.size(); ++i) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if (byte >= 0xE2 && byte <= 0xE9) {
            return true;
        }
    }
    return false;
}

}  // namespace

std::vector<int32_t> Audio8TtsTextTokenizer::encode(const std::string & text) const {
    // The reference pipeline normalizes text before it reaches the model:
    // without it, digits, phone numbers, currency and units are read
    // inconsistently (the same sentence reads "138" differently run to run).
    // Reuse the same rules as the index_tts2 and fireredtts3 tokenizers.
    if (contains_cjk(text)) {
        return impl_->tokenizer.encode(engine::text::normalize_chinese_text(text), true);
    }
    return impl_->tokenizer.encode(text, true);
}

int32_t Audio8TtsTextTokenizer::token_id(const std::string & token) const {
    return require_token_id(impl_->tokenizer, token);
}

int32_t Audio8TtsTextTokenizer::im_end_id() const noexcept {
    return impl_->im_end;
}

int32_t Audio8TtsTextTokenizer::semantic_begin_id() const noexcept {
    return impl_->semantic_begin;
}

int32_t Audio8TtsTextTokenizer::semantic_end_id() const noexcept {
    return impl_->semantic_end;
}

}  // namespace engine::models::audio8_tts
