#pragma once
#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace rq {
constexpr uint32_t TOKEN_VERSION = 1, RQL_V1_VERSION = 1, RQL_V2_VERSION = 2;
constexpr uint32_t RQL_V1_HEADER_SIZE = 233, RQL_V2_HEADER_SIZE = 321, DTYPE_F32 = 1;
constexpr uint32_t KV_F16 = 1, FLASH_DISABLED = 0, FLASH_ENABLED = 1, SAMPLING_NONE = 0;
constexpr uint32_t RQSEG_VERSION = 1, RQSEG_HEADER_SIZE = 256, RQSEG_RECORD_HEADER_SIZE = 68;
struct RqlHeader {
    uint32_t schema_version = RQL_V2_VERSION;
    uint32_t header_size = RQL_V2_HEADER_SIZE;
    uint32_t vocab_size = 0;
    uint64_t positions = 0;
    uint32_t logits_dtype = DTYPE_F32;
    std::array<uint8_t, 32> model_sha256{};
    std::array<uint8_t, 32> token_sha256{};
    std::string model_id;
    std::string llama_commit;
    uint32_t n_ctx = 0, n_batch = 0, n_ubatch = 0;
    uint32_t kv_k_type = 0, kv_v_type = 0;
    int32_t gpu_layers = 0;
    uint32_t flash_attention = FLASH_DISABLED;
    bool tokenizer_add_special = false, tokenizer_parse_special = false;
    uint32_t sampling = SAMPLING_NONE;
    bool has_execution_contract = false;
};
enum class CompatibilityMode { Basic, Strict };
struct Comparison {
    uint64_t positions = 0, values = 0, changed_top1 = 0, changed_top2 = 0, differing_values = 0;
    uint64_t first_differing_position = UINT64_MAX;
    uint32_t first_differing_vocab_id = UINT32_MAX;
    double sum_abs = 0, sum_sq = 0;
    float max_abs = 0;
};
struct Segment { uint32_t segment_id = 0; std::string domain; std::array<uint8_t,32> token_sha256{}; std::vector<int32_t> tokens; };
struct SegmentedCorpus { uint32_t tokens_per_segment=0,burn_in_tokens=0,scored_tokens_per_segment=0;bool tokenizer_add_special=false,tokenizer_parse_special=false;std::array<uint8_t,32> model_sha256{},token_payload_sha256{};std::string model_id,llama_commit;std::vector<Segment> segments; };
void write_tokens(const std::string &, const std::vector<int32_t> &);
std::vector<int32_t> read_tokens(const std::string &);
std::array<uint8_t, 32> sha256_file(const std::string &);
std::string hex(const std::array<uint8_t, 32> &);
std::pair<int32_t, int32_t> top2(const float *, uint32_t);
SegmentedCorpus read_segmented_corpus(const std::string &);

class RqlWriter {
public:
    RqlWriter(const std::string &, const RqlHeader &);
    void write(const float * logits);
    void close();
    ~RqlWriter();
private:
    std::ofstream out_; RqlHeader h_; uint64_t written_ = 0;
};
class RqlReader {
public:
    explicit RqlReader(const std::string &);
    const RqlHeader & header() const { return h_; }
    bool read(std::vector<float> &);
    void require_finished();
private:
    std::ifstream in_; RqlHeader h_; uint64_t read_ = 0;
};
void require_compatible(const RqlHeader &, const RqlHeader &, CompatibilityMode = CompatibilityMode::Basic);
Comparison compare(const std::string &, const std::string &, std::ostream * per_position = nullptr,
                   CompatibilityMode = CompatibilityMode::Basic);
}
