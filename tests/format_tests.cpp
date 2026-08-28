#include "formats.h"
#include "metrics.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
namespace fs=std::filesystem;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (0)
template<class F> static bool throws(F fn){try{fn();return false;}catch(...){return true;}}
template<class F> static bool throws_with(F fn,const std::string&text){try{fn();return false;}catch(const std::exception&e){return std::string(e.what()).find(text)!=std::string::npos;}}
static rq::RqlHeader h(uint32_t vocab=4,uint32_t version=2){rq::RqlHeader x;x.schema_version=version;x.header_size=version==1?rq::RQL_V1_HEADER_SIZE:rq::RQL_V2_HEADER_SIZE;x.vocab_size=vocab;x.positions=2;x.model_sha256.fill(8);x.token_sha256.fill(7);x.model_id="unit";x.llama_commit=std::string(40,'a');if(version==2){x.n_ctx=4096;x.n_batch=512;x.n_ubatch=512;x.kv_k_type=rq::KV_F16;x.kv_v_type=rq::KV_F16;x.gpu_layers=-1;x.flash_attention=rq::FLASH_ENABLED;x.tokenizer_add_special=true;x.tokenizer_parse_special=false;x.sampling=rq::SAMPLING_NONE;x.has_execution_contract=true;}return x;}
static void write(const std::string&p,const rq::RqlHeader&h){float x[]={-1,9,4,3};rq::RqlWriter w(p,h);w.write(x);float y[]={0,1,2,3};w.write(y);w.close();}
int main(){auto d=fs::temp_directory_path()/"ramgpt-format-tests";fs::create_directories(d);auto tok=(d/"a.tokens").string();std::vector<int32_t>ids={1,2,-3,2147483647};rq::write_tokens(tok,ids);CHECK(rq::read_tokens(tok)==ids);CHECK(rq::hex(rq::sha256_file(tok))=="2eff7759ccbce5afc2829fe921ca039b53de2f07648b70d609a0ec067f894d6f");CHECK(rq::RQL_V1_HEADER_SIZE==233&&rq::RQL_V2_HEADER_SIZE==321);
float x[]={-1,9,4,3};auto t=rq::top2(x,4);CHECK(t.first==1&&t.second==2);
auto v1=(d/"v1.rql").string(),a=(d/"v2.rql").string(),b=(d/"v2-copy.rql").string();write(v1,h(4,1));write(a,h());fs::copy_file(a,b,fs::copy_options::overwrite_existing);
{rq::RqlReader r(v1);CHECK(r.header().schema_version==1&&!r.header().has_execution_contract);}
{rq::RqlReader r(a);auto&z=r.header();CHECK(z.schema_version==2&&z.header_size==321&&z.model_sha256[0]==8&&z.n_ctx==4096&&z.n_batch==512&&z.n_ubatch==512&&z.gpu_layers==-1&&z.flash_attention==rq::FLASH_ENABLED&&z.tokenizer_add_special&&!z.tokenizer_parse_special);std::vector<float>row;CHECK(r.read(row));CHECK(!memcmp(row.data(),x,sizeof x));CHECK(r.read(row));CHECK(!r.read(row));r.require_finished();}
auto c=rq::compare(a,b,nullptr,rq::CompatibilityMode::Strict);CHECK(c.positions==2&&c.changed_top1==0&&c.max_abs==0);
auto mismatch=(d/"mismatch.rql").string();auto mh=h();mh.n_batch=256;mh.n_ubatch=256;write(mismatch,mh);CHECK(throws_with([&]{rq::compare(a,mismatch,nullptr,rq::CompatibilityMode::Strict);},"n_batch"));CHECK(rq::compare(a,mismatch).positions==2);CHECK(throws_with([&]{rq::compare(v1,a,nullptr,rq::CompatibilityMode::Strict);},"schema_version"));
auto corpus=(d/"corpus.rql").string();mh=h();mh.token_sha256[0]=9;write(corpus,mh);CHECK(throws([&]{rq::compare(a,corpus);}));
auto malformed=(d/"malformed.rql").string();fs::copy_file(a,malformed,fs::copy_options::overwrite_existing);{std::fstream f(malformed,std::ios::binary|std::ios::in|std::ios::out);f.seekp(12);char bad[4]={0,0,0,0};f.write(bad,4);}CHECK(throws([&]{rq::RqlReader r(malformed);}));
auto trunc=(d/"trunc.rql").string();{std::ifstream i(a,std::ios::binary);std::ofstream o(trunc,std::ios::binary);std::vector<char>q(330);i.read(q.data(),q.size());o.write(q.data(),i.gcount());}CHECK(throws([&]{rq::RqlReader r(trunc);std::vector<float>z;while(r.read(z)){};}));
auto m=rq::calculate_metrics(x,x,4,2);CHECK(m.ref_top1==1&&m.ref_top2==2&&!m.decision_flip&&m.kl==0&&m.delta_nll==0);double pv[]={1,2,3,4,5};CHECK(rq::percentile_sorted(pv,5,.5)==3);std::cout<<"v1="<<v1<<"\nv2="<<a<<"\nall format and metric tests passed\n";}
