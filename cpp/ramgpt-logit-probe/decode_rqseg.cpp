#include "formats.h"
#include "llama.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
std::string arg(int n,char**v,const std::string&k){for(int i=1;i+1<n;i++)if(v[i]==k)return v[i+1];throw std::runtime_error("missing "+k);}
std::set<uint32_t> ids(const std::string&p,uint32_t n){std::ifstream f(p);if(!f)throw std::runtime_error("cannot open selected IDs");std::set<uint32_t>s;std::string x;while(std::getline(f,x)){if(x.empty())continue;size_t used=0;auto v=std::stoul(x,&used);if(used!=x.size()||v>=n||!s.insert(v).second)throw std::runtime_error("invalid selected segment ID");}return s;}
std::string clean_utf8(const std::string&s){std::string o;for(size_t i=0;i<s.size();){unsigned char c=s[i];size_t z=c<0x80?1:(c>=0xC2&&c<=0xDF?2:(c>=0xE0&&c<=0xEF?3:(c>=0xF0&&c<=0xF4?4:0)));bool ok=z&&i+z<=s.size();for(size_t j=1;ok&&j<z;j++)ok=(uint8_t(s[i+j])&0xC0)==0x80;if(ok){o.append(s,i,z);i+=z;}else{o+="\xEF\xBF\xBD";i++;}}return o;}
std::string detok(const llama_vocab*v,const std::vector<int32_t>&x,size_t a,size_t b){std::vector<llama_token>t(x.begin()+a,x.begin()+b);int n=llama_detokenize(v,t.data(),t.size(),nullptr,0,false,true);if(n>=0)throw std::runtime_error("unexpected detokenize sizing result");std::vector<char>o(-n);n=llama_detokenize(v,t.data(),t.size(),o.data(),o.size(),false,true);if(n<0)throw std::runtime_error("detokenize failed");return clean_utf8(std::string(o.data(),n));}
std::string esc(const std::string&s){std::string o;const char*h="0123456789abcdef";for(unsigned char c:s)if(c=='"'||c=='\\')o+='\\',o+=c;else if(c=='\n')o+="\\n";else if(c=='\r')o+="\\r";else if(c=='\t')o+="\\t";else if(c<32){o+="\\u00";o+=h[c>>4];o+=h[c&15];}else o+=c;return o;}
}
int main(int n,char**v){try{auto model_path=arg(n,v,"--model"),corpus_path=arg(n,v,"--segmented-corpus"),selected_path=arg(n,v,"--segment-ids"),output=arg(n,v,"--output");auto c=rq::read_segmented_corpus(corpus_path);auto wanted=ids(selected_path,c.segments.size());llama_backend_init();auto mp=llama_model_default_params();mp.vocab_only=true;std::unique_ptr<llama_model,decltype(&llama_model_free)>m(llama_model_load_from_file(model_path.c_str(),mp),llama_model_free);if(!m)throw std::runtime_error("vocabulary load failed");auto*vocab=llama_model_get_vocab(m.get());std::ofstream o(output);if(!o)throw std::runtime_error("cannot create decoded output");for(auto id:wanted){auto&s=c.segments[id];auto emit=[&](const char*name,size_t a,size_t b){o<<"\""<<name<<"\":{\"token_start\":"<<a<<",\"token_end_exclusive\":"<<b<<",\"text\":\""<<esc(detok(vocab,s.tokens,a,b))<<"\"}";};o<<"{\"segment_id\":"<<id<<",\"domain\":\""<<esc(s.domain)<<"\",";emit("beginning",0,40);o<<",";emit("boundary",246,266);o<<",";emit("middle",364,404);o<<",";emit("end",728,768);o<<",";emit("full",0,s.tokens.size());o<<",\"token_pieces\":[";for(size_t i=0;i<s.tokens.size();++i){if(i)o<<",";o<<"\""<<esc(detok(vocab,s.tokens,i,i+1))<<"\"";}o<<"]}\n";}m.reset();llama_backend_free();return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
