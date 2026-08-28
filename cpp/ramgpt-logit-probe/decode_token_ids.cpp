#include "llama.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string arg(int n,char**v,const std::string&k){for(int i=1;i+1<n;i++)if(v[i]==k)return v[i+1];throw std::runtime_error("missing "+k);}
std::string piece(const llama_vocab*v,llama_token t){std::vector<char>b(256);int n=llama_token_to_piece(v,t,b.data(),b.size(),0,true);if(n<0){b.resize(-n);n=llama_token_to_piece(v,t,b.data(),b.size(),0,true);}if(n<0)throw std::runtime_error("token-to-piece failed");return std::string(b.data(),n);}
std::string valid_utf8(const std::string&s){std::string o;for(size_t i=0;i<s.size();){unsigned char c=s[i];size_t n=c<0x80?1:(c>=0xc2&&c<=0xdf?2:(c>=0xe0&&c<=0xef?3:(c>=0xf0&&c<=0xf4?4:0)));bool ok=n&&i+n<=s.size();for(size_t j=1;ok&&j<n;j++)ok=(static_cast<unsigned char>(s[i+j])&0xc0)==0x80;if(ok&&n==3){unsigned char d=s[i+1];ok=!(c==0xe0&&d<0xa0)&&!(c==0xed&&d>=0xa0);}if(ok&&n==4){unsigned char d=s[i+1];ok=!(c==0xf0&&d<0x90)&&!(c==0xf4&&d>=0x90);}if(ok){o.append(s,i,n);i+=n;}else{o+="\xef\xbf\xbd";i++;}}return o;}
std::string esc(const std::string&s){std::string o;char h[]="0123456789abcdef";for(unsigned char c:s)switch(c){case'"':o+="\\\"";break;case'\\':o+="\\\\";break;case'\n':o+="\\n";break;case'\r':o+="\\r";break;case'\t':o+="\\t";break;default:if(c<0x20){o+="\\u00";o+=h[c>>4];o+=h[c&15];}else o+=char(c);}return o;}
}
int main(int n,char**v){try{auto model_path=arg(n,v,"--model"),ids_path=arg(n,v,"--ids"),out_path=arg(n,v,"--output");std::ifstream in(ids_path);if(!in)throw std::runtime_error("cannot open IDs");std::set<int32_t>ids;std::string line;while(std::getline(in,line)){if(line.empty())continue;size_t used=0;long x=std::stol(line,&used);if(used!=line.size()||x<0)throw std::runtime_error("invalid token ID");ids.insert(int32_t(x));}if(ids.empty())throw std::runtime_error("empty token ID set");llama_backend_init();auto p=llama_model_default_params();p.vocab_only=true;std::unique_ptr<llama_model,decltype(&llama_model_free)>m(llama_model_load_from_file(model_path.c_str(),p),llama_model_free);if(!m)throw std::runtime_error("vocabulary load failed");auto*vocab=llama_model_get_vocab(m.get());std::ofstream out(out_path);if(!out)throw std::runtime_error("cannot create output");for(auto id:ids)out<<"{\"token_id\":"<<id<<",\"piece\":\""<<esc(valid_utf8(piece(vocab,id)))<<"\"}\n";m.reset();llama_backend_free();return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
