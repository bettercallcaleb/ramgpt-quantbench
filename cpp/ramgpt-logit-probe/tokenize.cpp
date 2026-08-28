#include "formats.h"
#include "llama.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string arg(int argc,char**argv,const std::string&key,bool optional=false){for(int i=1;i+1<argc;i++)if(argv[i]==key)return argv[i+1];if(optional)return{};throw std::runtime_error("missing "+key);}
int main(int argc,char**argv){try{
    auto model_path=arg(argc,argv,"--model"),list=arg(argc,argv,"--batch-list",true),input=arg(argc,argv,"--input",true),output=arg(argc,argv,"--output",true);if(list.empty()&&(input.empty()||output.empty()))throw std::runtime_error("supply --input/--output or --batch-list");
    auto add_special_arg=arg(argc,argv,"--add-special",true);bool add_special=true;if(!add_special_arg.empty()){if(add_special_arg=="true"||add_special_arg=="1")add_special=true;else if(add_special_arg=="false"||add_special_arg=="0")add_special=false;else throw std::runtime_error("--add-special must be true or false");}
    llama_backend_init();auto mp=llama_model_default_params();mp.vocab_only=true;llama_model*m=llama_model_load_from_file(model_path.c_str(),mp);if(!m)throw std::runtime_error("model load failed");
    const llama_vocab*v=llama_model_get_vocab(m);auto one=[&](const std::string&in,const std::string&out){std::ifstream f(in,std::ios::binary);if(!f)throw std::runtime_error("cannot open input text: "+in);std::string text((std::istreambuf_iterator<char>(f)),{});int32_t n=llama_tokenize(v,text.data(),text.size(),nullptr,0,add_special,false);if(n>0)throw std::runtime_error("unexpected tokenizer sizing result");std::vector<llama_token>t(size_t(-n));if(n<0){n=llama_tokenize(v,text.data(),text.size(),t.data(),t.size(),add_special,false);if(n<0)throw std::runtime_error("tokenization failed");t.resize(n);}std::vector<int32_t>ids(t.begin(),t.end());rq::write_tokens(out,ids);std::cout<<in<<"\t"<<out<<"\t"<<ids.size()<<"\t"<<rq::hex(rq::sha256_file(out))<<"\n";};
    if(list.empty())one(input,output);else{std::ifstream f(list);if(!f)throw std::runtime_error("cannot open batch list");std::string line;while(std::getline(f,line)){if(line.empty())continue;auto tab=line.find('\t');if(tab==std::string::npos||line.find('\t',tab+1)!=std::string::npos)throw std::runtime_error("batch list must contain input<TAB>output");one(line.substr(0,tab),line.substr(tab+1));}}
    char desc[128]={},name[256]={},tok_model[256]={},tok_pre[256]={};llama_model_desc(m,desc,sizeof desc);llama_model_meta_val_str(m,"general.name",name,sizeof name);llama_model_meta_val_str(m,"tokenizer.ggml.model",tok_model,sizeof tok_model);llama_model_meta_val_str(m,"tokenizer.ggml.pre",tok_pre,sizeof tok_pre);std::cout<<"model_id="<<desc<<" general_name="<<name<<" tokenizer_model="<<tok_model<<" tokenizer_pre="<<tok_pre<<" vocab_size="<<llama_vocab_n_tokens(v)<<" add_special="<<(add_special?"true":"false")<<" parse_special=false\n";
    llama_model_free(m);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
