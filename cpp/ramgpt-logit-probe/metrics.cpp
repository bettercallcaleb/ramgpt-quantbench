#include "metrics.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rq {
static void best2(const float *x,uint32_t n,int32_t&a,int32_t&b){if(n<2)throw std::runtime_error("vocabulary too small");a=0;b=1;if(x[b]>x[a])std::swap(a,b);for(uint32_t i=2;i<n;i++)if(x[i]>x[a]){b=a;a=i;}else if(x[i]>x[b])b=i;}
static double lse(const float*x,uint32_t n){float m=*std::max_element(x,x+n);if(!std::isfinite(m))throw std::runtime_error("non-finite logits");double s=0,c=0;for(uint32_t i=0;i<n;i++){double y=std::exp(double(x[i]-m))-c,t=s+y;c=(t-s)-y;s=t;}return double(m)+std::log(s);}
RowMetrics calculate_metrics(const float*r,const float*q,uint32_t n,int32_t target){if(target<0||uint32_t(target)>=n)throw std::runtime_error("target outside vocabulary");RowMetrics z{};best2(r,n,z.ref_top1,z.ref_top2);best2(q,n,z.quant_top1,z.quant_top2);z.ref_top1_logit=r[z.ref_top1];z.ref_top2_logit=r[z.ref_top2];z.quant_top1_logit=q[z.quant_top1];z.quant_top2_logit=q[z.quant_top2];z.ref_margin=z.ref_top1_logit-z.ref_top2_logit;double lr=lse(r,n),lq=lse(q,n),c=0;z.kl=0;for(uint32_t i=0;i<n;i++){double logp=double(r[i])-lr,p=std::exp(logp),logq=double(q[i])-lq,y=p*(logp-logq)-c,t=z.kl+y;c=(t-z.kl)-y;z.kl=t;}if(z.kl<0&&z.kl>-1e-12)z.kl=0;z.p_ref_target=std::exp(double(r[target])-lr);z.p_quant_target=std::exp(double(q[target])-lq);z.delta_p_target=z.p_quant_target-z.p_ref_target;z.nll_ref=lr-r[target];z.nll_quant=lq-q[target];z.delta_nll=z.nll_quant-z.nll_ref;z.decision_flip=z.quant_top1!=z.ref_top1;return z;}
double percentile_sorted(const double*v,size_t n,double p){if(!n)return std::numeric_limits<double>::quiet_NaN();double x=(n-1)*p;size_t lo=size_t(x),hi=std::min(lo+1,n-1);return v[lo]+(v[hi]-v[lo])*(x-lo);}
}
