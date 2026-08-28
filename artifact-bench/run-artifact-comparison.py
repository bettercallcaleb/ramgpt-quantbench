#!/usr/bin/env python3
"""Low-disk, resumable RAMGPT QuantBench multi-artifact publisher.

Numerical helpers generalize the frozen Qwen3-4B V1 implementation.  Live
model-family adapters are intentionally explicit: unsupported tokenizers stop
before download/inference rather than changing the scientific contract.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, re, shutil, struct, sys, tempfile
import time, urllib.error, urllib.parse, urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT=Path(__file__).resolve().parents[1]; PUBLIC=ROOT/"artifact-bench/public"
COMMIT="70adb1b4cea5ee39f867792c78dc59320921eda7"; N=65536; S=128; P=512
REPS=10000; SEED=20260823; THRESHOLDS=(1,2,4); GIB=1<<30
V1_SHA="f7fce69568d3e8a55afc2dfca086a95afc5897d15f13673cb99e9e9d93eb4ae2"
sys.path.insert(0,str(ROOT/"artifact-bench"));from adapters import qwen
CONTRACT={"version":"QuantBench-V1","n_ctx":4096,"n_batch":512,"n_ubatch":512,
 "kv_k_type":"F16","kv_v_type":"F16","gpu_layers":-1,"sampling":False,
 "full_vocabulary_fp32_logits":True,"reference":"native BF16","flash_attention":True,
 "fresh_context_each_segment":True,"rql_version":2,"tokenizer_add_special":True,
 "tokenizer_parse_special":False,"total_positions":N,"segments":S,"positions_per_segment":P,
 "hcdf_thresholds":[1,2,4],"bootstrap":{"replicates":REPS,"seed":SEED,"resampling_unit":"independent segment"}}

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def atomic_json(p:Path,x:Any):
 p.parent.mkdir(parents=True,exist_ok=True); fd,n=tempfile.mkstemp(prefix=p.name+".",dir=p.parent)
 try:
  with os.fdopen(fd,"w") as f: json.dump(x,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
  os.replace(n,p)
 finally:
  if os.path.exists(n): os.unlink(n)
def slug(x): return re.sub(r"[^a-z0-9]+","-",x.lower()).strip("-")
def validate_url(u):
 q=urllib.parse.urlparse(u)
 if q.scheme not in ("http","https") or not q.netloc or q.username or q.password:
  raise argparse.ArgumentTypeError("inputs must be credential-free HTTP(S) URLs")
def model_identity(u):
 q=urllib.parse.urlparse(u); x=[z for z in q.path.split("/") if z]
 if q.netloc.lower() in ("huggingface.co","www.huggingface.co") and len(x)>=2:return "/".join(x[:2])
 return re.sub(r"(?i)[-_](bf16|f16)\.gguf$","",Path(q.path).name) or q.netloc
def derive_id(ref,quant): return slug(f"{model_identity(ref)}-{quant}-open-artifact-rank-v1")
def artifact_arg(x):
 if "=" not in x: raise argparse.ArgumentTypeError("artifact must use LABEL=URL")
 label,url=x.split("=",1)
 if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,63}",label): raise argparse.ArgumentTypeError("invalid artifact label")
 validate_url(url); return label,url
def make_parser():
 p=argparse.ArgumentParser(description="Run and publish an N-artifact QuantBench comparison")
 p.add_argument("--reference",required=True); p.add_argument("--artifact",action="append",type=artifact_arg,required=True,metavar="LABEL=URL")
 p.add_argument("--quantization-label",required=True); p.add_argument("--benchmark-id")
 p.add_argument("--work-dir",type=Path); p.add_argument("--output-dir",type=Path)
 p.add_argument("--llama-cpp-dir",type=Path,default=ROOT/"third_party/llama.cpp")
 p.add_argument("--adapter",choices=("auto","qwen"),default="auto");p.add_argument("--min-free-gib",type=float,default=10.0)
 p.add_argument("--reference-gpu-layers",type=int,default=-1,help="GPU layers for the native/full-precision reference only; Q4 artifacts remain full GPU (-1)")
 p.add_argument("--resume",action="store_true"); p.add_argument("--cleanup",choices=("none","safe","aggressive"),default="safe")
 p.add_argument("--dry-run",action="store_true"); return p
def parse_args(argv=None):
 p=make_parser(); a=p.parse_args(argv)
 if len(a.artifact)<2:p.error("at least 2 --artifact arguments are required")
 labels=[x[0].casefold() for x in a.artifact]
 if len(labels)!=len(set(labels)):p.error("duplicate artifact labels are not allowed")
 validate_url(a.reference); a.benchmark_id=a.benchmark_id or derive_id(a.reference,a.quantization_label)
 if a.min_free_gib<10:p.error("--min-free-gib must be at least 10")
 if a.reference_gpu_layers < -1:p.error("--reference-gpu-layers must be -1 or a non-negative integer")
 try:qwen.parse_hf_url(a.reference)
 except ValueError as e:p.error(str(e))
 if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*",a.benchmark_id):p.error("--benchmark-id must be a lowercase slug")
 return a

class State:
 def __init__(self,p:Path,resume=False):
  self.path=p
  if resume:
   if not p.exists():raise RuntimeError("--resume requested but run-state.json is absent")
   self.data=json.loads(p.read_text()); self.verify()
  else:
   if p.exists():raise RuntimeError("run exists; use --resume or another benchmark ID")
   self.data={"schema":"ramgpt-artifact-bench-run-state-v1","status":"ACTIVE","created_at":now(),"stages":{},"space_freed_bytes":0}; self.save()
 def save(self):atomic_json(self.path,self.data)
 def complete(self,stage,files=()):
  self.data["stages"][stage]={"status":"COMPLETE","at":now(),"files":{str(p.resolve()):{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files}};self.save()
 def done(self,s):return self.data["stages"].get(s,{}).get("status")=="COMPLETE"
 def fail(self,s,e):self.data["status"]="FAILED";self.data["failure"]={"stage":s,"at":now(),"error":str(e)};self.save()
 def verify(self):
  deleted=set(self.data.get("deleted_checkpoint_files",[]))
  for s,r in self.data.get("stages",{}).items():
   for n,e in r.get("files",{}).items():
    if n in deleted:continue
    p=Path(n)
    if not p.exists() or p.stat().st_size!=e["bytes"] or sha256(p)!=e["sha256"]:raise RuntimeError(f"resume checkpoint integrity failure: {s}: {p}")
def check_space(p,peak,stage,margin=10*GIB,input_bytes=0,output_bytes=0):
 free=shutil.disk_usage(p).free;print(f"SPACE: stage={stage} free={free} expected_peak_temporary={peak}")
 print(f"FREE_SPACE: {free}\nEXPECTED_STAGE_INPUT: {input_bytes}\nEXPECTED_STAGE_OUTPUT: {output_bytes}\nEXPECTED_PEAK: {peak}\nSAFETY_MARGIN: {margin}")
 if peak and free<peak+margin:raise RuntimeError(f"insufficient disk before {stage}")
def delete_safe(p:Path,state:State,checkpoint):
 if not state.done(checkpoint):raise RuntimeError(f"refusing cleanup before {checkpoint}")
 if not p.exists():return
 roots=[state.path.parent.resolve()]+[Path(x).resolve() for x in state.data.get("cleanup_roots",[])]
 if not any(p.resolve().is_relative_to(x) for x in roots):raise RuntimeError("refusing deletion outside run workspace")
 n=p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
 resolved=p.resolve();deleted=state.data.setdefault("deleted_checkpoint_files",[])
 for rec in state.data.get("stages",{}).values():
  for name in rec.get("files",{}):
   q=Path(name)
   if q==resolved or (p.is_dir() and q.is_relative_to(resolved)):deleted.append(name)
 shutil.rmtree(p) if p.is_dir() else p.unlink();state.data["space_freed_bytes"]+=n
 state.save();print(f"DELETE_SAFE: {p} {n} bytes freed")

def resolve_url(url):
 url=url.replace("huggingface.co/","huggingface.co/").replace("/blob/","/resolve/")
 req=urllib.request.Request(url,method="HEAD",headers={"User-Agent":"RAMGPT-QuantBench/1"})
 with urllib.request.urlopen(req,timeout=30) as r:
  length=r.headers.get("Content-Length") or r.headers.get("X-Linked-Size")
  if not length:raise RuntimeError("artifact server did not provide content size; cannot enforce disk preflight")
  return {"source_url":url,"resolved_url":r.geturl(),"content_length":int(length),"etag":r.headers.get("ETag"),"linked_etag":r.headers.get("X-Linked-Etag")}
def download(url,target:Path,size=None):
 target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix(target.suffix+".partial")
 attempts=6
 for attempt in range(1,attempts+1):
  h=hashlib.sha256();n=0
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"RAMGPT-QuantBench/1"}),timeout=60) as a,partial.open("wb") as b:
    while x:=a.read(8<<20):b.write(x);h.update(x);n+=len(x)
    b.flush();os.fsync(b.fileno())
   if size is not None and n!=size:raise RuntimeError("download size changed after preflight")
   os.replace(partial,target)
   return {"bytes":n,"sha256":h.hexdigest()}
  except urllib.error.HTTPError as e:
   if partial.exists():partial.unlink()
   if e.code not in (429,500,502,503,504) or attempt==attempts:raise
   retry_after=e.headers.get("Retry-After") if e.headers else None
   try:delay=min(300,max(1,int(retry_after)))
   except (TypeError,ValueError):delay=min(300,30*(2**(attempt-1)))
   print(f"DOWNLOAD_RETRY: http={e.code} attempt={attempt}/{attempts} sleep={delay}s url={url}",flush=True)
   time.sleep(delay)
  except (urllib.error.URLError,TimeoutError,ConnectionError) as e:
   if partial.exists():partial.unlink()
   if attempt==attempts:raise
   delay=min(300,30*(2**(attempt-1)))
   print(f"DOWNLOAD_RETRY: error={type(e).__name__} attempt={attempt}/{attempts} sleep={delay}s url={url}",flush=True)
   time.sleep(delay)
 raise RuntimeError("download retry loop exhausted")

def rql_header(p:Path,expected_gpu_layers=None):
 with p.open("rb") as f:b=f.read(321)
 if len(b)!=321 or b[:8]!=b"RQLOGIT\0":raise RuntimeError("invalid RQL v2")
 schema,header,vocab,dtype=struct.unpack_from("<IIII",b,8);v=struct.unpack_from("<14I",b,265);gpu=struct.unpack("<i",struct.pack("<I",v[5]))[0]
 got={"schema":schema,"header":header,"vocab":vocab,"dtype":dtype,"positions":struct.unpack_from("<Q",b,24)[0],"model_sha":b[32:64].hex(),"token_sha":b[64:96].hex(),"commit":b[96:137].split(b"\0",1)[0].decode(),"ctx":v[0],"batch":v[1],"ubatch":v[2],"k":v[3],"vv":v[4],"gpu":gpu,"fa":v[6],"add":v[7],"parse":v[8],"sampling":v[9],"reserved":list(v[10:])}
 exp={"schema":2,"header":321,"dtype":1,"positions":N,"commit":COMMIT,"ctx":4096,"batch":512,"ubatch":512,"k":1,"vv":1,"fa":1,"add":1,"parse":0,"sampling":0,"reserved":[0,0,0,0]}
 if expected_gpu_layers is not None:
  exp["gpu"]=expected_gpu_layers
 bad={k:(got[k],x) for k,x in exp.items() if got[k]!=x}
 if bad:raise RuntimeError(f"frozen execution contract mismatch: {bad}")
 if p.stat().st_size!=321+N*vocab*4:raise RuntimeError("RQL size/header mismatch")
 return got
def positions(p):
 x=[json.loads(z) for z in p.read_text().splitlines()]
 if len(x)!=N:raise RuntimeError("position count mismatch")
 return x
def analyze_pair(ref:Path,cand:Path,rpos:Path,cpos:Path,compact:Path,independent=False):
 import numpy as np
 rh,ch=rql_header(ref),rql_header(cand)
 if rh["vocab"]!=ch["vocab"] or rh["token_sha"]!=ch["token_sha"]:raise RuntimeError("tokenizer/vocabulary mismatch")
 rp,cp=positions(rpos),positions(cpos);keys=("global_scored_row_index","segment_id","input_position","target_position","input_token_id","target_token_id")
 if any(any(a.get(k)!=b.get(k) for k in keys) for a,b in zip(rp,cp)):raise RuntimeError("exact position alignment failed")
 R=np.memmap(ref,dtype="<f4",mode="r",offset=321,shape=(N,rh["vocab"]));Q=np.memmap(cand,dtype="<f4",mode="r",offset=321,shape=(N,rh["vocab"]));kl=np.empty(N);margin=np.empty(N);flip=np.empty(N,bool);top=np.array([x["top1_token_id"] for x in rp])
 chunk=16 if independent else 32
 for s in range(0,N,chunk):
  e=min(N,s+chunk);r=np.asarray(R[s:e],dtype=np.float64);q=np.asarray(Q[s:e],dtype=np.float64)
  if independent:
   zr=np.logaddexp.reduce(r,1);zq=np.logaddexp.reduce(q,1);pr=np.exp(r-zr[:,None]);kl[s:e]=np.einsum("ij,ij->i",pr,(r-zr[:,None])-(q-zq[:,None]))
  else:
   mr=r.max(1);er=np.exp(r-mr[:,None]);sr=er.sum(1);zr=mr+np.log(sr);mq=q.max(1);zq=mq+np.log(np.exp(q-mq[:,None]).sum(1));kl[s:e]=(er/sr[:,None]*(r-q)).sum(1)+zq-zr
  two=np.partition(r,-2,axis=1)[:,-2:];margin[s:e]=two.max(1)-two.min(1);flip[s:e]=np.argmax(q,1)!=top[s:e]
 seg=np.array([x["segment_id"] for x in rp],np.int16)
 if len(np.unique(seg))!=S or any((seg==x).sum()!=P for x in np.unique(seg)):raise RuntimeError("128 x 512 geometry failed")
 hc={str(t):{"eligible_positions":int((margin>=t).sum()),"flips":int((flip&(margin>=t)).sum()),"hcdf_rate":float((flip&(margin>=t)).sum()/(margin>=t).sum()) if (margin>=t).any() else None} for t in THRESHOLDS}
 m={"mean_kl_bf16_to_artifact":float(kl.mean()),"top1_flips":int(flip.sum()),"hcdfr":hc,"max_flipped_bf16_margin":float(margin[flip].max()) if flip.any() else None,"segment_mean_kl":[float(kl[seg==x].mean()) for x in sorted(np.unique(seg))]}
 if not independent:compact.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(compact,kl=kl,segment_id=seg,top1_flip=flip,bf16_margin=margin);m["compact_sha256"]=sha256(compact)
 return m
def bootstrap(rows):
 import numpy as np
 rank=sorted(rows,key=lambda x:x["mean_kl_bf16_to_artifact"]);w=rank[0];rng=np.random.default_rng(SEED);draw=rng.integers(0,S,size=(REPS,S));out=[]
 for x in rank[1:]:
  d=np.asarray(x["segment_mean_kl"])-np.asarray(w["segment_mean_kl"]);b=d[draw].mean(1);lo,hi=np.quantile(b,[.025,.975]);out.append({"winner":w["artifact"],"other":x["artifact"],"delta_definition":"other_mean_KL - winner_mean_KL","point_estimate_delta_kl":float(d.mean()),"ci_95_low":float(lo),"ci_95_high":float(hi),"seed":SEED,"replicates":REPS,"resampling_unit":"segment","segments":S})
 return {"winner":w["artifact"],"winner_vs_all":out,"winner_vs_runner_up":out[0] if out else None}

CSV_FIELDS="benchmark_id model quantization_label rank artifact artifact_url artifact_sha256 artifact_bytes mean_kl_bf16_to_artifact delta_kl_vs_winner relative_kl_difference_vs_winner_percent top1_flips hcdfr_ge_1 hcdfr_ge_2 hcdfr_ge_4 max_flipped_bf16_margin paired_ci_low_vs_winner paired_ci_high_vs_winner provenance_grade strict_recipe_eligible verification_status reference_sha256 llama_cpp_commit total_positions segments".split()
def write_csv(p,bid,model,quant,rows,u,refsha):
 rank=sorted(rows,key=lambda x:x["mean_kl_bf16_to_artifact"]);win=rank[0]["mean_kl_bf16_to_artifact"];cis={x["other"]:x for x in u["winner_vs_all"]};p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=CSV_FIELDS);w.writeheader()
  for i,x in enumerate(rank,1):
   d=x["mean_kl_bf16_to_artifact"]-win;c=cis.get(x["artifact"]);w.writerow({"benchmark_id":bid,"model":model,"quantization_label":quant,"rank":i,"artifact":x["artifact"],"artifact_url":x["artifact_url"],"artifact_sha256":x["artifact_sha256"],"artifact_bytes":x["artifact_bytes"],"mean_kl_bf16_to_artifact":x["mean_kl_bf16_to_artifact"],"delta_kl_vs_winner":d,"relative_kl_difference_vs_winner_percent":d/win*100 if win else 0,"top1_flips":x["top1_flips"],"hcdfr_ge_1":x["hcdfr"]["1"]["hcdf_rate"],"hcdfr_ge_2":x["hcdfr"]["2"]["hcdf_rate"],"hcdfr_ge_4":x["hcdfr"]["4"]["hcdf_rate"],"max_flipped_bf16_margin":x["max_flipped_bf16_margin"],"paired_ci_low_vs_winner":c["ci_95_low"] if c else 0,"paired_ci_high_vs_winner":c["ci_95_high"] if c else 0,"provenance_grade":x["provenance_grade"],"strict_recipe_eligible":str(x["strict_recipe_eligible"]).lower(),"verification_status":"PASS","reference_sha256":refsha,"llama_cpp_commit":COMMIT,"total_positions":N,"segments":S})
def next_registry(public=PUBLIC):
 x=[]
 for p in public.glob("leaderboard-index-v*.json"):
  m=re.fullmatch(r"leaderboard-index-v(\d+)\.json",p.name)
  if m:x.append((int(m.group(1)),p))
 if not x:raise RuntimeError("no frozen registry found")
 x.sort();return x[-1][0]+1,x[-1][1]
def publish_prompt(bid,v,hashes):return f"""Verify every evidence-package SHA-256 against publish-manifest.json: {json.dumps(hashes,sort_keys=True)}. Preserve all existing leaderboard-index-v*.json snapshots byte-for-byte. Publish forensic-evidence-v1.json and publication-provenance-v1.json under a benchmark-specific path for {bid}, and publish leaderboard-index-v{v}.json as a new immutable registry snapshot. Validate the destination project, inspect the diff, and stop on any hash or schema failure.\n"""


def publication_provenance(spec,refsha,verified_sha,forensic_sha,reference_preflight_sha):
 return {"schema":"ramgpt-quantbench-publication-provenance","schema_version":1,"benchmark_id":spec["benchmark_id"],"model":spec["model"],"resolved_reference_revision":spec["resolved_reference_revision"],"reference_bf16_sha256":refsha,"reference_preflight_sha256":reference_preflight_sha,"llama_cpp_commit":spec["llama_cpp_commit"],"execution_contract_version":spec["execution_contract"]["version"],"source_verified_result_sha256":verified_sha,"source_forensic_evidence_sha256":forensic_sha}

def copy_immutable(src:Path,dst:Path):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():
  if sha256(dst)!=sha256(src):raise RuntimeError(f"immutable release file already exists with different content: {dst}")
  return
 shutil.copy2(src,dst)

def freeze(out,spec,rows,u,refsha):
 rank=sorted(rows,key=lambda x:x["mean_kl_bf16_to_artifact"]);winner=rank[0]["artifact"];runner_up=rank[1] if len(rank)>1 else None;pair=u["winner_vs_runner_up"]
 common={"benchmark_id":spec["benchmark_id"],"model":spec["model"],"quantization_label":spec["quantization_label"],"geometry":{"total_positions":N,"segments":S,"positions_per_segment":P},"execution_contract":spec["execution_contract"]}
 verified={**common,"schema":"ramgpt-quantbench-verified-open-artifact-result","schema_version":1,"status":"PASS","comparison_class":"OPEN_ARTIFACT_DOWNLOAD_RANK","primary_metric":"mean KL(BF16 || artifact), lower is better","rankings":rank,"winner":winner,"paired_uncertainty":u,"reference_sha256":refsha,"claim_scope":f"Under RAMGPT QuantBench Open Artifact Rank V1, the tested {winner} artifact is behaviorally closer to the pinned native BF16 reference than the other tested {spec['quantization_label']} artifacts."};atomic_json(out/"verified-result.json",verified)
 validations=[]
 for p in sorted((out/"logs").glob("artifact-*-validation.json")):
  validations.append({"file":str(p.relative_to(out)),"sha256":sha256(p),"content":json.loads(p.read_text())})
 if len(validations)!=len(rank) or any(x["content"].get("status")!="PASS" for x in validations):raise RuntimeError("cannot freeze: consolidated independent validation is incomplete")
 independent=out/"independent-validation.json";atomic_json(independent,{"schema":"ramgpt-quantbench-independent-validation-v1","benchmark_id":spec["benchmark_id"],"status":"PASS","artifacts":validations})
 strict_reason="Strict Recipe Rank is ineligible because this open-artifact comparison does not establish controlled, equivalent quantization recipe provenance for every tested artifact. The result ranks exact downloadable artifacts, not quantization algorithms."
 logits=N*int(spec["reference_vocabulary_size"]);determinism=f"Two complete independent captures were bitwise identical across {logits:,} FP32 logits."
 forensic={**common,"schema":"ramgpt-quantbench-forensic-evidence","schema_version":1,"artifacts":{x["artifact"]:{"mean_kl":x["mean_kl_bf16_to_artifact"],"top1_flips":x["top1_flips"],"hcdfr":x["hcdfr"],"max_flipped_bf16_margin":x["max_flipped_bf16_margin"],"compact_evidence_sha256":x["compact_sha256"]} for x in rank},"independent_validation":{"sha256":sha256(independent),"status":"PASS"},"methodology":{"comparison_class":"OPEN_ARTIFACT_DOWNLOAD_RANK","primary_metric":"mean KL(BF16 || artifact), lower is better","reference":"native BF16","strict_recipe_rank":{"eligible":False,"reason":strict_reason}},"reference_determinism":determinism,"source_verified_result_sha256":sha256(out/"verified-result.json")};atomic_json(out/"forensic-evidence-v1.json",forensic)
 public_rankings=[{"artifact_sha256":x["artifact_sha256"],"changed_top1_count":x["top1_flips"],"mean_kl_bf16_to_artifact":x["mean_kl_bf16_to_artifact"],"publisher":x["artifact"],"rank":i} for i,x in enumerate(rank,1)]
 win_kl=rank[0]["mean_kl_bf16_to_artifact"];runner_kl=runner_up["mean_kl_bf16_to_artifact"] if runner_up else win_kl;relative=(runner_kl-win_kl)/runner_kl*100 if runner_kl else 0.0
 primary={"name":"mean KL(BF16 || artifact)","direction":"lower_is_better","winner_artifact":winner,"winner_mean_kl":win_kl,"winner_vs_runner_up_relative_reduction_percent":relative,"artifact_values":{x["artifact"]:x["mean_kl_bf16_to_artifact"] for x in rank}}
 paired_public={"ci_95":[pair["ci_95_low"],pair["ci_95_high"]],"delta":pair["point_estimate_delta_kl"],"delta_definition":f"mean_KL_{pair['other']} - mean_KL_{pair['winner']}","replicates":pair["replicates"],"resampling_unit":"segment","seed":pair["seed"],"segments":pair["segments"],"winner_vs_all":u["winner_vs_all"]}
 entry_base={"badge":"RAMGPT_QUANTBENCH_VERIFIED",**common,"comparison_class":"OPEN_ARTIFACT_DOWNLOAD_RANK","paired_uncertainty":paired_public,"primary_metric":primary,"ramgpt_recommended":{"awarded":False,"status":"NOT_AWARDED"},"rankings":public_rankings,"status":"VERIFIED","strict_recipe_rank":{"eligible":False,"reason":strict_reason,"status":"INELIGIBLE"},"winner":winner};atomic_json(out/"leaderboard-entry.json",entry_base)
 files=[out/"run-spec.json",out/"result.csv",out/"verified-result.json",out/"forensic-evidence-v1.json",out/"independent-validation.json",out/"leaderboard-entry.json"]+sorted((out/"compact").glob("*.npz"));reference_preflight=out/"reference-preflight.json"
 if reference_preflight.exists():files.append(reference_preflight)
 atomic_json(out/"release-manifest.json",{"schema":"ramgpt-artifact-bench-release-manifest-v1","benchmark_id":spec["benchmark_id"],"files":{str(x.relative_to(out)):{"bytes":x.stat().st_size,"sha256":sha256(x)} for x in files}})
 release_dir=PUBLIC/spec["benchmark_id"]
 for src in (out/"leaderboard-entry.json",out/"release-manifest.json",out/"forensic-evidence-v1.json",out/"verified-result.json",out/"independent-validation.json",out/"reference-preflight.json"):
  if src.exists():copy_immutable(src,release_dir/src.name)
 release={"directory":f"artifact-bench/public/{spec['benchmark_id']}","leaderboard_entry_sha256":sha256(out/"leaderboard-entry.json"),"release_manifest_sha256":sha256(out/"release-manifest.json")}
 entry={**entry_base,"release":release};v,latest=next_registry();old=json.loads(latest.read_text());new=json.loads(json.dumps(old));new["entries"].append(entry);new["entry_count"]=len(new["entries"]);new["schema_version"]=old.get("schema_version",1);new["registry_snapshot_version"]=v;reg=out/f"leaderboard-index-v{v}.json";atomic_json(reg,new)
 copy_immutable(reg,PUBLIC/reg.name)
 web=out/"web-package";web.mkdir(exist_ok=True)
 shutil.copy2(out/"forensic-evidence-v1.json",web/"forensic-evidence-v1.json")
 shutil.copy2(reg,web/reg.name)
 prov=web/"publication-provenance-v1.json"
 atomic_json(prov,publication_provenance(spec,refsha,sha256(out/"verified-result.json"),sha256(out/"forensic-evidence-v1.json"),sha256(reference_preflight)))
 h={x.name:sha256(x) for x in (web/"forensic-evidence-v1.json",prov,web/reg.name)}
 prompt=web/"CODEX_PUBLISH_PROMPT.txt";prompt.write_text(publish_prompt(spec["benchmark_id"],v,h));h[prompt.name]=sha256(prompt)
 manifest={"benchmark_id":spec["benchmark_id"],"model":spec["model"],"quantization_label":spec["quantization_label"],"registry_version":v,"source_paths":{"forensic":str(out/"forensic-evidence-v1.json"),"registry":str(reg)},"sha256":h,"suggested_public_forensic_path":f"/quantbench/data/{spec['benchmark_id']}/forensic-evidence-v1.json","suggested_public_provenance_path":f"/quantbench/data/{spec['benchmark_id']}/publication-provenance-v1.json","suggested_public_registry_path":f"/quantbench/leaderboard-index-v{v}.json"};atomic_json(web/"publish-manifest.json",manifest)

def frozen_regression():
 import numpy as np
 base=ROOT/"artifact-bench/results/qwen3-4b-q4-k-m/open-artifact-rank-v1";compact=base/"per-position.npz";valid=base/"independent-validation.json"
 expected={compact:"a800b87bae0dca102d6acecce41efbd3f63aaca9edc8724f883298884e7e007f",valid:"ac98a6b1f53a1466174f1b699eea6bc5c0fdb4e9b86d6f9f7a0cf49f19263828"}
 if any(not p.exists() or sha256(p)!=h for p,h in expected.items()):raise RuntimeError("frozen Qwen3-4B regression artifact hash mismatch")
 p=np.load(compact);a=p["kl_ggml_org"];b=p["kl_unsloth"];margin=p["bf16_margin"];fa=p["flip_ggml_org"];fb=p["flip_unsloth"]
 got={"ggml_mean":float(a.mean()),"unsloth_mean":float(b.mean()),"delta":float(a.mean()-b.mean()),"ggml_flips":int(fa.sum()),"unsloth_flips":int(fb.sum()),"ggml_max_margin":float(margin[fa].max()),"unsloth_max_margin":float(margin[fb].max())}
 exp={"ggml_mean":0.05396429797282026,"unsloth_mean":0.03854230747637248,"delta":0.015421990496447782,"ggml_flips":6437,"unsloth_flips":5699,"ggml_max_margin":24.74701690673828,"unsloth_max_margin":9.239395141601562}
 if got!=exp:raise RuntimeError(f"frozen primary regression mismatch: {got}")
 hc={}
 for name,f in (("ggml",fa),("unsloth",fb)):
  hc[name]={str(t):float((f&(margin>=t)).sum()/(margin>=t).sum()) for t in THRESHOLDS}
 expected_hc={"ggml":{"1":0.01141050316194666,"2":0.0035390894018620072,"4":0.0016737674984783932},"unsloth":{"1":0.00769865273577124,"2":0.0018492539216936614,"4":0.0006593629539460336}}
 if hc!=expected_hc:raise RuntimeError(f"frozen HCDFR regression mismatch: {hc}")
 rows=[{"artifact":"ggml-org","mean_kl_bf16_to_artifact":got["ggml_mean"],"segment_mean_kl":[float(a[p["segment_id"]==i].mean()) for i in range(S)]},{"artifact":"Unsloth","mean_kl_bf16_to_artifact":got["unsloth_mean"],"segment_mean_kl":[float(b[p["segment_id"]==i].mean()) for i in range(S)]}];u=bootstrap(rows)
 # The frozen independent validator uses absolute tolerance 1e-10 for reductions.
 if u["winner"]!="Unsloth" or abs(u["winner_vs_runner_up"]["point_estimate_delta_kl"]-exp["delta"])>=1e-10:raise RuntimeError("frozen winner/bootstrap regression mismatch")
 return {"status":"PASS",**got,"hcdfr":hc,"hashes":{x.name:h for x,h in expected.items()}}

def audit_large_files(work:Path):
 large=[{"path":str(p),"bytes":p.stat().st_size} for p in work.rglob("*") if p.is_file() and p.stat().st_size>GIB]
 print(f"LARGE_FILES_REMAINING: {len(large)}")
 if large:raise RuntimeError(f"aggressive cleanup left >1 GiB files: {large}")

def artifact_record(label,url,meta,metrics):
 return {"artifact":label,"artifact_url":url,"artifact_sha256":meta["sha256"],"artifact_bytes":meta["bytes"],**metrics,"provenance_grade":"Grade D: artifact identity verified; recipe unverified","strict_recipe_eligible":False}

def rql_expected_bytes(vocabulary_size:int)->int:
 if not isinstance(vocabulary_size,int) or vocabulary_size<=0:raise ValueError("vocabulary size must be a positive integer")
 return 321+N*vocabulary_size*4
def disk_model(reference_weights:int,vocabulary_size:int,candidates:list[dict],free_bytes:int,safety_margin:int,retain_bf16_during_candidates:bool=False)->dict[str,Any]:
 if reference_weights<=0 or not candidates or any(x["remote_bytes"]<=0 for x in candidates):raise ValueError("complete positive remote sizes are required")
 rql=rql_expected_bytes(vocabulary_size);bf16=math.ceil(reference_weights*1.05)+GIB;conversion=max(4*GIB,math.ceil(reference_weights*.10));cache=max(2*GIB,math.ceil(reference_weights*.02));compact=max(GIB,math.ceil(rql*.01));largest=max(candidates,key=lambda x:x["remote_bytes"])
 reference=reference_weights+bf16+conversion+cache;determinism=bf16+2*rql;candidate=(bf16 if retain_bf16_during_candidates else 0)+rql+largest["remote_bytes"]+rql+compact+GIB;peak=max(reference,determinism,candidate)
 return {"native_hf_reference_weights":reference_weights,"converted_bf16_gguf":bf16,"bf16_rql_a":rql,"bf16_rql_b":rql,"candidate_gguf_max":largest["remote_bytes"],"candidate_rql":rql,"temporary_conversion_overhead":conversion,"hf_cache_overhead":cache,"compact_output_overhead":compact,"safety_margin":safety_margin,"reference_stage_peak":reference,"determinism_stage_peak":determinism,"candidate_stage_peak":candidate,"candidate_peak_artifact":largest["label"],"retain_bf16_during_candidates":retain_bf16_during_candidates,"expected_global_peak":peak,"expected_free_at_peak":free_bytes-peak,"pass":free_bytes-peak>=safety_margin}
def gib(x:int)->str:return f"{x/GIB:.3f}"
def lifecycle_table(artifacts:list[dict],d:dict):
 release=not d.get("retain_bf16_during_candidates",False)
 det_cleanup="BF16 RQL B + BF16 GGUF" if release else "BF16 RQL B"
 det_retain="canonical BF16 RQL + reference-preflight manifest" if release else "BF16 GGUF + canonical BF16 RQL + reference-preflight manifest"
 candidate_retain="canonical BF16 RQL + reference-preflight manifest + compact evidence only" if release else "BF16 GGUF + canonical BF16 RQL + reference-preflight manifest + compact evidence only"
 lines=["DOWNLOAD_LIFECYCLE:",f"REFERENCE_DOWNLOAD\n  creates: native HF files + run cache ({d['native_hf_reference_weights']+d['hf_cache_overhead']} bytes)\n  deletes after: verified BF16 conversion/source audit",f"BF16_CONVERSION\n  creates: BF16 GGUF + scratch ({d['converted_bf16_gguf']+d['temporary_conversion_overhead']} bytes)\n  deletes after: source snapshot and conversion scratch; BF16 GGUF retained through determinism",f"REFERENCE_PREFLIGHT_MANIFEST\n  creates: compact tensor topology + tokenizer metadata/fixture expectations\n  retained after cleanup: reference-preflight manifest",f"REFERENCE_CAPTURE_A\n  creates: BF16 RQL A ({d['bf16_rql_a']} bytes)\n  retained after cleanup: BF16 GGUF + RQL A",f"REFERENCE_CAPTURE_B\n  creates: BF16 RQL B ({d['bf16_rql_b']} bytes)\n  retained until: bitwise determinism validation",f"REFERENCE_DETERMINISM\n  deletes after: {det_cleanup}\n  retained after cleanup: {det_retain}"]
 for i,x in enumerate(artifacts,1):lines.append(f"ARTIFACT_{i}\n  creates: {x['label']} GGUF ({x['remote_bytes']} bytes) + candidate RQL ({d['candidate_rql']} bytes)\n  deletes after: canonical analysis + independent validation + compact checkpoint\n  retained after cleanup: {candidate_retain}")
 lines.append("FINAL_FREEZE\n  deletes after: canonical BF16 RQL, run caches, corpus scratch (and BF16 GGUF if cleanup=none previously retained it)\n  retained large-file bytes after aggressive cleanup: 0")
 return "\n".join(lines)

def dry_run(a,out):
 try:
  free=shutil.disk_usage(out.parent if out.parent.exists() else ROOT).free;v,_=next_registry();identity=qwen.resolve_revision(a.reference);ref=qwen.reference_metadata(identity);artifacts=[]
  for label,url in a.artifact:artifacts.append({"label":label,**qwen.resolve_artifact(url)})
  d=disk_model(ref["weight_bytes_total"],ref["vocabulary_size"],artifacts,free,int(a.min_free_gib*GIB),retain_bf16_during_candidates=(a.cleanup=="none"))
  print(f"BENCHMARK: {a.benchmark_id}\nRESOLVED_ADAPTER: qwen\nARTIFACT_COUNT: {len(artifacts)}\nLLAMA_CPP_COMMIT_REQUIRED: {COMMIT}\nEXECUTION_CONTRACT: {CONTRACT['version']}\nREFERENCE_GPU_LAYERS: {a.reference_gpu_layers}\nARTIFACT_GPU_LAYERS: -1\nHCDFR_THRESHOLDS: 1,2,4\nBOOTSTRAP: replicates={REPS} seed={SEED} unit=segment")
  print(f"REFERENCE_REPO: {identity['repository_id']}\nREFERENCE_REVISION: {identity['commit']}\nREFERENCE_REVISION_IMMUTABLE: true\nREFERENCE_WEIGHT_FILES: {json.dumps(ref['weight_files'])}\nREFERENCE_WEIGHT_BYTES_TOTAL: {ref['weight_bytes_total']}\nREFERENCE_CONFIG_FILES: {json.dumps(ref['config_files'])}\nREFERENCE_TOKENIZER_FILES: {json.dumps(ref['tokenizer_files'])}\nREFERENCE_VOCABULARY_SIZE: {ref['vocabulary_size']}\nRQL_EXPECTED_BYTES: {d['bf16_rql_a']}")
  for x in artifacts:
   print(f"ARTIFACT: {x['label']}\nINPUT_URL: {x['input_url']}\nRESOLVED_URL: {x['resolved_url']}\nRESOLVED_REVISION: {x['resolved_revision']}\nRESOLVED_REVISION_IMMUTABLE: true\nREMOTE_BYTES: {x['remote_bytes']}\nREMOTE_GIB: {gib(x['remote_bytes'])}\nREMOTE_ETAG: {x['remote_etag']}\nREMOTE_LFS_SHA256: {x['remote_lfs_sha256'] or 'unavailable'}")
  print(f"DISK_OBJECT_NATIVE_HF_BYTES: {d['native_hf_reference_weights']}\nDISK_OBJECT_BF16_GGUF_BYTES: {d['converted_bf16_gguf']}\nDISK_OBJECT_BF16_RQL_A_BYTES: {d['bf16_rql_a']}\nDISK_OBJECT_BF16_RQL_B_BYTES: {d['bf16_rql_b']}\nDISK_OBJECT_LARGEST_CANDIDATE_GGUF_BYTES: {d['candidate_gguf_max']}\nDISK_OBJECT_CANDIDATE_RQL_BYTES: {d['candidate_rql']}\nDISK_OBJECT_CONVERSION_OVERHEAD_BYTES: {d['temporary_conversion_overhead']}\nDISK_OBJECT_HF_CACHE_OVERHEAD_BYTES: {d['hf_cache_overhead']}\nDISK_OBJECT_COMPACT_OVERHEAD_BYTES: {d['compact_output_overhead']}")
  print(f"FREE_DISK_BYTES: {free}\nFREE_DISK_GIB: {gib(free)}\nSAFETY_MARGIN_GIB: {a.min_free_gib:.3f}\nREFERENCE_STAGE_PEAK_GIB: {gib(d['reference_stage_peak'])}\nDETERMINISM_STAGE_PEAK_GIB: {gib(d['determinism_stage_peak'])}\nCANDIDATE_STAGE_PEAK_GIB: {gib(d['candidate_stage_peak'])}\nCANDIDATE_PEAK_ARTIFACT: {d['candidate_peak_artifact']}\nEXPECTED_GLOBAL_PEAK_GIB: {gib(d['expected_global_peak'])}\nEXPECTED_FREE_AT_PEAK_GIB: {gib(d['expected_free_at_peak'])}\nDISK_PREFLIGHT: {'PASS' if d['pass'] else 'FAIL'}")
  print(lifecycle_table(artifacts,d));print(f"TARGETS: run-spec.json run-state.json result.csv verified-result.json forensic-evidence-v1.json leaderboard-entry.json leaderboard-index-v{v}.json release-manifest.json compact/*.npz web-package/*")
  print("DRY_RUN_NETWORK_METADATA_ONLY: true\nLARGE_DOWNLOADS_PERFORMED: 0\nINFERENCE_PERFORMED: 0\nDELETIONS_PERFORMED: 0")
  return 0 if d["pass"] else 1
 except Exception as e:
  print(f"DRY_RUN_PREFLIGHT: FAIL\nERROR: {e}\nDRY_RUN_NETWORK_METADATA_ONLY: true\nLARGE_DOWNLOADS_PERFORMED: 0\nINFERENCE_PERFORMED: 0\nDELETIONS_PERFORMED: 0",file=sys.stderr);return 1
def main(argv=None):
 a=parse_args(argv)
 CONTRACT.pop("gpu_layers",None)
 CONTRACT["reference_gpu_layers"]=a.reference_gpu_layers
 CONTRACT["artifact_gpu_layers"]=-1
 CONTRACT["placement_policy"]="reference_partial_offload_artifact_full_gpu"
 CONTRACT["version"]="QuantBench-V1" if a.reference_gpu_layers==-1 else f"QuantBench-V1-REFNGL{a.reference_gpu_layers}"
 out=(a.output_dir or ROOT/"artifact-bench/runs"/a.benchmark_id).resolve()
 if a.dry_run:return dry_run(a,out)
 out.mkdir(parents=True,exist_ok=True);work=(a.work_dir or out/"work").resolve();work.mkdir(parents=True,exist_ok=True);st=State(out/"run-state.json",a.resume);st.data["cleanup_roots"]=sorted(set(st.data.get("cleanup_roots",[])+[str(work)]));st.save();stage="CREATED";margin=int(a.min_free_gib*GIB)
 try:
  if sha256(PUBLIC/"leaderboard-index-v1.json")!=V1_SHA:raise RuntimeError("frozen leaderboard v1 hash mismatch")
  qwen.verify_llama(a.llama_cpp_dir)
  if (ROOT/"third_party/LLAMA_CPP_COMMIT").read_text().strip()!=COMMIT:raise RuntimeError("QuantBench probe source commit pin mismatch")
  cache=(ROOT/"build-cuda/CMakeCache.txt").read_text()
  if "GGML_CUDA:BOOL=ON" not in cache:raise RuntimeError("QuantBench probe was not built with CUDA")
  os.environ.setdefault("CUDA_VISIBLE_DEVICES","0");probe=ROOT/"build-cuda/ramgpt-logit-probe";tokenizer=ROOT/"build/ramgpt-tokenize"
  if not probe.exists() or not tokenizer.exists():raise RuntimeError("required QuantBench CUDA probe/tokenizer binaries are absent")
  reg_path=out/"qwen3-4b-regression.json"
  if not st.done("QWEN3_4B_FROZEN_REGRESSION"):stage="QWEN3_4B_FROZEN_REGRESSION";atomic_json(reg_path,frozen_regression());st.complete(stage,[reg_path]);print("QWEN3_4B_FROZEN_REGRESSION: PASS")
  identity_path=out/"logs/reference-identity.json"
  if not st.done("REFERENCE_RESOLVED"):stage="REFERENCE_RESOLVED";print("[1/8] Reference preflight");identity=qwen.resolve_revision(a.reference);atomic_json(identity_path,identity);st.complete(stage,[identity_path])
  identity=json.loads(identity_path.read_text());spec_path=out/"run-spec.json"
  if not st.done("CREATED"):
   pass
  remote_path=out/"logs/remote-preflight.json"
  if not st.done("INPUT_PREFLIGHT"):
   stage="INPUT_PREFLIGHT";refmeta=qwen.reference_metadata(identity);remote_artifacts=[{"label":label,**qwen.resolve_artifact(url)} for label,url in a.artifact];free=shutil.disk_usage(work).free;disk=disk_model(refmeta["weight_bytes_total"],refmeta["vocabulary_size"],remote_artifacts,free,margin,retain_bf16_during_candidates=(a.cleanup=="none"));atomic_json(remote_path,{"reference":refmeta,"artifacts":remote_artifacts,"disk":disk});print(lifecycle_table(remote_artifacts,disk))
   if not disk["pass"]:raise RuntimeError("global disk preflight failed")
   st.complete(stage,[remote_path])
  remote=json.loads(remote_path.read_text());remote_artifacts=remote["artifacts"]
  spec={"schema":"ramgpt-artifact-bench-run-spec-v1","reference_url":a.reference,"resolved_reference_revision":identity["commit"],"artifacts":[{"label":x["label"],"input_url":x["input_url"],"resolved_url":x["resolved_url"],"resolved_revision":x["resolved_revision"]} for x in remote_artifacts],"quantization_label":a.quantization_label,"benchmark_id":a.benchmark_id,"model":identity["repository_id"],"adapter":"qwen-v1","execution_contract":CONTRACT,"reference_vocabulary_size":remote["reference"]["vocabulary_size"],"cleanup_policy":a.cleanup,"min_free_gib":a.min_free_gib,"tool_versions":{"python":sys.version.split()[0]},"llama_cpp_commit":COMMIT,"start_time":st.data["created_at"]};atomic_json(spec_path,spec)
  if not st.done("CREATED"):st.complete("CREATED",[spec_path,reg_path,identity_path,remote_path])
  snapshot=work/"hf-cache/snapshot";snapshot_manifest=out/"logs/reference-snapshot.json"
  if not st.done("REFERENCE_DOWNLOADED"):
   stage="REFERENCE_DOWNLOADED";expected=sum(x for x in identity.get("file_sizes",{}).values() if isinstance(x,int));check_space(work,expected,stage,margin,0,expected);envhome=work/"hf-cache";os.environ.update({"HF_HOME":str(envhome),"HUGGINGFACE_HUB_CACHE":str(envhome/"hub"),"TRANSFORMERS_CACHE":str(envhome/"transformers")});manifest=qwen.download_snapshot(identity,snapshot,download);manifest["native_dtype"]=qwen.validate_native_dtype(snapshot);atomic_json(snapshot_manifest,manifest);st.complete(stage,[snapshot_manifest]+[snapshot/x["path"] for x in manifest["files"]])
  source_bytes=sum(x["bytes"] for x in json.loads(snapshot_manifest.read_text())["files"]);bf16=work/"reference-bf16.gguf";conversion=out/"logs/reference-conversion.json"
  if not st.done("REFERENCE_CONVERTED"):
   stage="REFERENCE_CONVERTED";check_space(work,source_bytes*2,stage,margin,source_bytes,source_bytes);meta=qwen.convert(snapshot,bf16,a.llama_cpp_dir,out/"logs/reference-conversion.log");audit=qwen.inspect_gguf(bf16,a.llama_cpp_dir)
   bad=[x for x in audit["tensors"] if not any(t in x["type"].upper() for t in ("BF16","F16","F32"))]
   if bad:raise RuntimeError(f"converted reference contains non-native tensor types: {bad[:3]}")
   atomic_json(conversion,{**meta,"sha256":sha256(bf16),"audit":audit});st.complete(stage,[bf16,conversion])
  if a.cleanup in ("safe","aggressive") and snapshot.exists():delete_safe(snapshot,st,"REFERENCE_CONVERTED")
  conversion_data=json.loads(conversion.read_text());refaudit=conversion_data["audit"];refsha=conversion_data["sha256"];vocab=refaudit["fields"]["tokenizer.ggml.token_count"]
  if not isinstance(vocab,int) or vocab<=0:raise RuntimeError("reference GGUF vocabulary size unavailable")
  if vocab!=remote["reference"]["vocabulary_size"]:raise RuntimeError(f"reference vocabulary mismatch: metadata={remote['reference']['vocabulary_size']} GGUF={vocab}")
  corpus=work/"qwen-fidelity-v2.rqseg";corpus_manifest=out/"logs/corpus.json"
  if not st.done("CORPUS_READY"):
   stage="CORPUS_READY";cm=qwen.build_corpus(bf16,tokenizer,ROOT/"data/fidelity-v2-candidate/source",ROOT/"data/fidelity-v2-candidate/source-plan.json",corpus);atomic_json(corpus_manifest,cm);st.complete(stage,[corpus,corpus_manifest])
  fixtures=[p.read_text()[:500] for p in sorted((ROOT/"data/fidelity-v2-candidate/source").iterdir())[:32]]+["Qwen tokenizer parity 中文 test <|im_start|>","def f(x): return x + 1",""," ","https://example.org/a?q=雪","P(A|B)=P(B|A)P(A)/P(B)","{\"ok\":true}","你好，世界！"]
  reference_preflight_path=out/"reference-preflight.json"
  if not st.done("REFERENCE_PREFLIGHT_MANIFEST"):
   stage="REFERENCE_PREFLIGHT_MANIFEST"
   with tempfile.TemporaryDirectory(prefix="qwen-reference-preflight-",dir=work) as td:reference_preflight=qwen.build_reference_preflight(refaudit,bf16,a.llama_cpp_dir,tokenizer,fixtures,Path(td),identity["commit"],refsha)
   atomic_json(reference_preflight_path,reference_preflight);st.complete(stage,[reference_preflight_path])
  reference_preflight=json.loads(reference_preflight_path.read_text());qwen.validate_reference_preflight(reference_preflight)
  rql_bytes=321+N*vocab*4;ra=work/"reference-a";rb=work/"reference-b"
  for name,prefix in (("REFERENCE_CAPTURE_A",ra),("REFERENCE_CAPTURE_B",rb)):
   if not st.done(name):stage=name;print("[2/8] Reference capture");check_space(work,rql_bytes*2,name,margin,bf16.stat().st_size,rql_bytes);qwen.capture(bf16,corpus,prefix,probe,False,out/f"logs/{name.lower()}.log",a.reference_gpu_layers);rql_header(prefix.with_suffix(".rql"),a.reference_gpu_layers);st.complete(name,[prefix.with_suffix(".rql"),prefix.with_suffix(".positions.jsonl"),prefix.with_suffix(".execution.json")])
  det=out/"logs/reference-determinism.json"
  if not st.done("REFERENCE_VERIFIED"):
   stage="REFERENCE_VERIFIED";checks={"rql_a":sha256(ra.with_suffix(".rql")),"rql_b":sha256(rb.with_suffix(".rql")),"positions_a":sha256(ra.with_suffix(".positions.jsonl")),"positions_b":sha256(rb.with_suffix(".positions.jsonl"))};checks["status"]="PASS" if checks["rql_a"]==checks["rql_b"] and checks["positions_a"]==checks["positions_b"] else "FAIL";atomic_json(det,checks)
   if checks["status"]!="PASS":raise RuntimeError("two BF16 reference captures are not bitwise identical")
   st.complete(stage,[det,ra.with_suffix(".rql"),ra.with_suffix(".positions.jsonl")])
  reference_ready=out/"logs/reference-ready-for-artifacts.json"
  if not st.done("REFERENCE_READY_FOR_ARTIFACTS"):
   stage="REFERENCE_READY_FOR_ARTIFACTS"
   if not st.done("REFERENCE_VERIFIED") or not st.done("REFERENCE_PREFLIGHT_MANIFEST"):raise RuntimeError("reference cannot be compacted before determinism and reference-preflight manifest complete")
   ready={"status":"PASS","canonical_bf16_rql_sha256":sha256(ra.with_suffix(".rql")),"canonical_positions_sha256":sha256(ra.with_suffix(".positions.jsonl")),"reference_preflight_sha256":sha256(reference_preflight_path),"bf16_gguf_sha256":refsha};atomic_json(reference_ready,ready);st.complete(stage,[reference_ready,det,reference_preflight_path,ra.with_suffix(".rql"),ra.with_suffix(".positions.jsonl")])
  if a.cleanup in ("safe","aggressive"):
   for p in (rb.with_suffix(".rql"),rb.with_suffix(".positions.jsonl"),rb.with_suffix(".execution.json"),bf16):
    if p.exists():delete_safe(p,st,"REFERENCE_READY_FOR_ARTIFACTS")
  rows=[]
  for i,(label,url) in enumerate(a.artifact,1):
   key=f"ARTIFACT_{i:03d}";record=out/f"logs/artifact-{i:03d}-record.json";meta_path=out/f"logs/artifact-{i:03d}-download.json";audit_path=out/f"logs/artifact-{i:03d}-preflight.json";metrics_path=out/f"logs/artifact-{i:03d}-metrics.json";validation_path=out/f"logs/artifact-{i:03d}-validation.json";compact=out/f"compact/{slug(label)}.npz";cand=work/f"candidate-{i:03d}.gguf";prefix=work/f"candidate-{i:03d}"
   print(f"[3/8] Artifact {i}/{len(a.artifact)}")
   if not st.done(key+"_DOWNLOADED"):
    stage=key+"_DOWNLOADED";resolved=remote_artifacts[i-1];size=resolved["remote_bytes"];check_space(work,size+rql_bytes,stage,margin,size,rql_bytes);meta=download(resolved["resolved_url"],cand,size)
    if resolved.get("remote_lfs_sha256") and meta["sha256"]!=resolved["remote_lfs_sha256"]:raise RuntimeError(f"downloaded artifact SHA differs from frozen LFS identity: {label}")
    atomic_json(meta_path,{"label":label,"input_url":url,"resolved_url":resolved["resolved_url"],"resolved_revision":resolved["resolved_revision"],**meta});st.complete(stage,[cand,meta_path])
   download_meta=json.loads(meta_path.read_text())
   if not st.done(key+"_PREFLIGHTED"):
    stage=key+"_PREFLIGHTED";caudit=qwen.inspect_gguf(cand,a.llama_cpp_dir);topology=qwen.topology_compatible(reference_preflight["gguf_audit"],caudit);quantcheck=qwen.validate_quantization(caudit,a.quantization_label)
    with tempfile.TemporaryDirectory(prefix="qwen-tokenizer-",dir=work) as td:
     tokencheck=qwen.tokenizer_compatible_manifest(reference_preflight,cand,a.llama_cpp_dir,tokenizer,fixtures,Path(td),json.loads(corpus_manifest.read_text()),ROOT/"data/fidelity-v2-candidate/source")
     tokencheck["reference_preflight_sha256"]=sha256(reference_preflight_path)
    atomic_json(audit_path,{"status":"PASS","topology":topology,"tokenizer":tokencheck,"quantization":quantcheck,"gguf":{"tensor_count":caudit["tensor_count"],"tensor_types":caudit["tensor_types"],"fields":caudit["fields"]}});st.complete(stage,[cand,audit_path])
   if not st.done(key+"_CAPTURED"):
    stage=key+"_CAPTURED";check_space(work,rql_bytes,stage,margin,cand.stat().st_size,rql_bytes);qwen.capture(cand,corpus,prefix,probe,True,out/f"logs/artifact-{i:03d}-capture.log",-1);rql_header(prefix.with_suffix(".rql"),-1);st.complete(stage,[prefix.with_suffix(".rql"),prefix.with_suffix(".positions.jsonl"),prefix.with_suffix(".execution.json")])
   if not st.done(key+"_ANALYZED"):
    stage=key+"_ANALYZED";metrics=analyze_pair(ra.with_suffix(".rql"),prefix.with_suffix(".rql"),ra.with_suffix(".positions.jsonl"),prefix.with_suffix(".positions.jsonl"),compact);atomic_json(metrics_path,metrics);st.complete(stage,[compact,metrics_path])
   metrics=json.loads(metrics_path.read_text())
   if not st.done(key+"_VALIDATED"):
    stage=key+"_VALIDATED";ind=analyze_pair(ra.with_suffix(".rql"),prefix.with_suffix(".rql"),ra.with_suffix(".positions.jsonl"),prefix.with_suffix(".positions.jsonl"),compact,True)
    if abs(metrics["mean_kl_bf16_to_artifact"]-ind["mean_kl_bf16_to_artifact"])>=1e-10:raise RuntimeError(f"independent validation mismatch for {label}: mean KL")
    for k in ("top1_flips","hcdfr","max_flipped_bf16_margin"):
     if metrics[k]!=ind[k]:raise RuntimeError(f"independent validation mismatch for {label}: {k}")
    atomic_json(validation_path,{"status":"PASS","mean_kl_absolute_tolerance":1e-10,"independent":ind});audit=json.loads(audit_path.read_text());rec=artifact_record(label,url,download_meta,metrics);rec["resolved_url"]=download_meta["resolved_url"];rec["resolved_revision"]=download_meta["resolved_revision"];rec["gguf_audit"]=audit["gguf"];atomic_json(record,rec);st.complete(stage,[record,compact,validation_path])
   rows.append(json.loads(record.read_text()))
   if a.cleanup in ("safe","aggressive"):
    for p in (cand,prefix.with_suffix(".rql"),prefix.with_suffix(".positions.jsonl"),prefix.with_suffix(".execution.json")):
     if p.exists():delete_safe(p,st,key+"_VALIDATED")
  u=bootstrap(rows);stage="RANKING_COMPLETE";print("[7/8] Ranking / uncertainty")
  if not st.done(stage):write_csv(out/"result.csv",a.benchmark_id,identity["repository_id"],a.quantization_label,rows,u,refsha);st.complete(stage,[out/"result.csv"])
  stage="FREEZE_COMPLETE";print("[8/8] Freeze / web package")
  if not st.done(stage):freeze(out,spec,rows,u,refsha);st.complete(stage,[out/"verified-result.json",out/"forensic-evidence-v1.json",out/"release-manifest.json"])
  if a.cleanup=="aggressive":
   for p in (ra.with_suffix(".rql"),ra.with_suffix(".positions.jsonl"),ra.with_suffix(".execution.json"),bf16,corpus,work/"hf-cache",work/"tokenize-temp"):
    if p.exists():delete_safe(p,st,"FREEZE_COMPLETE")
   audit_large_files(work);audit_large_files(out)
  st.data["status"]="DONE";st.save();print(f"BENCHMARK: {a.benchmark_id}\nWINNER: {u['winner']}\nCSV: {out/'result.csv'}\nWEB_PACKAGE: {out/'web-package'}\nSPACE_FREED: {st.data['space_freed_bytes']}\nQWEN3_4B_FROZEN_REGRESSION: PASS\nQWEN_LIVE_PIPELINE_READY: PASS\nSTATUS: PASS");return 0
 except Exception as e:st.fail(stage,e);print(f"STATUS: FAIL\nSTAGE: {stage}\nERROR: {e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
