"""Qwen-family adapter V1 for the frozen QuantBench execution contract."""
from __future__ import annotations
import hashlib, json, os, re, struct, subprocess, sys, urllib.parse, urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

COMMIT="70adb1b4cea5ee39f867792c78dc59320921eda7"; ALLOW=("config.json","generation_config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","merges.txt","vocab.json")
WEIGHT_RE=re.compile(r"(?:model(?:-\d{5}-of-\d{5})?\.safetensors|model\.safetensors\.index\.json)$")

def parse_hf_url(url:str)->tuple[str,str|None]:
 p=urllib.parse.urlparse(url);parts=[x for x in p.path.split("/") if x]
 if p.scheme not in ("http","https") or p.netloc.lower() not in ("huggingface.co","www.huggingface.co") or len(parts)<2:raise ValueError("Qwen V1 requires a canonical huggingface.co/<owner>/<repo> URL")
 repo="/".join(parts[:2]);rev=None
 if len(parts)>=4 and parts[2] in ("tree","resolve"):rev=parts[3]
 if "qwen" not in repo.casefold():raise ValueError("unsupported model: Qwen adapter accepts Qwen repositories only")
 return repo,rev
def api_json(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"RAMGPT-QuantBench/1"}),timeout=30) as r:return json.load(r)
def small_json(url:str,max_bytes:int=8<<20):
 req=urllib.request.Request(url,headers={"User-Agent":"RAMGPT-QuantBench/1","Accept":"application/json"})
 with urllib.request.urlopen(req,timeout=30) as r:
  length=int(r.headers.get("Content-Length") or 0)
  if length and length>max_bytes:raise RuntimeError(f"metadata response too large: {length}")
  body=r.read(max_bytes+1)
  if len(body)>max_bytes:raise RuntimeError("metadata response exceeded limit")
 return json.loads(body)
def resolve_revision(url:str)->dict[str,Any]:
 repo,requested=parse_hf_url(url);rev=requested or "main";info=api_json(f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(rev,safe='')}?blobs=true")
 commit=info.get("sha")
 if not commit or not re.fullmatch(r"[0-9a-f]{40}",commit):raise RuntimeError("Hugging Face did not return an immutable 40-hex commit")
 if re.fullmatch(r"[0-9a-f]{40}",rev) and commit!=rev:raise RuntimeError("immutable reference revision resolved to a different commit")
 siblings=[x["rfilename"] for x in info.get("siblings",[])];files=[x for x in siblings if x in ALLOW or WEIGHT_RE.search(x)]
 if "config.json" not in files or "tokenizer.json" not in files or not any(WEIGHT_RE.search(x) and x.endswith(".safetensors") for x in files):raise RuntimeError("Qwen repository lacks required config/tokenizer/native safetensors")
 sizes={x["rfilename"]:x.get("size") or x.get("lfs",{}).get("size") for x in info.get("siblings",[]) if x["rfilename"] in files};oids={x["rfilename"]:x.get("lfs",{}).get("oid") for x in info.get("siblings",[]) if x["rfilename"] in files and x.get("lfs",{}).get("oid")}
 return {"repository_id":repo,"requested_revision":rev,"commit":commit,"files":sorted(files),"file_sizes":sizes,"lfs_sha256":oids,"source_url":url}
def reference_metadata(identity:dict[str,Any])->dict[str,Any]:
 base=f"https://huggingface.co/{identity['repository_id']}/resolve/{identity['commit']}"
 cfg=small_json(base+"/config.json");tok=small_json(base+"/tokenizer_config.json") if "tokenizer_config.json" in identity["files"] else {}
 vocab=cfg.get("vocab_size") or cfg.get("text_config",{}).get("vocab_size") or tok.get("vocab_size")
 if not isinstance(vocab,int) or vocab<=0:raise RuntimeError("exact vocabulary size unavailable from pinned reference metadata")
 weights=[x for x in identity["files"] if x.endswith(".safetensors")];unknown=[x for x in weights if not isinstance(identity["file_sizes"].get(x),int)]
 if unknown:raise RuntimeError(f"reference weight sizes unavailable: {unknown}")
 configs=[x for x in identity["files"] if x.endswith("config.json") or x=="config.json"]
 tokenizers=[x for x in identity["files"] if x.startswith(("tokenizer","special_tokens")) or x in ("merges.txt","vocab.json")]
 return {"vocabulary_size":vocab,"weight_files":weights,"weight_bytes_total":sum(identity["file_sizes"][x] for x in weights),"config_files":configs,"tokenizer_files":tokenizers,"config":cfg}
def parse_artifact_url(url:str)->tuple[str,str,str]:
 p=urllib.parse.urlparse(url);parts=[x for x in p.path.split("/") if x]
 if p.scheme not in ("http","https") or p.netloc.lower() not in ("huggingface.co","www.huggingface.co") or len(parts)<5 or parts[2] not in ("resolve","blob"):raise ValueError("Qwen V1 artifact must be a Hugging Face file URL")
 return "/".join(parts[:2]),parts[3],"/".join(parts[4:])
def resolve_artifact(url:str)->dict[str,Any]:
 repo,requested,name=parse_artifact_url(url);info=api_json(f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(requested,safe='')}?blobs=true");commit=info.get("sha")
 if not commit or not re.fullmatch(r"[0-9a-f]{40}",commit):raise RuntimeError(f"artifact revision could not be frozen: {repo}@{requested}")
 if re.fullmatch(r"[0-9a-f]{40}",requested) and commit!=requested:raise RuntimeError("immutable artifact revision resolved to a different commit")
 matches=[x for x in info.get("siblings",[]) if x.get("rfilename")==name]
 if len(matches)!=1:raise RuntimeError(f"artifact file is absent or ambiguous: {repo}/{name}")
 f=matches[0];size=f.get("size") or f.get("lfs",{}).get("size");oid=f.get("lfs",{}).get("oid")
 if not isinstance(size,int) or size<=0:raise RuntimeError(f"artifact remote size unavailable: {repo}/{name}")
 resolved=f"https://huggingface.co/{repo}/resolve/{commit}/{urllib.parse.quote(name,safe='/')}"
 # LFS OID is the immutable whole-file SHA-256 when supplied by Hugging Face.
 return {"input_url":url,"repository_id":repo,"filename":name,"requested_revision":requested,"resolved_revision":commit,"resolved_revision_immutable":True,"resolved_url":resolved,"remote_bytes":size,"remote_etag":oid or f.get("oid"),"remote_lfs_sha256":oid.removeprefix("sha256:") if oid else None}
def _file_sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8<<20),b""):
   h.update(chunk)
 return h.hexdigest()

def _reference_reuse_authority(identity:dict[str,Any]):
 raw_path=os.environ.get("RAMGPT_REFERENCE_REUSE_MANIFEST")
 if not raw_path:return None

 p=Path(raw_path).resolve()
 if not p.is_file():
  raise RuntimeError(f"reference reuse manifest is absent: {p}")

 raw=p.read_bytes()
 manifest_sha=hashlib.sha256(raw).hexdigest()

 pinned=os.environ.get("RAMGPT_REFERENCE_REUSE_MANIFEST_SHA256")
 if pinned and manifest_sha!=pinned:
  raise RuntimeError(
   "reference reuse manifest SHA-256 differs from pinned environment value"
  )

 try:
  d=json.loads(raw)
 except Exception as e:
  raise RuntimeError(
   f"reference reuse manifest is invalid JSON: {e}"
  ) from e

 if d.get("repository_id")!=identity["repository_id"]:
  raise RuntimeError(
   "reference reuse manifest repository differs from current reference"
  )

 if d.get("commit")!=identity["commit"]:
  raise RuntimeError(
   "reference reuse manifest commit differs from current immutable reference"
  )

 rows=d.get("files")
 if not isinstance(rows,list):
  raise RuntimeError("reference reuse manifest files list is absent")

 by_name={}
 for rec in rows:
  if not isinstance(rec,dict):
   raise RuntimeError("reference reuse manifest contains invalid file record")

  name=rec.get("path")
  size=rec.get("bytes")
  digest=rec.get("sha256")

  if not isinstance(name,str) or not name:
   raise RuntimeError("reference reuse manifest contains invalid path")

  if name in by_name:
   raise RuntimeError(
    f"reference reuse manifest contains duplicate path: {name}"
   )

  if not isinstance(size,int) or size<0:
   raise RuntimeError(
    f"reference reuse manifest contains invalid byte size: {name}"
   )

  if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest):
   raise RuntimeError(
    f"reference reuse manifest contains invalid SHA-256: {name}"
   )

  by_name[name]={
   "bytes":size,
   "sha256":digest,
  }

 current=set(identity["files"])
 historical=set(by_name)

 if current!=historical:
  missing=sorted(current-historical)
  extra=sorted(historical-current)
  raise RuntimeError(
   f"reference reuse manifest file set differs: "
   f"missing={missing} extra={extra}"
  )

 for name in identity["files"]:
  current_size=identity.get("file_sizes",{}).get(name)
  historical_size=by_name[name]["bytes"]

  if isinstance(current_size,int) and current_size!=historical_size:
   raise RuntimeError(
    f"reference reuse manifest size differs from current metadata: {name}"
   )

 print(
  f"REFERENCE_REUSE_AUTHORITY: PASS "
  f"repository={identity['repository_id']} "
  f"commit={identity['commit']} "
  f"files={len(by_name)} "
  f"manifest_sha256={manifest_sha}",
  flush=True,
 )

 return {
  "path":str(p),
  "sha256":manifest_sha,
  "repository_id":d["repository_id"],
  "commit":d["commit"],
  "files":by_name,
 }

def download_snapshot(identity:dict[str,Any],dest:Path,download_fn)->dict[str,Any]:
 dest.mkdir(parents=True,exist_ok=True)
 records=[]
 authority=_reference_reuse_authority(identity)

 for name in identity["files"]:
  target=dest/name
  target.parent.mkdir(parents=True,exist_ok=True)

  url=(
   f"https://huggingface.co/{identity['repository_id']}"
   f"/resolve/{identity['commit']}/{urllib.parse.quote(name)}"
  )

  size=identity.get("file_sizes",{}).get(name)

  hf_expected=identity.get("lfs_sha256",{}).get(name)
  hf_expected=hf_expected.removeprefix("sha256:") if hf_expected else None

  historical=None
  if authority:
   historical=authority["files"][name]

  if hf_expected:
   expected_hex=hf_expected
   hash_authority="hf_lfs"

   if historical and historical["sha256"]!=hf_expected:
    raise RuntimeError(
     f"HF LFS SHA differs from historical successful snapshot: {name}"
    )

  elif historical:
   expected_hex=historical["sha256"]
   hash_authority="historical_successful_snapshot"

  else:
   expected_hex=None
   hash_authority="download_observed_only"

  if historical and isinstance(size,int) and historical["bytes"]!=size:
   raise RuntimeError(
    f"historical snapshot size differs from current metadata: {name}"
   )

  meta=None

  if (
   target.exists()
   and isinstance(size,int)
   and target.stat().st_size==size
   and expected_hex
  ):
   got=_file_sha256(target)

   if got==expected_hex:
    meta={
     "bytes":size,
     "sha256":got,
     "reused_verified":True,
     "hash_authority":hash_authority,
    }

    print(
     f"REFERENCE_REUSE_VERIFIED: {name} "
     f"bytes={size} "
     f"sha256={got} "
     f"authority={hash_authority}",
     flush=True,
    )

   else:
    print(
     f"REFERENCE_REUSE_REJECTED: {name} "
     f"reason=sha256_mismatch",
     flush=True,
    )

  elif target.exists():
   if isinstance(size,int) and target.stat().st_size!=size:
    reason="size_mismatch"
   elif not expected_hex:
    reason="no_trusted_sha256"
   else:
    reason="not_reusable"

   print(
    f"REFERENCE_REUSE_REJECTED: {name} reason={reason}",
    flush=True,
   )

  if meta is None:
   meta=download_fn(url,target,size)
   meta["reused_verified"]=False
   meta["hash_authority"]=hash_authority

   if expected_hex and meta["sha256"]!=expected_hex:
    raise RuntimeError(
     f"downloaded reference SHA-256 differs from trusted identity: {name}"
    )

   if expected_hex:
    print(
     f"REFERENCE_DOWNLOAD_VERIFIED: {name} "
     f"bytes={meta['bytes']} "
     f"sha256={meta['sha256']} "
     f"authority={hash_authority}",
     flush=True,
    )

  if expected_hex and meta["sha256"]!=expected_hex:
   raise RuntimeError(
    f"reference SHA-256 mismatch after acquisition: {name}"
   )

  records.append({
   "path":name,
   "source_sha256":hf_expected,
   "verified_sha256":expected_hex,
   **meta,
  })

 result={
  "repository_id":identity["repository_id"],
  "commit":identity["commit"],
  "files":records,
 }

 if authority:
  result["reuse_authority"]={
   "path":authority["path"],
   "sha256":authority["sha256"],
   "repository_id":authority["repository_id"],
   "commit":authority["commit"],
   "file_count":len(authority["files"]),
  }

 return result

def validate_native_dtype(snapshot:Path)->dict[str,Any]:
    """Audit native source precision from actual safetensors headers.

    config.json dtype is advisory. Some valid HF repositories omit torch_dtype /
    dtype entirely. The authoritative source audit is the tensor dtype encoded
    in each safetensors header.
    """
    cfg=json.loads((snapshot/"config.json").read_text())

    declared=str(
        cfg.get("torch_dtype")
        or cfg.get("dtype")
        or ""
    ).strip().lower()

    # Normalize common serialized spelling.
    if declared.startswith("torch."):
        declared=declared[len("torch."):]

    # If a config declaration exists, it must still be compatible.
    if declared and declared not in (
        "bfloat16","bf16","float16","f16"
    ):
        raise RuntimeError(
            f"native Qwen config declares unsupported source dtype: {declared!r}"
        )

    counts=Counter()
    tensors=0

    for path in sorted(snapshot.glob("*.safetensors")):
        with path.open("rb") as f:
            raw=f.read(8)
            if len(raw)!=8:
                raise RuntimeError(
                    f"truncated safetensors header: {path.name}"
                )

            n=struct.unpack("<Q",raw)[0]
            if n<=0 or n>100_000_000:
                raise RuntimeError(
                    f"invalid safetensors header: {path.name}"
                )

            header=json.loads(f.read(n))

        for name,meta in header.items():
            if name=="__metadata__":
                continue

            dtype=meta["dtype"]
            counts[dtype]+=1
            tensors+=1

    if not tensors:
        raise RuntimeError("native source contains no safetensors tensors")

    allowed={"BF16","F16","F32"}
    unsupported=set(counts)-allowed

    if unsupported:
        raise RuntimeError(
            f"native shard tensor dtype audit failed: {dict(counts)}"
        )

    # QuantBench's native reference must actually contain BF16/F16 tensors.
    # F32 auxiliary tensors are allowed.
    if not (counts.get("BF16",0) or counts.get("F16",0)):
        raise RuntimeError(
            f"native source contains no BF16/F16 tensors: {dict(counts)}"
        )

    return {
        "config_dtype":declared or None,
        "config_dtype_declared":bool(declared),
        "dtype_authority":"safetensors_headers",
        "tensor_dtypes":dict(counts),
        "tensor_count":tensors,
        "architectures":cfg.get("architectures"),
        "model_type":cfg.get("model_type"),
        "vocab_size":cfg.get("vocab_size"),
    }

def verify_llama(llama:Path):
 got=subprocess.check_output(["git","-C",str(llama),"rev-parse","HEAD"],text=True).strip()
 if got!=COMMIT:raise RuntimeError(f"llama.cpp commit mismatch: {got}")
def convert(snapshot:Path,out:Path,llama:Path,log:Path):
 verify_llama(llama);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+".partial")
 cmd=[sys.executable,str(llama/"convert_hf_to_gguf.py"),str(snapshot),"--outfile",str(tmp),"--outtype","bf16"]
 with log.open("w") as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 os.replace(tmp,out);return {"command":cmd,"bytes":out.stat().st_size}
def _reader(path:Path,llama:Path):
 sys.path.insert(0,str(llama/"gguf-py"));from gguf import GGUFReader
 return GGUFReader(path,"r")
def _plain(x):
 if hasattr(x,"tolist"):x=x.tolist()
 if isinstance(x,(list,tuple)):return [_plain(v) for v in x]
 if isinstance(x,dict):return {str(k):_plain(v) for k,v in x.items()}
 if isinstance(x,bytes):return x.decode("utf-8",errors="replace")
 return x.item() if hasattr(x,"item") else x

def _canonical_field_sha256(value:Any)->str:
 payload=json.dumps(_plain(value),ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
 return hashlib.sha256(payload).hexdigest()

def tokenizer_metadata_fingerprints(model:Path,llama:Path)->dict[str,Any]:
 r=_reader(model,llama);fields={}
 for key in sorted(k for k in r.fields if k.startswith("tokenizer.")):
  fields[key]=_canonical_field_sha256(r.fields[key].contents())
 if not fields:raise RuntimeError("reference/candidate GGUF has no tokenizer metadata")
 return {"field_count":len(fields),"field_sha256":fields}

def _read_token_ids(path:Path)->list[int]:
 b=path.read_bytes()
 if len(b)<24 or b[:8]!=b"RQTOKEN\0":raise RuntimeError(f"invalid tokenizer fixture output: {path}")
 n=struct.unpack_from("<Q",b,16)[0]
 if len(b)!=24+4*n:raise RuntimeError(f"tokenizer fixture output length mismatch: {path}")
 return list(struct.unpack_from(f"<{n}i",b,24))

def _run_fixture_token_ids(model:Path,tokenizer_bin:Path,fixtures:list[str],temp:Path,add:bool,tag:str)->list[list[int]]:
 temp.mkdir(parents=True,exist_ok=True);batch=temp/f"{tag}.tsv";outs=[];lines=[]
 for i,text in enumerate(fixtures):
  inp=temp/f"{tag}-{i}.txt";out=temp/f"{tag}-{i}.tokens";inp.write_text(text);outs.append(out);lines.append(f"{inp}\t{out}\n")
 batch.write_text("".join(lines));subprocess.run([str(tokenizer_bin),"--model",str(model),"--batch-list",str(batch),"--add-special",str(add).lower()],check=True,stdout=subprocess.DEVNULL)
 return [_read_token_ids(p) for p in outs]
def inspect_gguf(path:Path,llama:Path)->dict[str,Any]:
 r=_reader(path,llama);fields={k:_plain(r.fields[k].contents()) for k in r.fields if k in ("general.architecture","general.file_type","tokenizer.ggml.model","tokenizer.ggml.pre","tokenizer.ggml.add_bos_token","tokenizer.ggml.add_eos_token")};fields["tokenizer.ggml.token_count"]=len(r.fields["tokenizer.ggml.tokens"].contents()) if "tokenizer.ggml.tokens" in r.fields else None
 names={"0":"F32","1":"F16","2":"Q4_0","3":"Q4_1","6":"Q5_0","7":"Q5_1","8":"Q8_0","9":"Q8_1","10":"Q2_K","11":"Q3_K","12":"Q4_K","13":"Q5_K","14":"Q6_K","15":"Q8_K","16":"IQ2_XXS","17":"IQ2_XS","18":"IQ3_XXS","19":"IQ1_S","20":"IQ4_NL","21":"IQ3_S","22":"IQ2_S","23":"IQ4_XS","24":"I8","25":"I16","26":"I32","27":"I64","28":"F64","29":"IQ1_M","30":"BF16"};tensors=[{"name":t.name,"shape":[int(x) for x in t.shape],"type":names.get(str(t.tensor_type),str(t.tensor_type))} for t in r.tensors]
 arch=str(fields.get("general.architecture","")).lower()
 if "qwen" not in arch:raise RuntimeError(f"GGUF is not Qwen architecture: {arch}")
 return {"fields":fields,"tensor_count":len(tensors),"tensor_types":dict(Counter(x["type"] for x in tensors)),"tensors":tensors}
def validate_quantization(audit:dict[str,Any],label:str):
 types={x.upper() for x in audit["tensor_types"]};wanted=label.upper()
 if wanted.startswith("Q4_K") and not any("Q4_K" in x for x in types):raise RuntimeError(f"candidate has no Q4_K tensors for nominal {label}: {sorted(types)}")
 if not any(re.search(r"(^|\.)Q[2-8]",x) or "IQ" in x for x in types):raise RuntimeError("candidate GGUF does not contain quantized tensors")
 return {"status":"PASS","nominal_label":label,"observed_tensor_types":audit["tensor_types"]}
def topology_compatible(ref:dict,cand:dict):
    """Validate scored next-token forward-path topology equivalence.

    Exact whole-GGUF equality is preferred. For Qwen3.5-family artifacts,
    a producer may intentionally omit trailing NextN/MTP decoder blocks.
    llama.cpp excludes those blocks from hparams.n_layer() and therefore
    from the ordinary next-token logits graph.

    Such an omission is accepted only when:
      * architectures match and are qwen35,
      * there are no candidate-extra tensors,
      * there are no shape mismatches among common tensors,
      * every missing tensor belongs to one or more whole trailing blocks,
      * every omitted block is completely absent from the candidate,
      * every omitted block contains explicit blk.<N>.nextn.* tensors
        in the native reference,
      * all non-auxiliary/scored-path tensors match exactly.
    """
    ra={x["name"]:x for x in ref["tensors"]}
    ca={x["name"]:x for x in cand["tensors"]}

    rshape={k:v["shape"] for k,v in ra.items()}
    cshape={k:v["shape"] for k,v in ca.items()}

    missing=sorted(ra.keys()-ca.keys())
    extra=sorted(ca.keys()-ra.keys())
    shape=sorted(
        k for k in ra.keys() & ca.keys()
        if rshape[k] != cshape[k]
    )

    # Ideal case: whole-artifact topology is identical.
    if not missing and not extra and not shape:
        return {
            "status":"PASS",
            "mode":"EXACT_GGUF_TOPOLOGY",
            "reference_tensor_count":len(ra),
            "candidate_tensor_count":len(ca),
            "scored_path_equivalent":True,
            "auxiliary_nextn_omission":False,
            "ignored_auxiliary_tensors":[],
        }

    # Extra tensors or common-name shape differences are never waived.
    if extra or shape:
        raise RuntimeError(
            "GGUF topology mismatch: "
            f"missing={missing[:5]} "
            f"extra={extra[:5]} "
            f"shape={shape[:5]}"
        )

    rarch=str(ref.get("fields",{}).get(
        "general.architecture",""
    )).lower()
    carch=str(cand.get("fields",{}).get(
        "general.architecture",""
    )).lower()

    if rarch != "qwen35" or carch != "qwen35":
        raise RuntimeError(
            "GGUF topology mismatch: "
            f"missing={missing[:5]} extra=[] shape=[]"
        )

    import re

    block_re=re.compile(r"^blk\.(\d+)\.")

    def block_index(name):
        m=block_re.match(name)
        return int(m.group(1)) if m else None

    missing_blocks={block_index(x) for x in missing}

    # Missing global tensors or otherwise unclassified tensors are fatal.
    if None in missing_blocks:
        raise RuntimeError(
            "GGUF scored-path topology mismatch: "
            f"non-block missing tensors={missing[:8]}"
        )

    ref_blocks={
        block_index(x)
        for x in ra
        if block_index(x) is not None
    }
    cand_blocks={
        block_index(x)
        for x in ca
        if block_index(x) is not None
    }

    if not ref_blocks or not cand_blocks:
        raise RuntimeError(
            "GGUF topology block structure unavailable"
        )

    last_candidate=max(cand_blocks)
    last_reference=max(ref_blocks)

    omitted=sorted(missing_blocks)

    # Auxiliary omission must be a contiguous trailing suffix.
    expected=list(range(
        last_candidate + 1,
        last_reference + 1,
    ))

    if omitted != expected or not omitted:
        raise RuntimeError(
            "GGUF scored-path topology mismatch: "
            f"missing blocks={omitted}, "
            f"expected trailing auxiliary blocks={expected}"
        )

    ignored=[]

    for bi in omitted:
        prefix=f"blk.{bi}."

        ref_block=sorted(
            x for x in ra if x.startswith(prefix)
        )
        cand_block=sorted(
            x for x in ca if x.startswith(prefix)
        )

        # The candidate must omit the entire auxiliary block.
        if cand_block:
            raise RuntimeError(
                "partial auxiliary block omission is not allowed: "
                f"blk.{bi}"
            )

        # Explicit NextN tensors prove this is an MTP/NextN block,
        # rather than a normal trunk block accidentally removed.
        nextn=[
            x for x in ref_block
            if x.startswith(f"blk.{bi}.nextn.")
        ]

        if not nextn:
            raise RuntimeError(
                "trailing missing block is not proven NextN/MTP: "
                f"blk.{bi}"
            )

        # Every tensor in the reference block must be part of the
        # missing set. No partial/truncated comparisons.
        if set(ref_block) != {
            x for x in missing if x.startswith(prefix)
        }:
            raise RuntimeError(
                "incomplete auxiliary-block classification: "
                f"blk.{bi}"
            )

        ignored.extend(ref_block)

    # Final guard: absolutely every missing tensor must be accounted for.
    if set(ignored) != set(missing):
        raise RuntimeError(
            "unclassified missing tensors remain after "
            "NextN/MTP exclusion"
        )

    return {
        "status":"PASS",
        "mode":"SCORED_PATH_TOPOLOGY_EQUIVALENCE",
        "reference_tensor_count":len(ra),
        "candidate_tensor_count":len(ca),
        "scored_path_equivalent":True,
        "auxiliary_nextn_omission":True,
        "auxiliary_block_indices":omitted,
        "ignored_auxiliary_tensor_count":len(ignored),
        "ignored_auxiliary_tensors":ignored,
        "rationale":
            "Trailing Qwen35 NextN/MTP blocks are excluded from "
            "the ordinary next-token logits graph under the pinned "
            "llama.cpp runtime.",
    }
def validate_reference_preflight(manifest:dict[str,Any])->dict[str,Any]:
 if manifest.get("schema")!="ramgpt-qwen-reference-preflight-v1" or manifest.get("schema_version")!=1:raise RuntimeError("unsupported Qwen reference-preflight manifest")
 if not re.fullmatch(r"[0-9a-f]{40}",str(manifest.get("canonical_source_revision",""))):raise RuntimeError("reference-preflight manifest lacks immutable source revision")
 if not re.fullmatch(r"[0-9a-f]{64}",str(manifest.get("bf16_gguf_sha256",""))):raise RuntimeError("reference-preflight manifest lacks BF16 GGUF SHA-256")
 audit=manifest.get("gguf_audit") or {};tensors=audit.get("tensors") or []
 if not tensors or audit.get("tensor_count")!=len(tensors):raise RuntimeError("reference-preflight manifest lacks complete tensor topology")
 tok=manifest.get("tokenizer") or {};meta=tok.get("metadata") or {};fields=meta.get("field_sha256") or {}
 if not fields or meta.get("field_count")!=len(fields):raise RuntimeError("reference-preflight manifest lacks tokenizer metadata fingerprints")
 fixtures=tok.get("fixtures") or {};expected=fixtures.get("expected_token_ids") or {}
 if not fixtures.get("fixture_sha256") or not expected.get("add_special=true") or not expected.get("add_special=false"):raise RuntimeError("reference-preflight manifest lacks tokenizer behavioral fixtures")
 n=len(fixtures["fixture_sha256"])
 if len(expected["add_special=true"])!=n or len(expected["add_special=false"])!=n:raise RuntimeError("reference-preflight tokenizer fixture cardinality mismatch")
 return manifest

def build_reference_preflight(audit:dict[str,Any],ref_model:Path,llama:Path,tokenizer_bin:Path,fixtures:list[str],temp:Path,source_revision:str,bf16_sha256:str)->dict[str,Any]:
 metadata=tokenizer_metadata_fingerprints(ref_model,llama);fixture_sha=[hashlib.sha256(x.encode("utf-8")).hexdigest() for x in fixtures]
 expected={"add_special=true":_run_fixture_token_ids(ref_model,tokenizer_bin,fixtures,temp,True,"reference-add"),"add_special=false":_run_fixture_token_ids(ref_model,tokenizer_bin,fixtures,temp,False,"reference-noadd")}
 compact_audit={"fields":audit["fields"],"tensor_count":audit["tensor_count"],"tensor_types":audit["tensor_types"],"tensors":audit["tensors"]}
 manifest={"schema":"ramgpt-qwen-reference-preflight-v1","schema_version":1,"canonical_source_revision":source_revision,"bf16_gguf_sha256":bf16_sha256,"architecture_identity":audit["fields"].get("general.architecture"),"native_tensor_type_inventory":audit["tensor_types"],"gguf_audit":compact_audit,"tokenizer":{"metadata":metadata,"fixtures":{"fixture_sha256":fixture_sha,"expected_token_ids":expected},"contract":{"add_special":True,"parse_special":False,"compatibility_modes":["add_special=true","add_special=false"]}}}
 return validate_reference_preflight(manifest)

def tokenizer_compatible_manifest(
 ref_manifest:dict[str,Any],
 cand_model:Path,
 llama:Path,
 tokenizer_bin:Path,
 fixtures:list[str],
 temp:Path,
 scored_corpus_manifest:dict[str,Any]|None=None,
 scored_source:Path|None=None,
):
 validate_reference_preflight(ref_manifest)
 expected=ref_manifest["tokenizer"]

 fixture_sha=[
  hashlib.sha256(x.encode("utf-8")).hexdigest()
  for x in fixtures
 ]
 if fixture_sha!=expected["fixtures"]["fixture_sha256"]:
  raise RuntimeError(
   "tokenizer compatibility fixture set differs from "
   "frozen reference-preflight manifest"
  )

 got_meta=tokenizer_metadata_fingerprints(cand_model,llama)

 exp_fields=expected["metadata"]["field_sha256"]
 got_fields=got_meta["field_sha256"]

 missing=sorted(exp_fields.keys()-got_fields.keys())
 extra=sorted(got_fields.keys()-exp_fields.keys())
 changed=sorted(
  k for k in exp_fields.keys()&got_fields.keys()
  if exp_fields[k]!=got_fields[k]
 )

 metadata_exact=(
  not missing and
  not extra and
  not changed and
  got_meta["field_count"]==expected["metadata"]["field_count"]
 )

 allowed_non_scored={
  "tokenizer.chat_template":
   "chat template is not invoked by raw-text QuantBench corpus tokenization",
  "tokenizer.ggml.padding_token_id":
   "QuantBench scored corpus tokenization performs no padding",
  "tokenizer.ggml.add_bos_token":
   "effective behavior is explicitly checked under add_special=true and false",
 }

 diff_keys=set(missing)|set(extra)|set(changed)
 disallowed=sorted(diff_keys-set(allowed_non_scored))

 if disallowed:
  raise RuntimeError(
   "tokenizer metadata mismatch affects fields outside the "
   "scored-equivalence allowance: "
   f"{disallowed[:8]}"
  )

 # Frozen behavioral fixture equivalence is mandatory.
 for add,key,tag in (
  (True,"add_special=true","candidate-add"),
  (False,"add_special=false","candidate-noadd"),
 ):
  got=_run_fixture_token_ids(
   cand_model,
   tokenizer_bin,
   fixtures,
   temp,
   add,
   tag,
  )
  if got!=expected["fixtures"]["expected_token_ids"][key]:
   raise RuntimeError(
    f"token-ID equivalence failed {key}"
   )

 scored_match=None
 scored_total=None

 # Non-exact metadata requires stronger proof using the exact
 # 128 source documents and QuantBench's frozen 768-token
 # SHA-derived segment-selection rule.
 if not metadata_exact:
  if scored_corpus_manifest is None or scored_source is None:
   raise RuntimeError(
    "non-exact tokenizer metadata requires scored-corpus "
    "token equivalence evidence"
   )

  ref_segments=scored_corpus_manifest.get("segments",[])
  files=sorted(scored_source.iterdir())

  if len(files)!=128 or len(ref_segments)!=128:
   raise RuntimeError(
    "scored-corpus equivalence requires exactly 128 segments"
   )

  batch=temp/"scored-corpus-batch.tsv"
  outs=[
   temp/f"scored-{i:03d}.tokens"
   for i in range(128)
  ]

  batch.write_text(
   "".join(
    f"{src}\t{out}\n"
    for src,out in zip(files,outs)
   )
  )

  subprocess.run(
   [
    str(tokenizer_bin),
    "--model",str(cand_model),
    "--batch-list",str(batch),
    "--add-special","true",
   ],
   check=True,
   stdout=subprocess.DEVNULL,
  )

  bad=[]

  for sid,(src,tok_path,refseg) in enumerate(
   zip(files,outs,ref_segments)
  ):
   b=tok_path.read_bytes()

   if b[:8]!=b"RQTOKEN\0":
    raise RuntimeError(
     f"invalid candidate tokenizer output segment {sid}"
    )

   n=struct.unpack_from("<Q",b,16)[0]

   if len(b)!=24+4*n or n<768:
    raise RuntimeError(
     f"invalid/short candidate tokenizer output segment {sid}"
    )

   ids=struct.unpack_from(f"<{n}i",b,24)

   source_sha=hashlib.sha256(src.read_bytes()).digest()

   start=(
    int.from_bytes(source_sha[:8],"big")
    % (n-768+1)
   )

   raw=struct.pack(
    "<768i",
    *ids[start:start+768],
   )

   token_sha=hashlib.sha256(raw).hexdigest()

   if (
    src.name!=refseg["source"]
    or start!=refseg["start_token"]
    or token_sha!=refseg["token_sha256"]
   ):
    bad.append({
     "segment_id":sid,
     "source":src.name,
     "reference_source":refseg["source"],
     "candidate_start_token":start,
     "reference_start_token":refseg["start_token"],
     "candidate_token_sha256":token_sha,
     "reference_token_sha256":refseg["token_sha256"],
    })

  if bad:
   raise RuntimeError(
    "scored-corpus token equivalence failed: "
    + json.dumps(bad[0],sort_keys=True)
   )

  scored_match=128
  scored_total=128

 mode=(
  "EXACT_TOKENIZER_METADATA_AND_BEHAVIOR"
  if metadata_exact
  else "SCORED_TOKENIZATION_EQUIVALENCE"
 )

 return {
  "status":"PASS",
  "mode":mode,
  "metadata_exact":metadata_exact,
  "metadata_fields_reference":
   expected["metadata"]["field_count"],
  "metadata_fields_candidate":
   got_meta["field_count"],
  "metadata_differences":{
   "missing":missing,
   "extra":extra,
   "changed":changed,
  },
  "metadata_difference_classification":{
   k:allowed_non_scored[k]
   for k in sorted(diff_keys)
  },
  "fixtures":len(fixtures),
  "modes":[
   "add_special=true",
   "add_special=false",
  ],
  "parse_special":False,
  "scored_corpus_equivalence_required":
   not metadata_exact,
  "scored_corpus_segments_match":
   scored_match,
  "scored_corpus_segments_total":
   scored_total,
  "reference_preflight_sha256":
   ref_manifest.get("manifest_sha256"),
 }

def tokenizer_compatible(ref:dict,cand:dict,ref_model:Path,cand_model:Path,tokenizer_bin:Path,fixtures:list[str],temp:Path):
 # Exact tokenizer GGUF metadata is necessary, and token-ID equivalence in both contract modes is decisive.
 llama_dir=tokenizer_bin.parents[1]/"third_party/llama.cpp"
 sys.path.insert(0,str(llama_dir/"gguf-py"))
 rr=_reader(ref_model,llama_dir);cr=_reader(cand_model,llama_dir)
 keys=sorted(k for k in rr.fields if k.startswith("tokenizer."));bad=[k for k in keys if k not in cr.fields or rr.fields[k].contents()!=cr.fields[k].contents()]
 if bad:raise RuntimeError(f"tokenizer metadata mismatch: {bad[:8]}")
 def run(model,add,tag):
  batch=temp/f"{tag}.tsv";outs=[];lines=[]
  for i,text in enumerate(fixtures):
   inp=temp/f"{tag}-{i}.txt";out=temp/f"{tag}-{i}.tokens";inp.write_text(text);outs.append(out);lines.append(f"{inp}\t{out}\n")
  batch.write_text("".join(lines));subprocess.run([str(tokenizer_bin),"--model",str(model),"--batch-list",str(batch),"--add-special",str(add).lower()],check=True,stdout=subprocess.DEVNULL)
  return [p.read_bytes() for p in outs]
 for add in (True,False):
  if run(ref_model,add,f"r{int(add)}")!=run(cand_model,add,f"c{int(add)}"):raise RuntimeError(f"token-ID equivalence failed add_special={add}")
 return {"status":"PASS","metadata_fields":len(keys),"fixtures":len(fixtures),"modes":["add_special=true","add_special=false"],"parse_special":False}
def build_corpus(model:Path,tokenizer_bin:Path,source:Path,plan_path:Path,out:Path)->dict[str,Any]:
 plan=json.loads(plan_path.read_text());files=sorted(source.iterdir())
 if len(files)!=128 or len(plan["documents"])!=128:raise RuntimeError("frozen corpus requires exactly 128 sources")
 temp=out.parent/"tokenize-temp";temp.mkdir(parents=True,exist_ok=True);batch=temp/"batch.tsv";tokens=[]
 for i,p in enumerate(files):tokens.append(temp/f"{i}.tokens")
 batch.write_text("".join(f"{p}\t{t}\n" for p,t in zip(files,tokens)));subprocess.run([str(tokenizer_bin),"--model",str(model),"--batch-list",str(batch),"--add-special","true"],check=True)
 records=[];payload=bytearray();segments=[]
 for sid,(src,tp) in enumerate(zip(files,tokens)):
  b=tp.read_bytes();n=struct.unpack_from("<Q",b,16)[0]
  if b[:8]!=b"RQTOKEN\0" or len(b)!=24+4*n or n<768:raise RuntimeError(f"invalid/short tokenizer output segment {sid}")
  ids=struct.unpack_from(f"<{n}i",b,24);source_sha=hashlib.sha256(src.read_bytes()).digest();start=int.from_bytes(source_sha[:8],"big")%(n-768+1);raw=struct.pack("<768i",*ids[start:start+768]);payload.extend(raw);domain=plan["documents"][sid]["domain"];db=domain.encode();records.append(struct.pack("<I",sid)+db+b"\0"*(32-len(db))+hashlib.sha256(raw).digest()+raw);segments.append({"segment_id":sid,"source":src.name,"source_sha256":source_sha.hex(),"token_count":n,"start_token":start,"token_sha256":hashlib.sha256(raw).hexdigest()})
 model_sha=hashlib.sha256(model.read_bytes()).digest();payload_sha=hashlib.sha256(payload).digest();model_id=b"Qwen tokenizer-native fidelity-v2\0".ljust(115,b"\0");header=b"RQSEG\0\0\0"+struct.pack("<IIIIIII",1,256,128,768,256,512,1)+model_sha+payload_sha+(COMMIT.encode()+b"\0")[:41].ljust(41,b"\0")+model_id;out.write_bytes(header+b"".join(records))
 return {"status":"PASS","rqseg_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"source_selection":"frozen fidelity-v2 SHA-derived offset rule","segments":segments}
def capture(model:Path,corpus:Path,prefix:Path,probe:Path,quantized:bool,log:Path,gpu_layers:int=-1):
 if not isinstance(gpu_layers,int) or gpu_layers < -1:
  raise ValueError("gpu_layers must be -1 or a non-negative integer")
 prefix.parent.mkdir(parents=True,exist_ok=True);cmd=[str(probe),"--model",str(model),"--segmented-corpus",str(corpus),"--output",str(prefix),"--context-size","4096","--batch-size","512","--gpu-layers",str(gpu_layers),"--rql-version","2","--flash-attention","enabled","--tokenizer-add-special","true","--tokenizer-parse-special","false","--quantized-model",str(quantized).lower()]
 with log.open("w") as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 return cmd
