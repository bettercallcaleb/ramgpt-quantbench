import csv, hashlib, importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; SCRIPT=ROOT/"artifact-bench/run-artifact-comparison.py"
spec=importlib.util.spec_from_file_location("runner",SCRIPT); runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def row(name,mean):
 return {"artifact":name,"artifact_url":f"https://example.com/{name}.gguf","artifact_sha256":name*8,
 "artifact_bytes":123,"mean_kl_bf16_to_artifact":mean,"top1_flips":10,"hcdfr":{str(t):{"hcdf_rate":.01,"eligible_positions":100,"flips":1} for t in (1,2,4)},
 "max_flipped_bf16_margin":2.0,"segment_mean_kl":[mean+i*1e-6 for i in range(128)],
 "provenance_grade":"Grade D","strict_recipe_eligible":False,"compact_sha256":"0"*64}

class TestCLI(unittest.TestCase):
 def call(self,*args):return subprocess.run([sys.executable,str(SCRIPT),*args],text=True,capture_output=True)
 def test_help(self):self.assertEqual(self.call("--help").returncode,0)
 def test_too_few(self):self.assertNotEqual(self.call("--reference","https://huggingface.co/a/b","--artifact","a=https://e.test/a.gguf","--quantization-label","Q4_K_M","--dry-run").returncode,0)
 def test_duplicate(self):self.assertNotEqual(self.call("--reference","https://huggingface.co/a/b","--artifact","A=https://e.test/a","--artifact","a=https://e.test/b","--quantization-label","Q4_K_M","--dry-run").returncode,0)
 def test_id(self):self.assertEqual(runner.derive_id("https://huggingface.co/Qwen/Qwen3-4B","Q4_K_M"),"qwen-qwen3-4b-q4-k-m-open-artifact-rank-v1")
 def test_dry_run(self):
  a=runner.parse_args(["--reference","https://huggingface.co/Qwen/Qwen3-4B","--artifact","a=https://huggingface.co/a/q/resolve/main/a.gguf","--artifact","b=https://huggingface.co/b/q/resolve/main/b.gguf","--quantization-label","Q4_K_M","--dry-run"]);old=(runner.qwen.resolve_revision,runner.qwen.reference_metadata,runner.qwen.resolve_artifact,runner.download,runner.delete_safe,runner.qwen.capture)
  runner.qwen.resolve_revision=lambda u:{"repository_id":"Qwen/Qwen3-4B","commit":"a"*40}
  runner.qwen.reference_metadata=lambda x:{"weight_files":["model.safetensors"],"weight_bytes_total":10<<30,"config_files":["config.json"],"tokenizer_files":["tokenizer.json"],"vocabulary_size":1000}
  runner.qwen.resolve_artifact=lambda u:{"input_url":u,"resolved_url":u.replace("main","b"*40),"resolved_revision":"b"*40,"resolved_revision_immutable":True,"remote_bytes":2<<30,"remote_etag":"e","remote_lfs_sha256":"f"*64}
  runner.download=lambda *x:(_ for _ in ()).throw(AssertionError("body download"));runner.delete_safe=lambda *x:(_ for _ in ()).throw(AssertionError("delete"));runner.qwen.capture=lambda *x:(_ for _ in ()).throw(AssertionError("inference"))
  try:self.assertEqual(runner.dry_run(a,ROOT/"artifact-bench/runs/test"),0)
  finally:runner.qwen.resolve_revision,runner.qwen.reference_metadata,runner.qwen.resolve_artifact,runner.download,runner.delete_safe,runner.qwen.capture=old

class TestOutputs(unittest.TestCase):
 def test_csv_two_and_five(self):
  for n in (2,5):
   rows=[row(chr(97+i),.01+i*.001) for i in range(n)];u=runner.bootstrap(rows)
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"result.csv";runner.write_csv(p,"bench","m","Q4",rows,u,"f"*64)
    with p.open() as f:got=list(csv.DictReader(f))
    self.assertEqual(len(got),n);self.assertEqual(got[0]["paired_ci_low_vs_winner"],"0")
 def test_bootstrap_shapes(self):
  u=runner.bootstrap([row("a",.01),row("b",.02),row("c",.03)])
  self.assertEqual(len(u["winner_vs_all"]),2);self.assertEqual(u["winner_vs_runner_up"]["other"],"b")
 def test_registry_and_frozen_hash(self):
  with tempfile.TemporaryDirectory() as d:
   public=Path(d)
   (public/"leaderboard-index-v1.json").write_text("{}")
   (public/"leaderboard-index-v2.json").write_text("{}")
   v,p=runner.next_registry(public)
   self.assertEqual(v,3)
   self.assertEqual(p.name,"leaderboard-index-v2.json")
  v1=runner.PUBLIC/"leaderboard-index-v1.json"
  v2=runner.PUBLIC/"leaderboard-index-v2.json"
  self.assertEqual(runner.sha256(v1),runner.V1_SHA)
  self.assertEqual(
   runner.sha256(v2),
   "8792d49d4300a7ed77a34e32f6a3bdfc5c20c301dbb0e334ffd10248c1291980"
  )
 def test_prompt_and_path(self):
  x=runner.publish_prompt("abc",2,{"a":"b"});self.assertIn("leaderboard-index-v2.json",x);self.assertIn("immutable registry snapshot",x);self.assertIn("hash or schema failure",x)
 def test_state_resume_and_cleanup(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s=runner.State(root/"run-state.json");p=root/"large";p.write_bytes(b"123");s.complete("VALIDATED",[p]);runner.delete_safe(p,s,"VALIDATED");self.assertFalse(p.exists())
   s2=runner.State(root/"run-state.json",True);self.assertTrue(s2.done("VALIDATED"))
 def test_checkpoint_detects_change(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s=runner.State(root/"run-state.json");p=root/"x";p.write_text("a");s.complete("X",[p]);p.write_text("b")
   with self.assertRaises(RuntimeError):runner.State(root/"run-state.json",True)
 def test_frozen_qwen_regression(self):
  x=runner.frozen_regression();self.assertEqual(x["status"],"PASS");self.assertEqual(x["ggml_flips"],6437);self.assertEqual(x["unsloth_flips"],5699)
 def test_large_file_audit(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"sparse"
   with p.open("wb") as f:f.truncate((1<<30)+1)
   with self.assertRaises(RuntimeError):runner.audit_large_files(Path(d))
 def test_partial_download_not_checkpointed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s=runner.State(root/"run-state.json");(root/"x.partial").write_text("partial")
   self.assertFalse(s.done("DOWNLOAD"))
 def test_rql_size(self):self.assertEqual(runner.rql_expected_bytes(151936),321+65536*151936*4)
 def test_disk_peak_largest_and_pass_fail(self):
  cs=[{"label":"small","remote_bytes":2<<30},{"label":"large","remote_bytes":4<<30}]
  good=runner.disk_model(20<<30,151936,cs,500<<30,10<<30);self.assertEqual(good["candidate_peak_artifact"],"large");self.assertTrue(good["pass"]);self.assertGreater(good["determinism_stage_peak"],good["bf16_rql_a"]*2);self.assertFalse(good["retain_bf16_during_candidates"])
  retained=runner.disk_model(20<<30,151936,cs,500<<30,10<<30,retain_bf16_during_candidates=True);self.assertEqual(retained["candidate_stage_peak"]-good["candidate_stage_peak"],good["converted_bf16_gguf"])
  bad=runner.disk_model(20<<30,151936,cs,20<<30,10<<30);self.assertFalse(bad["pass"])
 def test_lifecycle_releases_bf16_before_candidates(self):
  cs=[{"label":"a","remote_bytes":2<<30},{"label":"b","remote_bytes":3<<30}];d=runner.disk_model(20<<30,151936,cs,500<<30,10<<30);text=runner.lifecycle_table(cs,d)
  self.assertIn("deletes after: BF16 RQL B + BF16 GGUF",text);self.assertIn("canonical BF16 RQL + reference-preflight manifest",text)

class TestQwenAdapter(unittest.TestCase):
 def test_hf_url(self):
  self.assertEqual(runner.qwen.parse_hf_url("https://huggingface.co/Qwen/Qwen3-4B"),("Qwen/Qwen3-4B",None))
  self.assertEqual(runner.qwen.parse_hf_url("https://huggingface.co/Qwen/Qwen3-4B/tree/abc")[1],"abc")
 def test_unsupported(self):
  with self.assertRaises(ValueError):runner.qwen.parse_hf_url("https://huggingface.co/meta-llama/Llama")
 def test_revision_resolution_logic(self):
  old=runner.qwen.api_json
  runner.qwen.api_json=lambda u:{"sha":"a"*40,"siblings":[{"rfilename":"config.json"},{"rfilename":"tokenizer.json"},{"rfilename":"model.safetensors","size":42},{"rfilename":"README.md"}]}
  try:
   x=runner.qwen.resolve_revision("https://huggingface.co/Qwen/Test");self.assertEqual(x["commit"],"a"*40);self.assertNotIn("README.md",x["files"]);self.assertEqual(x["file_sizes"]["model.safetensors"],42)
  finally:runner.qwen.api_json=old
 def test_artifact_main_and_immutable_resolution(self):
  old=runner.qwen.api_json;seen=[]
  runner.qwen.api_json=lambda u:(seen.append(u) or {"sha":("e"*40 if "e"*40 in u else "c"*40),"siblings":[{"rfilename":"x.gguf","size":123,"lfs":{"size":123,"oid":"d"*64}}]})
  try:
   x=runner.qwen.resolve_artifact("https://huggingface.co/o/r/resolve/main/x.gguf");self.assertEqual(x["resolved_revision"],"c"*40);self.assertIn("/resolve/"+"c"*40+"/",x["resolved_url"]);self.assertEqual(x["remote_bytes"],123);self.assertEqual(x["remote_lfs_sha256"],"d"*64)
   y=runner.qwen.resolve_artifact("https://huggingface.co/o/r/resolve/"+"e"*40+"/x.gguf");self.assertEqual(y["resolved_revision"],"e"*40);self.assertIn("e"*40,seen[-1])
  finally:runner.qwen.api_json=old
 def test_vocabulary_extraction(self):
  old=runner.qwen.small_json;runner.qwen.small_json=lambda u:{"vocab_size":151936} if u.endswith("config.json") and not u.endswith("tokenizer_config.json") else {}
  try:
   x=runner.qwen.reference_metadata({"repository_id":"Qwen/X","commit":"a"*40,"files":["config.json","model.safetensors"],"file_sizes":{"model.safetensors":99}});self.assertEqual(x["vocabulary_size"],151936);self.assertEqual(x["weight_bytes_total"],99)
  finally:runner.qwen.small_json=old
 def test_topology_mismatch(self):
  with self.assertRaises(RuntimeError):runner.qwen.topology_compatible({"tensors":[{"name":"a","shape":[1]}]},{"tensors":[{"name":"a","shape":[2]}]})
 def test_quantization_mismatch(self):
  with self.assertRaises(RuntimeError):runner.qwen.validate_quantization({"tensor_types":{"Q8_0":2,"F32":1}},"Q4_K_M")
 def test_tokenizer_failure(self):
  # Metadata mismatch is fail-closed before tokenization.
  class F:
   def __init__(self,x):self.x=x
   def contents(self):return self.x
  class R:
   def __init__(self,x):self.fields={"tokenizer.ggml.pre":F(x)}
  old=runner.qwen._reader;runner.qwen._reader=lambda p,l:R(str(p))
  try:
   with tempfile.TemporaryDirectory() as d:
    with self.assertRaises(RuntimeError):runner.qwen.tokenizer_compatible({}, {},Path("ref"),Path("cand"),ROOT/"build/ramgpt-tokenize",["x"],Path(d))
  finally:runner.qwen._reader=old
 def test_run_scoped_hf_cache_contract(self):
  source=SCRIPT.read_text();self.assertIn('"HF_HOME":str(envhome)',source);self.assertIn('work/"hf-cache"',source)
 def test_reference_determinism_transition(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s=runner.State(root/"run-state.json");e=root/"det.json";e.write_text('{"status":"PASS"}')
   s.complete("REFERENCE_VERIFIED",[e]);self.assertTrue(runner.State(root/"run-state.json",True).done("REFERENCE_VERIFIED"))
 def test_resume_after_candidate_cleanup(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s=runner.State(root/"run-state.json");raw=root/"candidate.rql";raw.write_bytes(b"raw");compact=root/"candidate.npz";compact.write_bytes(b"compact");s.complete("ARTIFACT_001_VALIDATED",[raw,compact]);runner.delete_safe(raw,s,"ARTIFACT_001_VALIDATED")
   resumed=runner.State(root/"run-state.json",True);self.assertTrue(resumed.done("ARTIFACT_001_VALIDATED"));self.assertTrue(compact.exists())

 def test_candidate_tokenizer_preflight_without_reference_bf16(self):
  fixtures=["hello"];manifest={"schema":"ramgpt-qwen-reference-preflight-v1","schema_version":1,"canonical_source_revision":"a"*40,"bf16_gguf_sha256":"b"*64,"gguf_audit":{"tensor_count":1,"tensors":[{"name":"x","shape":[1],"type":"BF16"}],"tensor_types":{"BF16":1},"fields":{"general.architecture":"qwen"}},"tokenizer":{"metadata":{"field_count":1,"field_sha256":{"tokenizer.ggml.model":"c"*64}},"fixtures":{"fixture_sha256":[hashlib.sha256(b"hello").hexdigest()],"expected_token_ids":{"add_special=true":[[1,2]],"add_special=false":[[2]]}}}}
  old_meta=runner.qwen.tokenizer_metadata_fingerprints;old_run=runner.qwen._run_fixture_token_ids
  runner.qwen.tokenizer_metadata_fingerprints=lambda model,llama:manifest["tokenizer"]["metadata"];runner.qwen._run_fixture_token_ids=lambda model,tokenizer_bin,fx,temp,add,tag:[[1,2]] if add else [[2]]
  try:
   with tempfile.TemporaryDirectory() as d:self.assertEqual(runner.qwen.tokenizer_compatible_manifest(manifest,Path("candidate.gguf"),Path("llama"),Path("tokenizer"),fixtures,Path(d))["status"],"PASS")
  finally:runner.qwen.tokenizer_metadata_fingerprints=old_meta;runner.qwen._run_fixture_token_ids=old_run
 def test_bf16_cleanup_requires_reference_ready_checkpoint(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s=runner.State(root/"run-state.json");bf16=root/"reference-bf16.gguf";bf16.write_bytes(b"bf16")
   with self.assertRaises(RuntimeError):runner.delete_safe(bf16,s,"REFERENCE_READY_FOR_ARTIFACTS")
   self.assertTrue(bf16.exists())
 def test_reference_preflight_manifest_validation(self):
  manifest={"schema":"ramgpt-qwen-reference-preflight-v1","schema_version":1,"canonical_source_revision":"a"*40,"bf16_gguf_sha256":"b"*64,"gguf_audit":{"tensor_count":1,"tensors":[{"name":"x","shape":[1],"type":"BF16"}],"tensor_types":{"BF16":1},"fields":{"general.architecture":"qwen"}},"tokenizer":{"metadata":{"field_count":1,"field_sha256":{"tokenizer.ggml.model":"c"*64}},"fixtures":{"fixture_sha256":["d"*64],"expected_token_ids":{"add_special=true":[[1,2]],"add_special=false":[[2]]}}}}
  self.assertIs(runner.qwen.validate_reference_preflight(manifest),manifest)
  self.assertEqual(runner.qwen.topology_compatible(manifest["gguf_audit"],{"tensors":[{"name":"x","shape":[1]}]})["status"],"PASS")
 def test_resume_after_reference_bf16_cleanup(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s=runner.State(root/"run-state.json");bf16=root/"reference-bf16.gguf";bf16.write_bytes(b"bf16");rql=root/"reference-a.rql";rql.write_bytes(b"rql");manifest=root/"reference-preflight.json";manifest.write_text("{}")
   s.complete("REFERENCE_CONVERTED",[bf16]);s.complete("REFERENCE_READY_FOR_ARTIFACTS",[rql,manifest]);runner.delete_safe(bf16,s,"REFERENCE_READY_FOR_ARTIFACTS")
   resumed=runner.State(root/"run-state.json",True);self.assertTrue(resumed.done("REFERENCE_CONVERTED"));self.assertTrue(resumed.done("REFERENCE_READY_FOR_ARTIFACTS"));self.assertFalse(bf16.exists());self.assertTrue(rql.exists());self.assertTrue(manifest.exists())

if __name__=="__main__":unittest.main()
