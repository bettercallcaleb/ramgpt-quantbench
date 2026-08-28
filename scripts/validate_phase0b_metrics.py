#!/usr/bin/env python3
"""Independent NumPy validation of 100 selected Phase 0B rows."""
import argparse, json, math, struct
from pathlib import Path
import numpy as np

RQL_V1_HEADER_SIZE = 233
RQL_V2_HEADER_SIZE = 321

def parse_rql_header(f):
    prefix = f.read(16)
    if len(prefix) != 16 or prefix[:8] != b"RQLOGIT\0":
        raise ValueError("invalid RQL header")
    version, hsize = struct.unpack_from("<II", prefix, 8)
    expected = {1: RQL_V1_HEADER_SIZE, 2: RQL_V2_HEADER_SIZE}.get(version)
    if expected is None or hsize != expected:
        raise ValueError(f"unsupported RQL schema/header: {version}/{hsize}")
    raw = prefix + f.read(hsize - 16)
    if len(raw) != hsize:
        raise ValueError("truncated RQL header")
    vocab, dtype = struct.unpack_from("<II", raw, 16)
    positions = struct.unpack_from("<Q", raw, 24)[0]
    if not vocab or dtype != 1: raise ValueError("invalid RQL vocabulary/dtype")
    out = {"schema_version": version, "header_size": hsize, "vocab_size": vocab,
           "logits_dtype": dtype, "number_of_positions": positions}
    if version == 1:
        out.update(token_sha256=raw[32:64], model_id=raw[64:192].split(b"\0",1)[0].decode(),
                   llama_cpp_commit=raw[192:233].split(b"\0",1)[0].decode())
    else:
        values = struct.unpack_from("<IIIIIiIIIIIIII", raw, 265)
        if any(values[10:]): raise ValueError("nonzero RQL v2 reserved field")
        if values[3:5] != (1, 1) or values[6] not in (0, 1) or values[7] not in (0, 1) or values[8] not in (0, 1) or values[9] != 0:
            raise ValueError("invalid RQL v2 execution enum/boolean")
        if raw[32:64] == bytes(32) or raw[64:96] == bytes(32):
            raise ValueError("missing RQL v2 SHA256")
        out.update(model_sha256=raw[32:64], token_sha256=raw[64:96],
                   llama_cpp_commit=raw[96:137].split(b"\0",1)[0].decode(),
                   model_id=raw[137:265].split(b"\0",1)[0].decode(), n_ctx=values[0],
                   n_batch=values[1], n_ubatch=values[2], kv_k_type=values[3],
                   kv_v_type=values[4], gpu_layers=values[5], flash_attention_resolved=values[6],
                   tokenizer_add_special=bool(values[7]), tokenizer_parse_special=bool(values[8]),
                   sampling=values[9])
        if not out["n_ctx"] or not out["n_batch"] or not out["n_ubatch"]:
            raise ValueError("missing RQL v2 execution geometry")
    return out

def read_rql_header(f):
    h = parse_rql_header(f)
    return h["vocab_size"], h["number_of_positions"]

def logsumexp(x):
    m = np.max(x)
    return float(m + np.log(np.exp(x - m, dtype=np.float64).sum(dtype=np.float64)))

def top2(x):
    # Stable full ordering matches the C++ strict-'greater-than' tie behavior.
    return np.argsort(-x, kind="stable")[:2]

def calc(r, q, target):
    rt, qt = top2(r), top2(q)
    lr, lq = logsumexp(r), logsumexp(q)
    logp = r.astype(np.float64) - lr
    logq = q.astype(np.float64) - lq
    kl = float(np.sum(np.exp(logp) * (logp - logq), dtype=np.float64))
    pr, pq = math.exp(float(r[target]) - lr), math.exp(float(q[target]) - lq)
    return {"ref_top1_id": int(rt[0]), "ref_top2_id": int(rt[1]),
            "quant_top1_id": int(qt[0]), "quant_top2_id": int(qt[1]),
            "ref_margin": float(r[rt[0]] - r[rt[1]]),
            "decision_flip": bool(qt[0] != rt[0]), "kl_ref_quant": max(0.0, kl),
            "p_ref_target": pr, "p_quant_target": pq,
            "delta_nll": (lq-float(q[target]))-(lr-float(r[target]))}

def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-10)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--quant-dump", required=True)
    p.add_argument("--cpp-positions", required=True)
    p.add_argument("--validation-positions")
    p.add_argument("--output")
    a = p.parse_args()
    cpp = {int(x["position"]): x for x in map(json.loads, Path(a.cpp_positions).read_text().splitlines())}
    with open(a.reference, "rb") as rf, open(a.quant_dump, "rb") as qf:
        header = parse_rql_header(rf)
        vocab, positions = header["vocab_size"], header["number_of_positions"]
        hdr = qf.read(24)
        if len(hdr) != 24 or hdr[:8] != b"RQSELQ4\0": raise ValueError("invalid selected-Q4 dump")
        version, qvocab, count, reserved = struct.unpack_from("<IIII", hdr, 8)
        if (version, qvocab, reserved) != (1, vocab, 0): raise ValueError("unexpected dump schema")
        if a.validation_positions:
            values = [int(x.strip()) for x in Path(a.validation_positions).read_text().splitlines() if x.strip()]
            if len(values) != len(set(values)): raise ValueError("duplicate validation position")
            expected = set(values)
        else:
            expected = {k*(positions-1)//99 for k in range(100)}
        if any(pos < 0 or pos >= positions for pos in expected): raise ValueError("validation position outside scored-position range")
        if count != len(expected): raise ValueError(f"dump count {count} does not match selected count {len(expected)}")
        seen = set()
        validation_rows = []
        for _ in range(count):
            raw = qf.read(8); pos = struct.unpack("<Q", raw)[0] if len(raw)==8 else -1
            q = np.fromfile(qf, dtype="<f4", count=vocab)
            if q.size != vocab: raise ValueError("truncated Q4 row")
            rf.seek(header["header_size"] + pos*vocab*4)
            r = np.fromfile(rf, dtype="<f4", count=vocab)
            if r.size != vocab: raise ValueError("truncated reference row")
            got, recorded = calc(r, q, int(cpp[pos]["target_token_id"])), cpp[pos]
            for key in ("ref_top1_id","ref_top2_id","quant_top1_id","quant_top2_id","decision_flip"):
                if got[key] != recorded[key]: raise AssertionError(f"position {pos} {key}: {got[key]} != {recorded[key]}")
            for key in ("ref_margin","kl_ref_quant","p_ref_target","p_quant_target","delta_nll"):
                if not close(got[key], recorded[key]): raise AssertionError(f"position {pos} {key}: {got[key]} != {recorded[key]}")
            validation_rows.append({"position": pos, "target_token_id": int(recorded["target_token_id"]), **got,
                                    "numpy_validation_pass": True})
            seen.add(pos)
        if qf.read(1): raise ValueError("trailing selected-Q4 data")
        if seen != expected: raise AssertionError("selected positions are not the deterministic 100-position set")
    validation_rows.sort(key=lambda x: x["position"])
    if a.output:
        Path(a.output).write_text(json.dumps({"schema_version": 1, "status": "PASS",
                                             "rtol": 1e-8, "atol": 1e-10,
                                             "positions": validation_rows}, indent=2) + "\n")
    print(f"validated_positions={len(expected)}")
    print("status=PASS")
    print("tolerances=rtol:1e-8,atol:1e-10")

if __name__ == "__main__": main()
