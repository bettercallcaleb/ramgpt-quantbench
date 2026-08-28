#!/usr/bin/env python3
import importlib.util, struct, subprocess, sys, tempfile
from pathlib import Path

exe, root = sys.argv[1], Path(sys.argv[2])
subprocess.run([exe], check=True, stdout=subprocess.DEVNULL)
spec = importlib.util.spec_from_file_location("validator", root / "scripts/validate_phase0b_metrics.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
base = Path(tempfile.gettempdir()) / "ramgpt-format-tests"
with (base / "v1.rql").open("rb") as f:
    h1 = module.parse_rql_header(f); assert f.tell() == 233
with (base / "v2.rql").open("rb") as f:
    raw = f.read(321); f.seek(0); h2 = module.parse_rql_header(f); assert f.tell() == 321
assert h1["schema_version"] == 1 and h1["header_size"] == module.RQL_V1_HEADER_SIZE == 233
assert h2["schema_version"] == 2 and h2["header_size"] == module.RQL_V2_HEADER_SIZE == 321
assert (h2["vocab_size"], h2["number_of_positions"]) == (4, 2)
assert raw[32:64] == bytes([8])*32 and raw[64:96] == bytes([7])*32
assert struct.unpack_from("<III", raw, 265) == (4096, 512, 512)
assert h2["gpu_layers"] == -1 and h2["flash_attention_resolved"] == 1
bad = bytearray(raw); struct.pack_into("<I", bad, 12, 229)
try:
    module.parse_rql_header(__import__("io").BytesIO(bad))
    raise AssertionError("malformed v2 header accepted")
except ValueError:
    pass
print("rql_v1_header_size=233 rql_v2_header_size=321 cpp_python_compat=PASS")
