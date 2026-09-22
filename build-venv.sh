#!/bin/bash
# One-shot venv build for arbiter on fkzllama.
# Mirrors run.sh cmd_setup's pip set verbatim, MINUS the huggingface download:
# the three checkpoints were shipped from gx10 and sha256-verified against it.
set -euo pipefail
trap 'echo "BUILD_FAIL rc=$? line=$LINENO cmd=[$BASH_COMMAND]"' ERR
cd /opt/arbiter
chmod +x run.sh
if [ ! -f integrations/claude-code/hooks/guard_policy.py ]; then
  echo "BUILD_FAIL guard_policy.py missing from fork — approval gate would be dead" >&2
  exit 1
fi
echo "GUARD_POLICY PRESENT"
rm -rf .venv
/usr/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel >/dev/null
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install "transformers>=5" safetensors huggingface_hub numpy fastapi "uvicorn[standard]" "laya==0.3.4" "mcp>=2" pytest httpx
.venv/bin/python - <<'PYCHECK'
import fastapi, transformers, torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("  dev", i, torch.cuda.get_device_name(i), "capability", torch.cuda.get_device_capability(i))
import server.app
print("server.app import OK (fastapi app constructs clean)")
PYCHECK
echo "BUILD_DONE"
