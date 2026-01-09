#!/usr/bin/env bash
# Build the wheel, install it into a fresh virtual environment, and run a default fine-tune from a directory
# outside the checkout, so nothing can be resolved from the source tree.
#   bash scripts/check_wheel_install.sh [out_dir]        (default: results/audits/wheel_install)
#   CFD2VEC_SCRATCH=<dir> bash scripts/check_wheel_install.sh   use <dir> instead of the system temp directory
# The environment reuses the current interpreter's site-packages for heavy dependencies (numpy, torch); cfd2vec
# itself must come from the wheel, which the check asserts. Writes <out_dir>/wheel_check.json and wheel_check.log.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
OUT="$(mkdir -p "${1:-results/audits/wheel_install}" && cd "${1:-results/audits/wheel_install}" && pwd)"
if [ -n "${CFD2VEC_SCRATCH:-}" ]; then          # scratch directory outside the checkout, kept afterwards
  WORK="$CFD2VEC_SCRATCH/wheel_check_$$"; mkdir -p "$WORK"
else
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
fi
LOG="$OUT/wheel_check.log"
{
  echo "repo: $REPO"; echo "work: $WORK"; date
  python -m pip wheel --no-build-isolation --no-deps -w "$WORK/dist" . -q
  WHEEL="$(ls "$WORK"/dist/cfd2vec-*.whl)"
  echo "wheel: $(basename "$WHEEL") sha256 $(shasum -a 256 "$WHEEL" | cut -d' ' -f1)"
  python -m zipfile -l "$WHEEL" | grep protocols/
  python -m venv --system-site-packages "$WORK/venv"
  "$WORK/venv/bin/pip" install --no-deps --no-index -q "$WHEEL"
  cd "$WORK"
  "$WORK/venv/bin/python" - "$OUT/wheel_check.json" "$REPO" <<'EOF'
import hashlib, json, os, sys
import numpy as np
import cfd2vec
from cfd2vec.api import CFD2vec, DEFAULT_PROTOCOL
from cfd2vec.model.network import CFD2vecNet, ModelConfig
from cfd2vec.schema import Case, Conditioning
out, repo = sys.argv[1], os.path.realpath(sys.argv[2])
pkg = os.path.realpath(os.path.dirname(cfd2vec.__file__))
assert not pkg.startswith(repo), f"cfd2vec imported from the checkout: {pkg}"
assert os.path.realpath(DEFAULT_PROTOCOL).startswith(pkg), DEFAULT_PROTOCOL
rng = np.random.default_rng(0); paths = []
for i in range(2):
    pts = rng.uniform([-3, -3, 0], [3, 3, 2], (1500, 3)).astype(np.float32)
    f = np.zeros((1500, 6), np.float32); f[:, 0] = np.tanh(pts[:, 2]); f[:, 4:] = 0.01
    c = Case(f"t{i}", "synthetic", pts, pts[:, 2].copy(), np.tile([0, 0, 1.0], (1500, 1)).astype(np.float32), f,
             np.ones(4, bool), Conditioning(closure="k-epsilon"), 1.0, 1.0)
    c.save(f"c{i}.npz"); paths.append(f"c{i}.npz")
cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, n_tokens_scale1=32, n_tokens_scale2=16, k_neighbors=8,
                  pointnet_hidden=[16, 32], n_context=1024, n_query=128)
m = CFD2vec(CFD2vecNet(cfg), "cpu"); m.stats = dict(mean=[0.0] * 6, std=[1.0] * 6)
hist = m.finetune(paths, max_epochs=1, log_path="ft.json")          # default protocol, no path given
ft = json.load(open("ft.json"))
rec = dict(passed=True, package_dir=pkg, default_protocol=DEFAULT_PROTOCOL, protocol_version=ft["protocol_version"],
           protocol_sha256=ft["protocol_sha256"],
           repo_protocol_sha256=hashlib.sha256(open(os.path.join(repo, "configs", "finetune_protocol.yaml"), "rb").read()).hexdigest(),
           finetune_epochs=len(hist), cfd2vec_version=cfd2vec.__version__)
rec["protocol_matches_repo_copy"] = rec["protocol_sha256"] == rec["repo_protocol_sha256"]
rec["passed"] = rec["protocol_matches_repo_copy"]
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(rec, indent=1))
sys.exit(0 if rec["passed"] else 1)
EOF
  date
} 2>&1 | sed "s#$WORK#<tmp>#g" | tee "$LOG"
