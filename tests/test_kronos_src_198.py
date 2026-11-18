#!/usr/bin/env python3
"""Tests for issue #198 (borsagator): Kronos on the worker path.

The 18 Nov worker sweep failed Kronos with `No module named 'torch'`
-- torch was never declared in requirements.txt -- and the torch
error masked a second defect: the worker path never provisioned
kronos_src/ (upstream Kronos model package), so `from model import
Kronos` would have failed next. The runner's own CI does both steps
in .github/workflows/run.yml; the worker path did neither.

Run: python3 tests/test_kronos_src_198.py   (stdlib + pandas only,
no network -- the git/pip calls are injected fakes)
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "kronos_forecast_198", REPO / "scripts" / "kronos_forecast.py")
kf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kf)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}: {name}"
          + (f" -- {detail}" if detail and not ok else ""))


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRun:
    """Records commands; a git clone materialises kronos_src/model."""

    def __init__(self, clone_rc=0):
        self.calls = []
        self.clone_rc = clone_rc

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if cmd[:2] == ["git", "clone"]:
            if self.clone_rc != 0:
                return FakeResult(self.clone_rc, "", "clone failed")
            dest = Path(cmd[-1])
            (dest / "model").mkdir(parents=True, exist_ok=True)
            (dest / "requirements.txt").write_text("torch>=2.0.0\n")
            return FakeResult(0)
        return FakeResult(0)


# 1. requirements.txt declares CPU torch (finbert-runner precedent).
req = (REPO / "requirements.txt").read_text()
lines = [l.strip() for l in req.splitlines()
         if l.strip() and not l.strip().startswith("#")]
check("requirements declare torch",
      any(l == "torch" or l.startswith(("torch ", "torch==", "torch>="))
          for l in lines),
      "requirements.txt=%r" % req)
check("requirements use the PyTorch CPU wheel index",
      "download.pytorch.org/whl/cpu" in req, req)

# 2. ensure_kronos_src clones + installs upstream requirements once.
tmp = Path(tempfile.mkdtemp(prefix="k198-"))
os.chdir(tmp)
fake = FakeRun()
check("ensure clones when kronos_src/model is missing",
      kf.ensure_kronos_src("kronos_src", run=fake) is True)
check("ensure ran git clone of the upstream Kronos repo",
      any(c[:3] == ["git", "clone", "--depth"]
          and "shiyu-coder/Kronos" in c[4] for c in fake.calls),
      str(fake.calls))
check("ensure installed the upstream requirements",
      any(c[:4] == [sys.executable, "-m", "pip", "install"]
          for c in fake.calls),
      str(fake.calls))
n_calls = len(fake.calls)
check("ensure is a no-op when kronos_src/model already exists",
      kf.ensure_kronos_src("kronos_src", run=fake) is False
      and len(fake.calls) == n_calls)

# 3. A failed clone fails loudly instead of running half-present.
tmp2 = Path(tempfile.mkdtemp(prefix="k198-fail-"))
os.chdir(tmp2)
try:
    kf.ensure_kronos_src("kronos_src", run=FakeRun(clone_rc=128))
    check("ensure raises RuntimeError when the clone fails", False)
except RuntimeError as e:
    check("ensure raises RuntimeError when the clone fails",
          "cloning upstream Kronos failed" in str(e), str(e))

# 4. A pre-existing kronos_src/model is left alone (no calls).
tmp3 = Path(tempfile.mkdtemp(prefix="k198-exists-"))
os.chdir(tmp3)
(tmp3 / "kronos_src" / "model").mkdir(parents=True)
fake3 = FakeRun()
check("ensure leaves a pre-existing kronos_src/model alone",
      kf.ensure_kronos_src("kronos_src", run=fake3) is False
      and fake3.calls == [])

failed = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
