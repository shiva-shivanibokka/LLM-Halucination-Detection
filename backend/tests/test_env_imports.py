"""
Guards the pinned dependency set: importing sentence_transformers must not
crash. In a clean torch-only environment (the Docker image and Linux CI, which
install exactly backend/requirements.txt with no tensorflow) this passes.

Known caveat: on a host whose *global* env also has tensorflow installed, the
TF/torch native combination segfaults during this import (returncode 139). That
is local-env pollution, not a defect in the shipped deps — run from a fresh
venv created off requirements.txt, or `pip uninstall tensorflow`.
"""

import subprocess
import sys


def test_sentence_transformers_imports_without_crash():
    # check=False on purpose: the assertion below inspects returncode, so
    # raising here would replace a readable failure with a CalledProcessError.
    r = subprocess.run(
        [sys.executable, "-c", "import sentence_transformers"],
        capture_output=True,
        check=False,
    )
    assert r.returncode == 0, (
        f"import crashed: rc={r.returncode}. If rc=139 (segfault), a stray "
        f"tensorflow in this env collides with torch — use a clean venv. "
        f"stderr tail: {r.stderr[-500:]!r}"
    )


if __name__ == "__main__":
    test_sentence_transformers_imports_without_crash()
    print("PASS")
