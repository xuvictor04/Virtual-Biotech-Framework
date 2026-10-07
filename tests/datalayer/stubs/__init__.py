"""Network-free stand-ins for the remote services the correctness tests touch (§19).

* ``eutils_stub`` -- a local NCBI E-utilities server (``VBT_EUTILS_BASE``) that logs every request.
* ``pybioportal/`` -- a ``pybioportal`` module with one study, two patients and three samples,
  put first on the clinicaltrials server's ``sys.path`` by ``clinicaltrials_stubbed_server.py``.
"""

from pathlib import Path

STUBS_DIR = Path(__file__).resolve().parent
CLINICALTRIALS_WRAPPER = STUBS_DIR / "clinicaltrials_stubbed_server.py"
