"""``vbt setup``: bring a host up from a clean clone or the harness image (docs/DEPLOYMENT.md).

* :mod:`.probe` -- host facts: CPUs and cgroup quota, RAM and the container's memory limit, GPUs (nvidia-smi),
  disk, memory containment (cgroup or RLIMIT_DATA), the bwrap sandbox, the container runtime, network reachability;
* :mod:`.needs` -- what the enabled roster needs: granted tools, the tables their overlays bind, their sources;
* :mod:`.hostconfig` -- sizing scaled to the host and the files setup writes (``host.yaml``, ``host.env``,
  ``compose.vllm.yaml``);
* :mod:`.steps` -- the generic steps (probe, configure, acquire, size, index, check, calibrate, smoke), each a
  ``vbt`` command run under the host configuration; :mod:`.state` -- resume and measured rates;
* :mod:`.layout` -- the deployment directories (``VBT_HOME``).

Nothing here is specific to a dataset: what to fetch, check and calibrate comes from the descriptors, the
overlays and the enabled roster.
"""

from __future__ import annotations

from typing import Any


def add_setup_parser(sub: Any) -> Any:
    """Register ``vbt setup`` on an argparse subparsers object (handler(args, config))."""
    from .cli import add_setup_parser as _add

    return _add(sub)


__all__ = ["add_setup_parser"]
