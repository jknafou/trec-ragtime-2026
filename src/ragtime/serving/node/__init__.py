"""The SLURM-first node lifecycle and discovery subpackage.

``discovery`` reads the cluster's own node table and is the only module re-exported here,
because it needs nothing beyond subprocess and ruamel. ``vllm_server`` is the other member:
it brings a generation server up and tears it down, and it is reached as a program rather
than as an import, which is how ``slurm/vllm_service.sbatch`` asks it for the exact argv a
run would launch. Keeping it out of this file is what makes ``import ragtime.serving`` cheap
on a machine with no GPU stack installed.
"""

from __future__ import annotations

from .discovery import NodeType, discover_nodes, read_profile, write_profile

__all__ = ["NodeType", "discover_nodes", "read_profile", "write_profile"]
