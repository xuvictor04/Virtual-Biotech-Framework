"""Data acquisition helpers: the paper's Zenodo case-study archive and Open Targets release tables."""

from .zenodo import PRESETS, HTTPRangeFile, ZenodoArchive, zenodo_root

__all__ = ["PRESETS", "HTTPRangeFile", "ZenodoArchive", "zenodo_root"]
