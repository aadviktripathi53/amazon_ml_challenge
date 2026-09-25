"""Shared command-line handling for pipeline stages (``python -m src.<stage> --split <split>``)."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

SPLITS = ("train", "val", "test")


def parse_split(description: str, argv: Optional[Sequence[str]] = None) -> str:
    """Parse the ``--split`` argument shared by every stage.

    Args:
        description: Stage description shown in ``--help``.
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The chosen split: ``"train"``, ``"val"`` or ``"test"``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--split", required=True, choices=SPLITS)
    return parser.parse_args(argv).split
