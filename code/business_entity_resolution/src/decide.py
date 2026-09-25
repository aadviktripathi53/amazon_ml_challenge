"""Stage 5: threshold/select matches and write output/matching_results.tsv + candidate_pairs.tsv.

STUB (owner: A). Run as ``python -m src.decide --split <train|val|test>``
from ``code/business_entity_resolution/``. Schemas: ``docs/contracts.md``.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .cli import parse_split


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for this stage; not implemented yet.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        NotImplementedError: Always, until the stage is implemented.
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    raise NotImplementedError(f"stage 'decide' is a stub (split={split})")


if __name__ == "__main__":
    main()
