"""Every pipeline stage stub imports and accepts --split."""
import importlib

import pytest

STAGES = ["normalize", "block", "features", "train", "decide"]


@pytest.mark.parametrize("stage", STAGES)
def test_stage_stub_runs_and_is_not_implemented(stage):
    """A stub parses --split and then raises NotImplementedError."""
    module = importlib.import_module(f"src.{stage}")
    with pytest.raises(NotImplementedError):
        module.main(["--split", "val"])


def test_stage_rejects_unknown_split():
    """--split only accepts train, val or test."""
    from src.block import main

    with pytest.raises(SystemExit):
        main(["--split", "France"])
