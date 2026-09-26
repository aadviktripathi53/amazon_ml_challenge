"""Every pipeline stage module imports, exposes ``main`` and validates ``--split``."""
import importlib

import pytest



def _import(stage):
    """Import a stage module, skipping when LightGBM cannot load (macOS without libomp: `brew install libomp`)."""
    try:
        return importlib.import_module(f"src.{stage}")
    except OSError as exc:  # pragma: no cover - environment specific
        pytest.skip(f"cannot import src.{stage}: {exc}")


STAGES = ["normalize", "block", "features", "train", "predict", "decide", "write_submission"]


@pytest.mark.parametrize("stage", STAGES)
def test_stage_exposes_main(stage):
    """Each stage module is importable and has a callable ``main``."""
    module = _import(stage)
    assert callable(module.main)


@pytest.mark.parametrize("stage", STAGES)
def test_stage_rejects_unknown_split(stage):
    """--split only accepts train, val or test (checked before any work happens)."""
    module = _import(stage)
    with pytest.raises(SystemExit):
        module.main(["--split", "France"])


def test_val_normalize_is_a_noop(capsys):
    """'val' is carved out of the train files, so normalize does nothing for it."""
    from src.normalize import main

    main(["--split", "val"])
    assert "reusing records_train" in capsys.readouterr().out
