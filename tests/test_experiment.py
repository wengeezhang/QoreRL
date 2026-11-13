import json
import sys

import torch

from qorerl.experiment import main


def test_twenty_step_schedule_probes_do_not_change_training(tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qorerl",
            "--steps",
            "40",
            "--warmup-steps",
            "1",
            "--prompts",
            "1",
            "--group-size",
            "2",
            "--arms",
            "mxfp8,adaptive",
            "--cosine-threshold=-1",
            "--out",
            str(output),
        ],
    )
    main()
    adaptive = output / "seed_0" / "adaptive"
    events = [json.loads(line) for line in (adaptive / "calibrations.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events] == [20, 40]
    assert all(not e["active_overrides"] and e["forward_identical"] for e in events)
    baseline = torch.load(output / "seed_0" / "mxfp8" / "final.pt", weights_only=True)
    actual = torch.load(adaptive / "final.pt", weights_only=True)
    assert baseline["step"] == actual["step"] == 40
    for name, parameter in baseline["model"].items():
        assert torch.equal(parameter, actual["model"][name]), name
    # Embeddings are active on every update, including zero-advantage batches.
    first_state = next(iter(actual["optimizer"]["state"].values()))
    assert first_state["step"].item() == 40
