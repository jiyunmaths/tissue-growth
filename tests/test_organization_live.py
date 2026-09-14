from dataclasses import asdict
import json

import numpy as np
import pytest

from tissue_growth.config import Config
from tissue_growth.organization import OrganizationModel, OrganizationParameters
from tissue_growth.organization_live import OrganizationSimulation
from tissue_growth.recording import Recorder


def test_live_matches_batch_and_records_cell_depletion(tmp_path):
    config = Config(backend="scipy", geometry="sphere", n=8, length=1,
                    max_length=1, growth_rate=0, dt=0.02, t_end=0.035)
    options = {"parameters": asdict(OrganizationParameters()), "initial_condition": "mosaic"}
    live = OrganizationSimulation(config, options)
    batch = OrganizationModel(config, initial_condition="mosaic")
    recorder = Recorder(tmp_path/"run", live)
    try:
        while live.step():
            pass
        batch.run_until(config.t_end)
        np.testing.assert_allclose(live.c, batch.state, rtol=1e-12, atol=1e-12)
        assert live.t == pytest.approx(config.t_end)
        before = live.c.copy()
        live.perturb(population="differentiated")
        np.testing.assert_array_equal(live.c[[0, 2]], before[[0, 2]])
        assert np.any(live.c[1] < before[1])
        assert live.events[-1]["removed_differentiated"] > 0
        recorder.finish(live)
        with np.load(tmp_path/"run"/"checkpoint.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["c"], live.c)
            metadata = json.loads(str(saved["metadata"]))
            assert metadata["parameters"] == options["parameters"]
            assert metadata["cumulative"] == live.cumulative
        assert json.loads((tmp_path/"run"/"run.json").read_text())["model"] == "aim1"
    finally:
        recorder.finish(live)
        live.close()
        batch.close()
