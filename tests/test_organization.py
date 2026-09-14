import numpy as np
import pytest

from tissue_growth.config import Config
from tissue_growth.organization import (
    Aim1Protocol,
    OrganizationModel,
    OrganizationParameters,
    run_aim1_protocol,
)


@pytest.fixture(scope="module")
def aim1_report():
    # Shorter and coarser than the reference protocol, but it exercises every
    # initial condition, disturbance, and single-mechanism control.
    return run_aim1_protocol(Aim1Protocol(
        n=8, dt=0.02, settle_time=40, turnover_time=10, repair_time=30, seed=42,
    ))


def test_common_rules_establish_same_organization(aim1_report):
    checks = aim1_report["criteria"]["establishment"]
    assert all(checks.values()), checks
    initial_means = [
        entry["initial"]["mean_occupancy"]
        for entry in aim1_report["establishment"].values()
    ]
    assert max(initial_means) > 5*min(initial_means)
    assert aim1_report["establishment"]["high"]["settled"]["cumulative_extrusion"] > 0


def test_turnover_continues_at_stable_population(aim1_report):
    checks = aim1_report["criteria"]["turnover"]
    assert all(checks.values()), checks
    fluxes = aim1_report["turnover"]["interval_fluxes"]
    assert fluxes["differentiation"] > 0
    assert fluxes["differentiated_loss"] > 0


def test_common_rules_repair_three_finite_disturbances(aim1_report):
    for label, checks in aim1_report["criteria"]["repair"].items():
        assert all(checks.values()), (label, checks)
    disturbances = list(aim1_report["repair"].values())
    assert disturbances[-1]["disturbed"]["cv_occupancy"] > disturbances[0]["disturbed"]["cv_occupancy"]


def test_imposed_pattern_is_not_a_restoring_target(aim1_report):
    checks = aim1_report["criteria"]["maintenance_control"]
    assert all(checks.values()), checks
    control = aim1_report["maintenance_control"]
    assert control["settled"]["cv_occupancy"] < 0.01*control["initial"]["cv_occupancy"]
    assert aim1_report["criteria"]["all_required_evidence_passed"]


def test_feedback_controls_separate_population_and_spatial_outputs(aim1_report):
    effects = aim1_report["criteria"]["mechanism_effects"]
    assert effects["no_differentiated_feedback"]["settled_occupancy_ratio_to_normal"] > 1.1
    assert effects["no_niche_coupling"]["settled_occupancy_ratio_to_normal"] > 1.1
    assert effects["no_mechanical_feedback"]["settled_occupancy_ratio_to_normal"] > 5
    # The first two controls still smooth a finite lesion, even though their
    # regulated population differs. Population and spatial outputs separate.
    assert effects["no_differentiated_feedback"]["recovered_cv"] < 0.01
    assert effects["no_niche_coupling"]["recovered_cv"] < 0.01


def test_implicit_diffusion_preserves_reaction_balance():
    config = Config(
        backend="scipy", geometry="sphere", n=8, dt=0.02,
        length=1, max_length=1, growth_rate=0,
    )
    model = OrganizationModel(config, seed=5, initial_condition="mosaic")
    try:
        before = model.state @ model.ops.mass
        expected = before+config.dt*(model.reaction() @ model.ops.mass)
        model.step()
        np.testing.assert_allclose(model.state @ model.ops.mass, expected, rtol=1e-9, atol=1e-11)
    finally:
        model.close()


def test_organization_input_validation(tmp_path):
    with pytest.raises(ValueError):
        OrganizationParameters(carrying_capacity=0)
    with pytest.raises(ValueError):
        OrganizationParameters(niche_coupling=1.1)
    with pytest.raises(ValueError):
        Aim1Protocol(n=2)
    with pytest.raises(FileNotFoundError):
        Aim1Protocol.read(tmp_path/"missing.json")
    config = Config(backend="scipy", geometry="square", n=8, length=1, max_length=1, growth_rate=0)
    model = OrganizationModel(config)
    try:
        with pytest.raises(ValueError):
            model.perturb(population="unknown")
        with pytest.raises(ValueError):
            model.initial_state("installed_target")
    finally:
        model.close()
