"""Target-free evidence for spatial lineage organization; no image-matching score."""
from dataclasses import replace
from time import perf_counter
import numpy as np

from .config import Config
from .spatial_organization import SpatialOrganizationModel


def organized(d):
    return (d["std_renewing_fraction"] > 0.08
            and d["renewing_rich_area_fraction"] > 0.03
            and d["differentiated_rich_area_fraction"] > 0.03
            and 0.1 < d["mean_occupancy"] < 1.1)


def relative_error(a, b):
    return abs(a-b)/max(abs(b), 1e-12)


def run_spatial_protocol(protocol):
    start = perf_counter()
    cfg = Config(backend="scipy", geometry=protocol.geometry, n=protocol.n, dt=protocol.dt,
                 length=1, max_length=1, growth_rate=0)
    report = {"protocol": protocol.to_dict(), "establishment": {}, "repair": {}, "controls": {}}
    reference = None
    for seed in (protocol.seed, protocol.seed+1):
        for kind in ("near_uniform", "low", "high", "mosaic", "segregated"):
            model = SpatialOrganizationModel(cfg, protocol.parameters, seed, kind)
            try:
                initial = model.diagnostics()
                settled = model.run_until(protocol.settle_time)
                totals = model.cumulative.copy()
                late = model.run_until(protocol.settle_time+protocol.turnover_time)
                report["establishment"][f"{kind}_seed_{seed}"] = {
                    "initial": initial, "settled": settled, "late": late,
                    "interval_fluxes": {key: model.cumulative[key]-value for key, value in totals.items()},
                }
                if reference is None:
                    reference = model.state.copy()
            finally:
                model.close()
    def branch():
        model = SpatialOrganizationModel(cfg, protocol.parameters, protocol.seed, "near_uniform")
        model.state = reference.copy()
        return model
    sham = branch()
    try:
        report["sham"] = sham.run_until(protocol.repair_time)
        sham_state = sham.state.copy()
    finally:
        sham.close()
    for radius, fraction in ((0.20, 0.4), (0.35, 0.6), (0.50, 0.9)):
        for erase in (False, True):
            model = branch()
            try:
                baseline = model.diagnostics()
                disturbed = model.perturb(radius=radius, fraction=fraction, erase_signals=erase)
                recovered = model.run_until(protocol.repair_time)
                field_difference = model.state[:2]-sham_state[:2]
                report["repair"][f"radius_{radius}_depletion_{fraction}_signals_{erase}"] = {
                    "radius": radius, "fraction": fraction, "erase_signals": erase,
                    "baseline": baseline, "disturbed": disturbed, "recovered": recovered,
                    "cell_field_rms_vs_sham": float(np.sqrt(np.sum(field_difference**2 @ model.ops.mass)/(2*model.ops.mass.sum()))),
                }
            finally:
                model.close()
    # These probe pattern generation and cellular response, not a fitted rescue.
    for label, parameters in {
        "no_local_activation": replace(protocol.parameters, activation=0),
        "no_niche_cell_coupling": replace(protocol.parameters, niche_coupling=0),
    }.items():
        model = SpatialOrganizationModel(cfg, parameters, protocol.seed, "near_uniform")
        try:
            report["controls"][label] = model.run_until(protocol.settle_time)
        finally:
            model.close()
    criteria = {"establishment": {}, "repair": {}}
    for name, row in report["establishment"].items():
        a, b = row["settled"], row["late"]
        f = row["interval_fluxes"]
        criteria["establishment"][name] = {
            "organized_at_both_times": organized(a) and organized(b),
            "population_drift_below_3_percent": relative_error(b["mean_occupancy"], a["mean_occupancy"]) < 0.03,
            "contrast_drift_below_15_percent": relative_error(b["std_renewing_fraction"], a["std_renewing_fraction"]) < 0.15,
            "ongoing_turnover": min(f["proliferation"], f["differentiation"], f["differentiated_loss"]) > 0.1,
        }
    normal = report["sham"]
    for name, row in report["repair"].items():
        d = row["recovered"]
        criteria["repair"][name] = {
            "organized": organized(d),
            "population_within_10_percent_of_sham": relative_error(d["mean_occupancy"], normal["mean_occupancy"]) < 0.1,
            "contrast_within_25_percent_of_sham": relative_error(d["std_renewing_fraction"], normal["std_renewing_fraction"]) < 0.25,
            "lineage_fraction_within_0.1_of_sham": abs(d["differentiated_fraction"]-normal["differentiated_fraction"]) < 0.1,
        }
    occupancies = [row["late"]["mean_occupancy"] for row in report["establishment"].values()]
    criteria["common_population_range_below_10_percent"] = float(np.ptp(occupancies)/np.mean(occupancies)) < 0.1
    criteria["near_uniform_contrast_amplified"] = all(
        row["late"]["std_renewing_fraction"] > 10*row["initial"]["std_renewing_fraction"]
        for name, row in report["establishment"].items() if name.startswith("near_uniform"))
    criteria["all_required_evidence_passed"] = bool(
        criteria["common_population_range_below_10_percent"] and criteria["near_uniform_contrast_amplified"]
        and all(all(row.values()) for group in ("establishment", "repair") for row in criteria[group].values()))
    report["criteria"] = criteria
    report["wall_seconds"] = perf_counter()-start
    return report
