import json

import pytest
from pydantic import ValidationError

from mrl.siminput import GrindingTrial, TurningTrial, build_decks

G = dict(trial_id="G1", material="IN718", wheel="cBN", wheel_diameter_mm=400, wheel_speed_m_s=35,
         work_speed_m_min=12, depth_mm=0.02, width_mm=10)


def test_grinding_derived_quantities():
    d = GrindingTrial(**G).derived()
    assert d["speed_ratio_q"] == pytest.approx(175)
    assert d["equiv_chip_thickness_um"] == pytest.approx(0.02 * 12 / 2100 * 1000)
    assert d["specific_mrr_mm3_mm_s"] == pytest.approx(4.0)
    assert d["contact_length_mm"] == pytest.approx((0.02 * 400) ** 0.5)


def test_unit_mixup_is_rejected():
    # work speed typed in mm/s instead of m/min -> q collapses below 10
    with pytest.raises(ValidationError, match="speed ratio"):
        GrindingTrial(**{**G, "work_speed_m_min": 59, "wheel_speed_m_s": 5})


def test_turning_geometry_checks():
    with pytest.raises(ValidationError):
        TurningTrial(trial_id="T", material="x", diameter_mm=32, speed_rpm=400, feed_mm_rev=0.1,
                     depth_mm=16, nose_radius_mm=0.8, length_mm=45)


def test_build_decks_writes_and_rejects(tmp_path):
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "process,trial_id,material,wheel,wheel_diameter_mm,wheel_speed_m_s,work_speed_m_min,depth_mm,width_mm\n"
        "grinding,G1,IN718,cBN,400,35,12,0.02,10\n"
        "grinding,G2,IN718,cBN,400,35,12,-1,10\n"
        "milling,M1,IN718,,,,,,\n")
    s = build_decks(plan, tmp_path / "decks")
    assert s["written"] == 1 and s["rejected"] == 2
    deck = (tmp_path / "decks" / "G1.inp").read_text()
    assert "speed_ratio_q = 175" in deck
    assert json.loads((tmp_path / "decks" / "summary.json").read_text())["errors"][1]["line"] == 4
