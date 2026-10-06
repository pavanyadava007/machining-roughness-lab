"""Trial plan (CSV) -> validated input decks for a process simulation tool.

The deck layout is a plain key = value template (templates/*.inp.tmpl) so it can be adapted to
whatever format an in-house tool reads. Every row is checked against unit-aware ranges first,
and derived process quantities are computed once, in one place, instead of by hand per deck.

    python -m mrl.siminput examples/grinding_plan.csv --out decks/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from mrl.config import ROOT
from mrl.features import kinematic_ra_um

TEMPLATES = ROOT / "templates"


class TurningTrial(BaseModel):
    process: Literal["turning"] = "turning"
    trial_id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    material: str
    diameter_mm: float = Field(gt=0, le=500)
    speed_rpm: float = Field(gt=0, le=4000)
    feed_mm_rev: float = Field(gt=0, le=1.0)
    depth_mm: float = Field(gt=0, le=10)
    nose_radius_mm: float = Field(gt=0, le=3.2)
    length_mm: float = Field(gt=0, le=2000)

    @model_validator(mode="after")
    def _geometry(self):
        if self.depth_mm >= self.diameter_mm / 2:
            raise ValueError("depth of cut must be below the workpiece radius")
        if self.feed_mm_rev > self.nose_radius_mm:
            raise ValueError("feed per rev larger than the nose radius gives a scalloped, non-physical profile")
        return self

    def derived(self) -> dict[str, float]:
        vc = math.pi * self.diameter_mm * self.speed_rpm / 1000.0  # m/min
        return {
            "cutting_speed_m_min": vc,
            "material_removal_mm3_min": vc * 1000.0 * self.feed_mm_rev * self.depth_mm,
            "kinematic_ra_um": kinematic_ra_um(self.feed_mm_rev, self.nose_radius_mm),
            "cut_time_s": self.length_mm / (self.feed_mm_rev * self.speed_rpm / 60.0),
        }


class GrindingTrial(BaseModel):
    process: Literal["grinding"] = "grinding"
    trial_id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    material: str
    wheel: str
    wheel_diameter_mm: float = Field(gt=0, le=1200)
    wheel_speed_m_s: float = Field(gt=0, le=200)
    work_speed_m_min: float = Field(gt=0, le=60)
    depth_mm: float = Field(gt=0, le=20)  # ae
    width_mm: float = Field(gt=0, le=500)  # b
    coolant: str = "emulsion"

    @model_validator(mode="after")
    def _ratio(self):
        q = self.wheel_speed_m_s * 60.0 / self.work_speed_m_min
        if q < 10:
            raise ValueError(f"speed ratio q = vs/vw = {q:.1f} is below 10; check units (vs in m/s, vw in m/min)")
        return self

    def derived(self) -> dict[str, float]:
        vs_m_min = self.wheel_speed_m_s * 60.0
        vw_mm_s = self.work_speed_m_min * 1000.0 / 60.0
        return {
            "speed_ratio_q": vs_m_min / self.work_speed_m_min,
            # equivalent chip thickness heq = ae * vw / vs, in um
            "equiv_chip_thickness_um": self.depth_mm * self.work_speed_m_min / vs_m_min * 1000.0,
            # specific material removal rate Q'w = ae * vw, mm^3/(mm s)
            "specific_mrr_mm3_mm_s": self.depth_mm * vw_mm_s,
            # geometric contact length lg = sqrt(ae * ds), mm
            "contact_length_mm": math.sqrt(self.depth_mm * self.wheel_diameter_mm),
            "material_removal_mm3_s": self.depth_mm * vw_mm_s * self.width_mm,
        }


MODELS = {"turning": TurningTrial, "grinding": GrindingTrial}


def render(trial: BaseModel, template: str) -> str:
    values = {**trial.model_dump(), **{k: f"{v:.6g}" for k, v in trial.derived().items()}}
    return template.format(**values)


def build_decks(plan_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, errors = [], []
    with open(plan_csv, newline="") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            process = row.get("process", "").strip()
            model = MODELS.get(process)
            if model is None:
                errors.append({"line": line_no, "trial_id": row.get("trial_id"),
                               "error": f"unknown process {process!r}"})
                continue
            try:
                trial = model(**{k: v for k, v in row.items() if v not in ("", None)})
            except ValidationError as exc:
                msgs = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'row'}: {e['msg']}" for e in exc.errors())
                errors.append({"line": line_no, "trial_id": row.get("trial_id"), "error": msgs})
                continue
            tmpl = (TEMPLATES / f"{process}.inp.tmpl").read_text()
            (out_dir / f"{trial.trial_id}.inp").write_text(render(trial, tmpl))
            ok.append({"trial_id": trial.trial_id, "process": process, **trial.derived()})
    summary = {"plan": str(plan_csv), "written": len(ok), "rejected": len(errors), "decks": ok, "errors": errors}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "decks")
    a = ap.parse_args()
    s = build_decks(a.plan, a.out)
    print(f"{s['written']} decks written to {a.out}, {s['rejected']} rows rejected")
    for e in s["errors"]:
        print(f"  line {e['line']} ({e['trial_id']}): {e['error']}")


if __name__ == "__main__":
    main()
