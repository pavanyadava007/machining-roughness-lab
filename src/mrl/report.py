"""Figures, a one-page HTML results view (docs/index.html) and an Excel workbook for process engineers."""

from __future__ import annotations

import base64
import html
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mrl.config import DATASET_DOI, DATASET_URL, DOCS, PROCESSED, RESULTS  # noqa: E402

BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
LABELS = {
    "mean_baseline": "mean of training cuts",
    "kinematic_f2_32r_raw": "kinematic f²/32r (raw)",
    "kinematic_f2_32r_calibrated": "kinematic f²/32r (calibrated)",
    "process_ridge_quadratic": "process params, quadratic ridge",
    "process_gpr": "process params, GP",
    "depth_group_mean": "mean per depth of cut",
    "sensor_ridge": "sensor features, ridge",
    "sensor_pls2": "sensor features, PLS-2",
    "sensor_select5_ridge": "sensor top-5 (chosen in fold), ridge",
    "sensor_lgbm_windows": "sensor windows, LightGBM",
    "sensor_process_ridge": "sensor + process, ridge",
    "sensor_process_lgbm_windows": "sensor + process windows, LightGBM",
}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_models(m: dict, target: str):
    t = m["targets"][target]
    loco = {k: v["rmse"] for k, v in t["leave_one_cut_out"].items() if k != "kinematic_f2_32r_raw"}
    names = sorted(loco, key=loco.get, reverse=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    colors = [ORANGE if n == "mean_baseline" else (BLUE if n.startswith("sensor") else AQUA) for n in names]
    ax.barh([LABELS[n] for n in names], [loco[n] for n in names], color=colors, height=0.6)
    floor = t["noise_floor"]["rmse_floor_cut_mean"]
    ax.axvline(floor, color=INK, linestyle=":", linewidth=1.5)
    ax.text(floor, len(names) - 0.4, f" measurement floor {floor:.3f}", fontsize=8, color=INK, va="bottom")
    ax.axvline(loco["mean_baseline"], color=ORANGE, linewidth=1)
    ax.set_xlabel(f"{target} RMSE on held-out cuts (µm), leave-one-cut-out, lower is better", color=MUTED, fontsize=9)
    raw = t["leave_one_cut_out"]["kinematic_f2_32r_raw"]["rmse"]
    fig.suptitle(f"{target}: no model beats the mean of the other cuts\n(raw f²/32r theory: {raw:.2f} µm, off scale)",
                 fontsize=9.5, color=INK, x=0.02, ha="left")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=ORANGE, label="baseline"), Patch(color=AQUA, label="process / theory"),
                       Patch(color=BLUE, label="sensor features")], fontsize=8, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.16), ncol=3)
    _style(ax)
    fig.tight_layout()
    return fig


def fig_leakage(m: dict):
    d = m["targets"]["Ra"]["leakage_demo_lgbm"]
    vals = [d["random_window_kfold5"]["r2"], d["leave_one_cut_out_windows"]["r2"]]
    fig, ax = plt.subplots(figsize=(6.0, 2.4), dpi=150)
    labels = ["random 1 s windows (5-fold)", "whole cuts held out"]
    ax.barh(labels[::-1], vals[::-1], color=[BLUE, ORANGE], height=0.5)
    for i, v in enumerate(vals[::-1]):
        ax.text(0.03 if v < 0 else v + 0.03, i, f"R² {v:+.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set_xlim(-0.9, 1.15)
    fig.suptitle("Same LightGBM, same windows: the split decides the headline", fontsize=9.5, color=INK, x=0.02,
                 ha="left")
    _style(ax)
    fig.tight_layout()
    return fig


def fig_audit(audit: dict):
    s = pd.Series(audit["cut_length_mm"])
    med = s.median()
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    colors = [ORANGE if abs(v - med) / med > 0.3 else BLUE for v in s]
    ax.bar(range(len(s)), s.to_numpy(), color=colors, width=0.7)
    ax.set_xticks(range(len(s)), [e.replace("_", " ") for e in s.index], rotation=90, fontsize=7)
    ax.set_ylabel("duration x feed x rev/s (mm)", color=MUTED, fontsize=9)
    ax.set_title("Implied cut length: 6 cuts at 630 rpm (f 0.10/0.16) are 2x the rest",
                 fontsize=9.5, color=INK, loc="left")
    _style(ax)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_drift(win: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    for e, c in (("0400_0.16_0.40", BLUE), ("0630_0.10_0.60", ORANGE), ("1000_0.05_0.60", AQUA)):
        w = win[win.exp == e].sort_values("win")
        ax.plot(w.win, w.Fres_mean, color=c, linewidth=2, label=f"{e[:4]} rpm, f {e[5:9]}, d {e[10:]}")
    ax.set_xlabel("time in cut (s)", color=MUTED, fontsize=9)
    ax.set_ylabel("mean resultant force per 1 s (N)", color=MUTED, fontsize=9)
    ax.set_title("Force is not stationary within a cut (median +38 % first to last 10 %)", fontsize=9.5, color=INK,
                 loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    fig.tight_layout()
    return fig


def _png(fig, path) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    fig.savefig(path)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:.3f}", escape=True)


def model_frame(m: dict, target: str) -> pd.DataFrame:
    t = m["targets"][target]
    rows = []
    for k, v in t["leave_one_cut_out"].items():
        b = t["vs_mean_baseline_bootstrap"].get(k)
        rows.append({"model": LABELS[k], "RMSE cut-out": v["rmse"], "R² cut-out": v["r2"],
                     "RMSE speed-out": t["leave_one_speed_out"][k]["rmse"],
                     "Δ RMSE vs mean, 95 % CI": "-" if b is None else f"{b['ci95'][0]:+.3f} .. {b['ci95'][1]:+.3f}"})
    return pd.DataFrame(rows).sort_values("RMSE cut-out")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    m = json.loads((RESULTS / "metrics.json").read_text())
    audit = json.loads((RESULTS / "audit.json").read_text())
    plan = json.loads((RESULTS / "plan.json").read_text())
    win = pd.read_parquet(PROCESSED / "windows.parquet")
    imgs = {
        "models_ra": _png(fig_models(m, "Ra"), fig_dir / "models_ra.png"),
        "leakage": _png(fig_leakage(m), fig_dir / "leakage.png"),
        "audit": _png(fig_audit(audit), fig_dir / "audit_cut_length.png"),
        "drift": _png(fig_drift(win), fig_dir / "force_drift.png"),
    }
    ra, rz = model_frame(m, "Ra"), model_frame(m, "Rz")
    aud = pd.DataFrame(audit["checks"])[["verdict", "check", "expect", "observed"]]
    nxt = pd.DataFrame(plan["next_trials"])
    rep = pd.DataFrame(plan["replicates"])
    preds = pd.read_csv(RESULTS / "predictions.csv")
    nf = m["targets"]["Ra"]["noise_floor"]
    leak = m["targets"]["Ra"]["leakage_demo_lgbm"]
    perm = m["targets"]["Ra"]["permutation_sensor_ridge"]

    with pd.ExcelWriter(RESULTS / "roughness_report.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({"item": ["dataset", "cuts", "1 s windows", "features per window", "Ra measurement floor (µm)",
                               "Ra R² ceiling", "random-split R² (leaky)", "cut-out R² (honest)",
                               "permutation p (sensor ridge)"],
                      "value": [f"{DATASET_URL} (CC BY 4.0)", m["n_cuts"], m["n_windows"], m["n_features"],
                                round(nf["rmse_floor_cut_mean"], 3), round(nf["r2_ceiling"], 2),
                                round(leak["random_window_kfold5"]["r2"], 2),
                                round(leak["leave_one_cut_out_windows"]["r2"], 2), round(perm["p_value"], 2)]}
                     ).to_excel(xw, sheet_name="Summary", index=False)
        ra.to_excel(xw, sheet_name="Models_Ra", index=False)
        rz.to_excel(xw, sheet_name="Models_Rz", index=False)
        preds.to_excel(xw, sheet_name="Predictions", index=False)
        aud.to_excel(xw, sheet_name="Plausibility", index=False)
        nxt.to_excel(xw, sheet_name="Next_trials", index=False)
        rep.to_excel(xw, sheet_name="Replicates", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, *(len(str(c.value or ""))
                                                                                     for c in col)))

    esc = html.escape
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Machining Roughness Lab</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e6e5e0; --card:#ffffff; --accent:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#ffffff; --muted:#c3c2b7;
  --line:#3a3a37; --card:#232322; --accent:#3987e5; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --card:#232322; --accent:#3987e5; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:900px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; }} h2 {{ font-size:1.15rem; margin:36px 0 8px; }}
p.sub {{ color:var(--muted); margin:0 0 20px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.kpi b {{ display:block; font-size:1.4rem; }} .kpi span {{ color:var(--muted); font-size:.85rem; }}
img {{ max-width:100%; height:auto; background:#fcfcfb; border-radius:6px; border:1px solid var(--line); }}
.scroll {{ overflow-x:auto; }}
table.t {{ border-collapse:collapse; font-size:.85rem; width:100%; }}
table.t th, table.t td {{ text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
table.t th {{ color:var(--muted); font-weight:600; }}
a {{ color:var(--accent); }} code {{ font-size:.9em; }}
</style></head><body><main>
<h1>Machining Roughness Lab</h1>
<p class="sub">Can 10 kHz force and vibration signals predict surface roughness in Inconel-625 turning?
27 cuts, {m['n_windows']} one-second windows, {m['n_features']} features per window. Data:
<a href="{DATASET_URL}">Mendeley Data {DATASET_DOI}</a> (CC BY 4.0, Amrita Vishwa Vidyapeetham). Code:
<a href="https://github.com/pavanyadava007/machining-roughness-lab">GitHub</a>.</p>
<div class="kpis">
<div class="kpi"><b>{leak['random_window_kfold5']['r2']:+.2f}</b><span>R² with a random window split (leaky)</span></div>
<div class="kpi"><b>{leak['leave_one_cut_out_windows']['r2']:+.2f}</b><span>R² with whole cuts held out</span></div>
<div class="kpi"><b>p = {perm['p_value']:.2f}</b><span>permutation test, sensor model vs shuffled labels</span></div>
<div class="kpi"><b>{nf['r2_ceiling']:.2f}</b><span>R² ceiling set by repeat Ra readings</span></div>
</div>
<h2>1. Plausibility audit first</h2>
<p>Before any model: does the trial data behave like turning? Two checks fail and the log needs a question to the lab.</p>
<div class="scroll">{_table(aud)}</div>
<p><img src="data:image/png;base64,{imgs['audit']}" alt="Implied cut length per cut; six cuts at 630 rpm are twice as long"></p>
<p><img src="data:image/png;base64,{imgs['drift']}" alt="Mean resultant force over time for three cuts, rising within each cut"></p>
<h2>2. The split decides the headline</h2>
<p>Neighbouring 1 s windows of one cut share the same roughness label and nearly the same signal. A random split puts
them on both sides and reports R² {leak['random_window_kfold5']['r2']:.2f}. Holding out whole cuts gives
{leak['leave_one_cut_out_windows']['r2']:+.2f}.</p>
<p><img src="data:image/png;base64,{imgs['leakage']}" alt="R2 random split vs held-out cuts"></p>
<h2>3. Twelve models, cut-level validation</h2>
<p>Leave-one-cut-out (27 folds) and leave-one-speed-out (3 folds). The bootstrap column is the 95 % interval of the
RMSE difference to the mean baseline; every interval includes zero or is worse.</p>
<p><img src="data:image/png;base64,{imgs['models_ra']}" alt="Ra RMSE per model with the measurement floor"></p>
<div class="scroll"><b>Ra (µm)</b>{_table(ra)}</div>
<div class="scroll"><b>Rz (µm)</b>{_table(rz)}</div>
<h2>4. What to run next</h2>
<p>A Gaussian process on speed, feed and depth finds only depth informative. Greedy max-variance selection proposes
these validation cuts (re-run the six double-length cuts first). Repeat readings per cut shrink the floor as 1/&radic;n.</p>
<div class="scroll">{_table(nxt[['order', 'speed', 'feed', 'depth', 'pred_Ra_um', 'pred_sd_um']])}</div>
<div class="scroll">{_table(rep)}</div>
<h2>5. Trial plan to simulation input decks</h2>
<p><code>python -m mrl.siminput examples/grinding_plan.csv</code> validates each row (units, ranges, speed ratio)
and writes one deck per trial with derived grinding quantities (q = vs/vw, equivalent chip thickness, Q'w, contact
length). Rejected rows are listed with the reason. The deck format is a template, not any company's real format.</p>
<h2>Limits</h2>
<p>One public dataset, 27 cuts, one bar of Inconel-625, turning (not grinding). Roughness is measured at three random
points per cut, so it cannot be tied to individual windows. A negative result here says this trial set is too small
and too inconsistent for a sensor model, not that sensor-based roughness models cannot work.</p>
<p class="sub">{esc('Pavan Yadav Annappa')} - all numbers are produced by <code>make all</code> from the public data.</p>
</main></body></html>"""
    (DOCS / "index.html").write_text(page)
    print(f"wrote {DOCS / 'index.html'} ({len(page) // 1024} KB), {RESULTS / 'roughness_report.xlsx'}, figures")


if __name__ == "__main__":
    main()
