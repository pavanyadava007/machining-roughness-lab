PY ?= .venv/bin/python
UV ?= $(HOME)/.local/bin/uv

.PHONY: setup data features evaluate audit plan report decks gate all

setup:      ## Python 3.11 venv with pinned dependencies
	$(UV) venv -p 3.11 .venv
	$(UV) pip install -p $(PY) -r requirements.lock
	$(UV) pip install -p $(PY) -e . --no-deps

data:       ## download 81 CSVs (about 1.7 GB) from Mendeley Data and check sha256
	$(PY) -m mrl.data

features:   ## 1 s windows -> data/processed/windows.parquet + labels.csv
	$(PY) -m mrl.prepare

audit:      ## plausibility checks on the trial data (needs data/raw for the spindle check)
	$(PY) -m mrl.audit

evaluate:   ## 12 models, leave-one-cut-out + leave-one-speed-out, permutation test, leakage demo
	$(PY) -m mrl.evaluate

plan:       ## next validation cuts (GP max variance) + replicate table
	$(PY) -m mrl.plan

report:     ## figures, docs/index.html, results/roughness_report.xlsx
	$(PY) -m mrl.report

decks:      ## example trial plans -> simulation input decks
	$(PY) -m mrl.siminput examples/grinding_plan.csv --out decks/grinding
	$(PY) -m mrl.siminput examples/turning_plan.csv --out decks/turning

gate:       ## ruff + pytest (synthetic data only, no download)
	$(PY) -m ruff check .
	$(PY) -m pytest -q

all: data features audit evaluate plan report decks gate
