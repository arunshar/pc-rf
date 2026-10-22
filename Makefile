# PC-RF convenience targets. The model is the installed `pcrf` package.

VENV      ?= .venv
PY        ?= ./$(VENV)/bin/python
ARTIFACTS ?= ./artifacts

.PHONY: help venv smoke test export export-aoti

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

venv: ## create a virtualenv and install the package with the test extras
	python3 -m venv $(VENV) && $(PY) -m pip install -U pip && \
	  $(PY) -m pip install -e ".[dev,data,train,export,serve]"

smoke: ## run the end-to-end smoke test on synthetic data
	$(PY) -m pcrf.smoke_test

test: ## run the test suite
	$(PY) -m pytest -q

export: ## train a tiny model and export artifacts/pcrf.onnx
	mkdir -p $(ARTIFACTS)
	$(PY) apps/training/export_onnx.py --epochs 2 --fine 64 --base 32 \
	  --out $(abspath $(ARTIFACTS))/pcrf.onnx

export-aoti: ## PyTorch-native AOTInductor export
	mkdir -p $(ARTIFACTS)
	$(PY) apps/training/export_aoti.py --epochs 8 --fine 64 --base 32 \
	  --out $(abspath $(ARTIFACTS))/pcrf_aoti
