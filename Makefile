.PHONY: help setup data train train-ci gen serve clean

help:
	@echo "Targets: setup data train train-ci gen serve clean"

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	python scripts/download_sample_data.py

train:
	PYTHONPATH=src python -m barebonellm.train --config configs/tiny.json

train-ci:
	PYTHONPATH=src python -m barebonellm.train --config configs/ci.json

gen:
	PYTHONPATH=src python -m barebonellm.generate --checkpoint checkpoints/model.pt --prompt "Hello"

serve:
	PYTHONPATH=src python -m barebonellm.server --checkpoint checkpoints/model.pt --host 0.0.0.0 --port 8000

clean:
	rm -rf .venv checkpoints __pycache__ .pytest_cache
