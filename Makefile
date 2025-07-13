.PHONY: help setup test clean lab diagram progress

help:
	@echo "Available commands:"
	@echo "  make setup     - Set up Python environment"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean temporary files"
	@echo "  make lab LAB=1 - Run a specific lab"
	@echo "  make diagram LAB=lab1 - Create diagram for a lab"
	@echo "  make progress  - Check learning progress"

setup:
	python3.11 -m venv venv
	. venv/bin/activate && pip install -r requirements_essential.txt

test:
	. venv/bin/activate && pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f **init**.py __init_py

lab:
	. venv/bin/activate && python hands-on-labs/vertex-ai/lab$(LAB)_*.py

diagram:
	. venv/bin/activate && python scripts/create_diagram.py --lab $(LAB)

progress:
	. venv/bin/activate && python scripts/track_progress.py
