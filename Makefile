.PHONY: mock real

mock:
	python scripts/run_matrix.py --config configs/mock.yaml
	python scripts/summarize_runs.py --matrix_out runs/mock

real:
	python scripts/run_matrix.py --config configs/real.yaml
	python scripts/summarize_runs.py --matrix_out runs/real