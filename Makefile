CONDA ?= conda
ENV_PREFIX ?= .conda
PYTHON_VERSION ?= 3.13

.PHONY: setup train test

setup:
	@if [ ! -x "$(ENV_PREFIX)/bin/python" ]; then \
		$(CONDA) create --prefix "$(ENV_PREFIX)" "python=$(PYTHON_VERSION)" pip -y; \
	fi
	$(CONDA) run --prefix "$(ENV_PREFIX)" python -m pip install -e '.[dev]'

train: setup
	$(CONDA) run --prefix "$(ENV_PREFIX)" train-transformer

test: setup
	$(CONDA) run --prefix "$(ENV_PREFIX)" python -m pytest