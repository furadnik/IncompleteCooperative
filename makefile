VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
FAIL_UNDER=50
COVERAGE_OMIT=incomplete_cooperative/game_loader.py

test_all: test

commit:
	git add .
	git commit --amend --no-edit
	git push --force-with-lease

$(VENV)/bin/activate: setup.cfg
	python -m venv $(VENV)
	$(PIP) install .[dev] --upgrade-strategy=eager

test: quality $(VENV)/bin/activate
	$(PYTHON) -m coverage run -m unittest -c
	$(PYTHON) -m coverage report --omit=$(COVERAGE_OMIT) --show-missing --fail-under=$(FAIL_UNDER)

quality: $(VENV)/bin/activate
	$(PYTHON) -m mypy incomplete_cooperative
	$(PYTHON) -m pydocstyle incomplete_cooperative
	$(PYTHON) -m bandit -r incomplete_cooperative

.PHONY: test test_all quality
