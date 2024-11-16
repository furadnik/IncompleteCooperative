VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
FAIL_UNDER=100

test: test_unittest test_types test_docs test_security

$(VENV)/bin/activate: setup.cfg
	python -m venv $(VENV)
	$(PIP) install --upgrade .[dev]

test_unittest: $(VENV)/bin/activate
	$(PYTHON) -m coverage run -m unittest --locals --failfast -k incomplete_cooperative.tests
	$(PYTHON) -m coverage report --show-missing --fail-under=$(FAIL_UNDER)
	
	

test_types: $(VENV)/bin/activate
	$(PYTHON) -m mypy incomplete_cooperative

test_docs: $(VENV)/bin/activate
	$(PYTHON) -m pydocstyle incomplete_cooperative

test_security: $(VENV)/bin/activate
	$(PYTHON) -m bandit -r incomplete_cooperative

.PHONY: test test_unittest test_types test_docs test_security

