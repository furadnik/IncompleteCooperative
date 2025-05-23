VENV=.venv
PYTEST_OPT=
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

FAIL_UNDER=100
PYTEST_FILE=incomplete_cooperative/tests

test: test_unittest test_types test_docs test_security

anonym.zip: setup.cfg clean
	mkdir -p anonym
	cp incomplete_cooperative scripts setup.cfg setup.py anonym -r
	sed -i 's/Filip Úradník/Anonymous Author/' anonym/setup.cfg 
	sed -i 's/filip.uradnik9/anonymous.author/' anonym/setup.cfg 
	sed -i 's/pyfmtools.*/pyfmtools/' anonym/setup.cfg 
	cat anonym/setup.cfg
	zip -r anonym anonym
	rm -r anonym

$(VENV)/bin/activate: setup.cfg
	rm -rf $(VENV) || echo "venv dir didn't exist"
	python -m venv $(VENV)
	$(PIP) install --upgrade .[dev]

test_unittest: $(VENV)/bin/activate
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m pytest $(PYTEST_OPT) $(PYTEST_FILE)
	$(PYTHON) -m coverage report -i --show-missing --fail-under=$(FAIL_UNDER)
	
test_types: $(VENV)/bin/activate
	$(PYTHON) -m mypy incomplete_cooperative

test_docs: $(VENV)/bin/activate
	$(PYTHON) -m pydocstyle incomplete_cooperative

test_security: $(VENV)/bin/activate
	$(PYTHON) -m bandit -r incomplete_cooperative -s B101

clean:
	rm -rf build
	rm -rf incomplete_cooperative.*
	rm -rf .venv
	find . -name __pycache__ -exec rm -rf {} \;
	rm -rf *.png *.json

.PHONY: test test_unittest test_types test_docs test_security clean

