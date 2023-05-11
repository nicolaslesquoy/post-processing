VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run:
	$(PYTHON) src/main.py

freeze:
	$(PIP) freeze > requirements.txt