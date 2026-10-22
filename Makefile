PYTHON ?= python
STREAMLIT ?= streamlit

.PHONY: help install data train app test lint clean all

help:
	@echo "HR Insight - Smart HR Decision Support Platform"
	@echo ""
	@echo "  make install   Install Python dependencies"
	@echo "  make data      Download and verify the IBM HR dataset"
	@echo "  make train     Run the full pipeline (models + dataset)"
	@echo "  make app       Start the Streamlit application"
	@echo "  make test      Run the test suite"
	@echo "  make clean     Remove generated artifacts and caches"
	@echo "  make all       install + data + train + test"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -c "import sys; sys.path.insert(0,'.'); from ml.common import io; p=io.download_ibm_hr(); print('dataset ready:', p)"

train:
	$(PYTHON) ml/train_all.py

app:
	$(STREAMLIT) run app/main.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf ml/artifacts/*.pkl ml/artifacts/*.json
	rm -f data/processed/*.csv
	rm -f data/db/*.db
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache

all: install data train test
