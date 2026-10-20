.PHONY: setup pipeline test notebook dashboard all

setup:
	pip install -r requirements.txt && pip install -e .

pipeline:
	python -m rfm.pipeline

test:
	pytest -q

notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/customer_rfm_analysis.ipynb

dashboard:
	streamlit run dashboard/app.py

all: pipeline test notebook
