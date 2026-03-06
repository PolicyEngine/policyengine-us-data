.PHONY: all format test install download upload docker documentation data validate-data calibrate calibrate-build publish-local-area upload-calibration upload-dataset upload-database push-to-modal build-matrices calibrate-modal calibrate-modal-national calibrate-both stage-h5s stage-national-h5 stage-all-h5s pipeline validate-staging validate-staging-full upload-validation check-staging check-sanity clean build paper clean-paper presentations database database-refresh promote-database promote-dataset promote build-h5s validate-local

GPU ?= A100-80GB
EPOCHS ?= 200
NATIONAL_GPU ?= T4
NATIONAL_EPOCHS ?= 200
BRANCH ?= $(shell git rev-parse --abbrev-ref HEAD)
NUM_WORKERS ?= 8
VERSION ?=

HF_CLONE_DIR ?= $(HOME)/huggingface/policyengine-us-data

all: data test

format:
	ruff format .

test:
	pytest

install:
	pip install policyengine-us
	pip install -e ".[dev]"  --config-settings editable_mode=compat

changelog:
	python .github/bump_version.py
	towncrier build --yes --version $$(python -c "import re; print(re.search(r'version = \"(.+?)\"', open('pyproject.toml').read()).group(1))")
download:
	python policyengine_us_data/storage/download_private_prerequisites.py

upload:
	python policyengine_us_data/storage/upload_completed_datasets.py

docker:
	docker buildx build --platform linux/amd64 . -t policyengine-us-data:latest

documentation:
	cd docs && \
	rm -rf _build .jupyter_cache && \
	rm -f _toc.yml && \
	myst clean && \
	myst build --html
	cd docs && test -d _build/html && touch _build/html/.nojekyll || true

documentation-build:
	cd docs && \
	rm -rf _build .jupyter_cache && \
	rm -f _toc.yml && \
	myst clean && \
	myst build --html

documentation-serve:
	cd docs/_build/site && python3 -m http.server 8080

documentation-dev:
	cd docs && \
	rm -rf _build .jupyter_cache && \
	rm -f _toc.yml && \
	myst clean && \
	myst start

database:
	rm -f policyengine_us_data/storage/calibration/policy_data.db
	python policyengine_us_data/db/create_database_tables.py
	python policyengine_us_data/db/create_initial_strata.py
	python policyengine_us_data/db/etl_national_targets.py
	python policyengine_us_data/db/etl_age.py
	python policyengine_us_data/db/etl_medicaid.py
	python policyengine_us_data/db/etl_snap.py
	python policyengine_us_data/db/etl_state_income_tax.py
	python policyengine_us_data/db/etl_irs_soi.py
	python policyengine_us_data/db/etl_pregnancy.py
	python policyengine_us_data/db/validate_database.py

database-refresh:
	rm -f policyengine_us_data/storage/calibration/policy_data.db
	rm -rf policyengine_us_data/storage/calibration/raw_inputs/
	$(MAKE) database

promote-database:
	sqlite3 policyengine_us_data/storage/calibration/policy_data.db "PRAGMA wal_checkpoint(TRUNCATE);"
	cp policyengine_us_data/storage/calibration/policy_data.db \
		$(HF_CLONE_DIR)/calibration/policy_data.db
	rm -rf $(HF_CLONE_DIR)/calibration/raw_inputs
	cp -r policyengine_us_data/storage/calibration/raw_inputs \
		$(HF_CLONE_DIR)/calibration/raw_inputs
	@echo "Copied DB and raw_inputs to HF clone. Now cd to HF repo, commit, and push."

promote-dataset:
	cp policyengine_us_data/storage/source_imputed_stratified_extended_cps_2024.h5 \
		$(HF_CLONE_DIR)/calibration/source_imputed_stratified_extended_cps.h5
	@echo "Copied dataset to HF clone. Now cd to HF repo, commit, and push."

data: download
	python policyengine_us_data/utils/uprating.py
	python policyengine_us_data/datasets/acs/acs.py
	python policyengine_us_data/datasets/cps/cps.py
	python policyengine_us_data/datasets/puf/irs_puf.py
	python policyengine_us_data/datasets/puf/puf.py
	python policyengine_us_data/datasets/cps/extended_cps.py
	python policyengine_us_data/calibration/create_stratified_cps.py
	python policyengine_us_data/calibration/create_source_imputed_cps.py

data-legacy: data
	python policyengine_us_data/datasets/cps/enhanced_cps.py
	python policyengine_us_data/datasets/cps/small_enhanced_cps.py

calibrate: data
	python -m policyengine_us_data.calibration.unified_calibration \
		--target-config policyengine_us_data/calibration/target_config.yaml

calibrate-build: data
	python -m policyengine_us_data.calibration.unified_calibration \
		--target-config policyengine_us_data/calibration/target_config.yaml \
		--build-only

validate-package:
	python -m policyengine_us_data.calibration.validate_package

publish-local-area:
	python policyengine_us_data/calibration/publish_local_area.py --upload

build-h5s:
	python -m policyengine_us_data.calibration.publish_local_area \
		--weights-path policyengine_us_data/storage/calibration/calibration_weights.npy \
		--dataset-path policyengine_us_data/storage/source_imputed_stratified_extended_cps_2024.h5 \
		--db-path policyengine_us_data/storage/calibration/policy_data.db \
		--calibration-blocks policyengine_us_data/storage/calibration/stacked_blocks.npy \
		--stacked-takeup policyengine_us_data/storage/calibration/stacked_takeup.npz \
		--states-only

validate-local:
	python -m policyengine_us_data.calibration.validate_staging \
		--hf-prefix local_area_build \
		--area-type states --output validation_results.csv

validate-data:
	python -c "from policyengine_us_data.storage.upload_completed_datasets import validate_all_datasets; validate_all_datasets()"

upload-calibration:
	python -c "from policyengine_us_data.utils.huggingface import upload_calibration_artifacts; \
		upload_calibration_artifacts()"

upload-dataset:
	python -c "from policyengine_us_data.utils.huggingface import upload; \
		upload('policyengine_us_data/storage/source_imputed_stratified_extended_cps_2024.h5', \
		'policyengine/policyengine-us-data', \
		'calibration/source_imputed_stratified_extended_cps.h5')"
	@echo "Dataset uploaded to HF."

upload-database:
	python -c "from policyengine_us_data.utils.huggingface import upload; \
		upload('policyengine_us_data/storage/calibration/policy_data.db', \
		'policyengine/policyengine-us-data', \
		'calibration/policy_data.db')"
	@echo "Database uploaded to HF."

push-to-modal:
	modal volume put local-area-staging \
		policyengine_us_data/storage/calibration/calibration_weights.npy \
		calibration_inputs/calibration/calibration_weights.npy --force
	modal volume put local-area-staging \
		policyengine_us_data/storage/calibration/stacked_blocks.npy \
		calibration_inputs/calibration/stacked_blocks.npy --force
	modal volume put local-area-staging \
		policyengine_us_data/storage/calibration/stacked_takeup.npz \
		calibration_inputs/calibration/stacked_takeup.npz --force
	modal volume put local-area-staging \
		policyengine_us_data/storage/calibration/policy_data.db \
		calibration_inputs/calibration/policy_data.db --force
	modal volume put local-area-staging \
		policyengine_us_data/storage/calibration/geo_labels.json \
		calibration_inputs/calibration/geo_labels.json --force
	modal volume put local-area-staging \
		policyengine_us_data/storage/source_imputed_stratified_extended_cps_2024.h5 \
		calibration_inputs/calibration/source_imputed_stratified_extended_cps.h5 --force
	@echo "All calibration inputs pushed to Modal volume."

build-matrices:
	modal run modal_app/remote_calibration_runner.py::build_package \
		--branch $(BRANCH)

calibrate-modal:
	modal run modal_app/remote_calibration_runner.py::main \
		--branch $(BRANCH) --gpu $(GPU) --epochs $(EPOCHS) \
		--push-results

calibrate-modal-national:
	modal run modal_app/remote_calibration_runner.py::main \
		--branch $(BRANCH) --gpu $(NATIONAL_GPU) \
		--epochs $(NATIONAL_EPOCHS) \
		--push-results --national

calibrate-both:
	$(MAKE) calibrate-modal & $(MAKE) calibrate-modal-national & wait

stage-h5s:
	modal run modal_app/local_area.py::main \
		--branch $(BRANCH) --num-workers $(NUM_WORKERS) \
		$(if $(SKIP_DOWNLOAD),--skip-download)

stage-national-h5:
	modal run modal_app/local_area.py::main_national \
		--branch $(BRANCH)

stage-all-h5s:
	$(MAKE) stage-h5s & $(MAKE) stage-national-h5 & wait

promote:
	$(eval VERSION := $(or $(VERSION),$(shell python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")))
	modal run modal_app/local_area.py::main_promote \
		--branch $(BRANCH) --version $(VERSION)

validate-staging:
	python -m policyengine_us_data.calibration.validate_staging \
		--area-type states --output validation_results.csv

validate-staging-full:
	python -m policyengine_us_data.calibration.validate_staging \
		--area-type states,districts --output validation_results.csv

upload-validation:
	python -c "from policyengine_us_data.utils.huggingface import upload; \
		upload('validation_results.csv', \
		'policyengine/policyengine-us-data', \
		'calibration/logs/validation_results.csv')"

check-staging:
	python -m policyengine_us_data.calibration.check_staging_sums

check-sanity:
	python -m policyengine_us_data.calibration.validate_staging \
		--sanity-only --area-type states --areas NC

pipeline: data upload-dataset build-matrices calibrate-both stage-all-h5s
	@echo ""
	@echo "========================================"
	@echo "Pipeline complete. H5s are in HF staging."
	@echo "Run 'Promote Local Area H5 Files' workflow in GitHub to publish."
	@echo "========================================"

clean:
	rm -f policyengine_us_data/storage/*.h5
	rm -f policyengine_us_data/storage/*.db
	git clean -fX -- '*.csv'
	rm -rf policyengine_us_data/docs/_build

build:
	python -m build

publish:
	twine upload dist/*

paper-content:
	@echo "Building paper sections and docs from unified content..."
	python paper/scripts/build_from_content.py

paper-tables:
	@echo "Generating all LaTeX tables..."
	python paper/scripts/generate_all_tables.py

paper-results: paper-tables
	@echo "Generating paper results tables and figures..."
	python paper/scripts/generate_validation_metrics.py
	python paper/scripts/calculate_distributional_metrics.py
	@echo "Paper results generated in paper/results/"

paper: paper-content paper-results paper/woodruff_ghenis_2024_enhanced_cps.pdf

paper/woodruff_ghenis_2024_enhanced_cps.pdf: $(wildcard paper/sections/**/*.tex) $(wildcard paper/bibliography/*.bib) paper/main.tex paper/macros.tex
	cd paper && \
	BIBINPUTS=./bibliography pdflatex main && \
	BIBINPUTS=./bibliography bibtex main && \
	pdflatex -jobname=woodruff_ghenis_2024_enhanced_cps main && \
	pdflatex -jobname=woodruff_ghenis_2024_enhanced_cps main

clean-paper:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc paper/*.pdf paper/sections/**/*.aux

presentations: presentations/nta_2024_11/nta_2024_slides.pdf

presentations/nta_2024_11/nta_2024_slides.pdf: presentations/nta_2024_11/main.tex
	cd presentations/nta_2024_11 && \
		pdflatex -jobname=nta_2024_slides main && \
		pdflatex -jobname=nta_2024_slides main
