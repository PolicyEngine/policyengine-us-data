.PHONY: all format test install download upload docker documentation data clean build paper clean-paper presentations

all: data test

format:
	black . -l 79

test:
	pytest

install:
	pip install policyengine-us
	pip install -e ".[dev]"  --config-settings editable_mode=compat

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 1.0.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine-us-data --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml

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
	timeout 10 myst build --html || true
	cd docs && test -d _build/html && touch _build/html/.nojekyll || true

documentation-build:
	cd docs && \
	rm -rf _build .jupyter_cache && \
	rm -f _toc.yml && \
	myst clean && \
	myst build --html

documentation-serve:
	cd docs/_build/html && python3 -m http.server 8080

documentation-dev:
	cd docs && \
	rm -rf _build .jupyter_cache && \
	rm -f _toc.yml && \
	myst clean && \
	myst start

database:
	python policyengine_us_data/db/create_database_tables.py
	python policyengine_us_data/db/create_initial_strata.py
	python policyengine_us_data/db/etl_national_targets.py
	python policyengine_us_data/db/etl_age.py
	python policyengine_us_data/db/etl_medicaid.py
	python policyengine_us_data/db/etl_snap.py
	python policyengine_us_data/db/etl_irs_soi.py
	python policyengine_us_data/db/validate_database.py

data:
	python policyengine_us_data/utils/uprating.py
	python policyengine_us_data/datasets/acs/acs.py
	python policyengine_us_data/datasets/cps/cps.py
	python policyengine_us_data/datasets/puf/irs_puf.py
	python policyengine_us_data/datasets/puf/puf.py
	python policyengine_us_data/datasets/cps/extended_cps.py
	python policyengine_us_data/datasets/cps/enhanced_cps.py
	python policyengine_us_data/datasets/cps/small_enhanced_cps.py
	mv policyengine_us_data/storage/enhanced_cps_2024.h5 policyengine_us_data/storage/dense_enhanced_cps_2024.h5
	cp policyengine_us_data/storage/sparse_enhanced_cps_2024.h5 policyengine_us_data/storage/enhanced_cps_2024.h5

data-geo: data
	GEO_STACKING=true python policyengine_us_data/datasets/cps/cps.py
	GEO_STACKING=true python policyengine_us_data/datasets/puf/puf.py
	GEO_STACKING_MODE=true python policyengine_us_data/datasets/cps/extended_cps.py
	python policyengine_us_data/datasets/cps/geo_stacking_calibration/create_stratified_cps.py 10000

calibration-package: data-geo
	python policyengine_us_data/datasets/cps/geo_stacking_calibration/create_calibration_package.py \
		--db-path policyengine_us_data/storage/policy_data.db \
		--dataset-uri policyengine_us_data/storage/stratified_extended_cps_2023.h5 \
		--mode Stratified \
		--local-output policyengine_us_data/storage/calibration

optimize-weights-local: calibration-package
	python policyengine_us_data/datasets/cps/geo_stacking_calibration/optimize_weights.py \
		--input-dir policyengine_us_data/storage/calibration \
		--output-dir policyengine_us_data/storage/calibration \
		--total-epochs 100 \
		--device cpu

create-state-files: optimize-weights-local
	python -m policyengine_us_data.datasets.cps.geo_stacking_calibration.create_sparse_cd_stacked \
		--weights-path policyengine_us_data/storage/calibration/w_cd.npy \
		--dataset-path policyengine_us_data/storage/stratified_extended_cps_2023.h5 \
		--db-path policyengine_us_data/storage/policy_data.db \
		--output-dir policyengine_us_data/storage/cd_states

upload-calibration-package: calibration-package
	$(eval GCS_DATE := $(shell date +%Y-%m-%d-%H%M))  # For bash: GCS_DATE=$$(date +%Y-%m-%d-%H%M)
	python policyengine_us_data/datasets/cps/geo_stacking_calibration/create_calibration_package.py \
		--db-path policyengine_us_data/storage/policy_data.db \
		--dataset-uri policyengine_us_data/storage/stratified_extended_cps_2023.h5 \
		--mode Stratified \
		--gcs-bucket policyengine-calibration \
		--gcs-date $(GCS_DATE)
	@echo ""
	@echo "Calibration package uploaded to GCS"
	@echo "Date prefix: $(GCS_DATE)"
	@echo ""
	@echo "To submit GCP batch job, update batch_pipeline/config.env:"
	@echo "  INPUT_PATH=$(GCS_DATE)/inputs"
	@echo "  OUTPUT_PATH=$(GCS_DATE)/outputs"

optimize-weights-gcp:
	@echo "Submitting Cloud Batch job for weight optimization..."
	@echo "Make sure you've run 'make upload-calibration-package' first"
	@echo "and updated batch_pipeline/config.env with the correct paths"
	@echo ""
	cd policyengine_us_data/datasets/cps/geo_stacking_calibration/batch_pipeline && ./submit_batch_job.sh

download-weights-from-gcs:
	@echo "Downloading weights from GCS..."
	rm -f policyengine_us_data/storage/calibration/w_cd.npy
	@read -p "Enter GCS date prefix (e.g., 2025-10-22-1630): " gcs_date; \
	gsutil ls gs://policyengine-calibration/$$gcs_date/outputs/**/w_cd.npy | head -1 | xargs -I {} gsutil cp {} policyengine_us_data/storage/calibration/w_cd.npy && \
	gsutil ls gs://policyengine-calibration/$$gcs_date/outputs/**/w_cd_*.npy | xargs -I {} gsutil cp {} policyengine_us_data/storage/calibration/ && \
	echo "Weights downloaded successfully"

upload-state-files-to-gcs:
	@echo "Uploading state files to GCS..."
	@read -p "Enter GCS date prefix (e.g., 2025-10-22-1721): " gcs_date; \
	gsutil -m cp policyengine_us_data/storage/cd_states/*.h5 gs://policyengine-calibration/$$gcs_date/state_files/ && \
	gsutil -m cp policyengine_us_data/storage/cd_states/*_household_mapping.csv gs://policyengine-calibration/$$gcs_date/state_files/ && \
	echo "" && \
	echo "State files uploaded to gs://policyengine-calibration/$$gcs_date/state_files/"

clean:
	rm -f policyengine_us_data/storage/*.h5
	rm -f policyengine_us_data/storage/*.db
	rm -f policyengine_us_data/storage/*.pkl
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
