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
	python policyengine_us_data/db/create_database_tables.py
	python policyengine_us_data/db/create_initial_strata.py
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

data-local-area: data
	LOCAL_AREA_CALIBRATION=true python policyengine_us_data/datasets/cps/cps.py
	LOCAL_AREA_CALIBRATION=true python policyengine_us_data/datasets/puf/puf.py
	LOCAL_AREA_CALIBRATION=true python policyengine_us_data/datasets/cps/extended_cps.py
	python policyengine_us_data/datasets/cps/local_area_calibration/create_stratified_cps.py 10500

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
