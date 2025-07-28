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

targets:
	python policyengine_us_data/storage/calibration_targets/pull_hardcoded_targets.py
	python policyengine_us_data/storage/calibration_targets/pull_age_targets.py
	python policyengine_us_data/storage/calibration_targets/pull_soi_targets.py
	python policyengine_us_data/storage/calibration_targets/pull_snap_targets.py

upload:
	python policyengine_us_data/storage/upload_completed_datasets.py

docker:
	docker buildx build --platform linux/amd64 . -t policyengine-us-data:latest
	
documentation:
	jb clean docs && jb build docs
	python docs/add_plotly_to_book.py docs


data:
	python policyengine_us_data/utils/uprating.py
	python policyengine_us_data/datasets/acs/acs.py
	python policyengine_us_data/datasets/cps/cps.py
	python policyengine_us_data/datasets/puf/irs_puf.py
	python policyengine_us_data/datasets/puf/puf.py
	python policyengine_us_data/datasets/cps/extended_cps.py
	python policyengine_us_data/datasets/cps/enhanced_cps.py
	python policyengine_us_data/datasets/cps/small_enhanced_cps.py

clean:
	rm -f policyengine_us_data/storage/*.h5
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
