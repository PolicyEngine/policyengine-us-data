.PHONY: all format test install download upload docker documentation data clean build

all: data test

format:
	black . -l 79

test:
	pytest

install:
	pip install -e ".[dev]"

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 1.0.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine-us-data --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml

download:
	python policyengine_us_data/storage/download_public_prerequisites.py
	python policyengine_us_data/storage/download_private_prerequisites.py

upload:
	python policyengine_us_data/storage/upload_completed_datasets.py

docker:
	docker buildx build --platform linux/amd64 . -t policyengine-us-data:latest
	
documentation:
	jb clean docs && jb build docs

data:
	python policyengine_us_data/datasets/cps/cps.py
	python policyengine_us_data/datasets/cps/enhanced_cps.py

clean:
	rm -f policyengine_us_data/storage/puf_2015.csv
	rm -f policyengine_us_data/storage/demographics_2015.csv

build:
	python -m build

publish:
	twine upload dist/*
