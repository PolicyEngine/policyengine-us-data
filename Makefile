format:
	black . -l 79

test:
	pytest policyengine_us_data/tests

install:
	pip install -e .[dev]

docker:
	docker buildx build --platform linux/amd64 . -t policyengine-us-data:latest

evaluate:
	python policyengine_us_data/evaluation/summary.py