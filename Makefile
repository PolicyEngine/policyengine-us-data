format:
	black . -l 79

test:
	pytest policyengine_us_data/tests

install:
	pip install -e .
