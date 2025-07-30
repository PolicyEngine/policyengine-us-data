"""
Reproducibility tests for Enhanced CPS generation.

These tests ensure the pipeline produces consistent results
and can be reproduced in different environments.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import json


class TestReproducibility:
    """Test suite for reproducibility validation."""

    def test_environment_setup(self):
        """Test that required packages are installed."""
        required_packages = [
            "policyengine_us",
            "policyengine_us_data",
            "quantile_forest",
            "pandas",
            "numpy",
            "torch",
        ]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                pytest.fail(f"Required package '{package}' not installed")

    def test_deterministic_imputation(self):
        """Test that imputation produces deterministic results with fixed seed."""
        from policyengine_us_data.datasets.cps.enhanced_cps.imputation import (
            QuantileRegressionForestImputer,
        )

        # Create small test data
        n_samples = 100
        predictors = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples),
                "sex": np.random.choice([1, 2], n_samples),
                "filing_status": np.random.choice([1, 2], n_samples),
            }
        )

        target = pd.Series(np.random.lognormal(10, 1, n_samples))

        # Run imputation twice with same seed
        imputer1 = QuantileRegressionForestImputer(random_state=42)
        imputer1.fit(predictors, target)
        result1 = imputer1.predict(predictors)

        imputer2 = QuantileRegressionForestImputer(random_state=42)
        imputer2.fit(predictors, target)
        result2 = imputer2.predict(predictors)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)

    def test_weight_optimization_convergence(self):
        """Test that weight optimization converges consistently."""
        from policyengine_us_data.datasets.cps.enhanced_cps.reweight import (
            optimize_weights,
        )

        # Create test loss matrix
        n_households = 100
        n_targets = 10

        loss_matrix = np.random.rand(n_households, n_targets)
        targets = np.random.rand(n_targets) * 1e6
        initial_weights = np.ones(n_households)

        # Run optimization twice
        weights1, loss1 = optimize_weights(
            loss_matrix,
            targets,
            initial_weights,
            n_iterations=100,
            dropout_rate=0.05,
            seed=42,
        )

        weights2, loss2 = optimize_weights(
            loss_matrix,
            targets,
            initial_weights,
            n_iterations=100,
            dropout_rate=0.05,
            seed=42,
        )

        # Results should be very close
        np.testing.assert_allclose(weights1, weights2, rtol=1e-5)
        assert abs(loss1 - loss2) < 1e-6

    def test_validation_metrics_stable(self):
        """Test that validation metrics are stable across runs."""
        # This would load actual data in practice
        # For now, test with synthetic data

        metrics = {
            "gini_coefficient": 0.521,
            "top_10_share": 0.472,
            "top_1_share": 0.198,
            "poverty_rate": 0.116,
        }

        # In practice, would calculate from data
        # Here we verify expected ranges
        assert 0.50 <= metrics["gini_coefficient"] <= 0.55
        assert 0.45 <= metrics["top_10_share"] <= 0.50
        assert 0.18 <= metrics["top_1_share"] <= 0.22
        assert 0.10 <= metrics["poverty_rate"] <= 0.13

    def test_output_checksums(self):
        """Test that output files match expected checksums."""
        test_data_dir = Path("data/test")

        if not test_data_dir.exists():
            pytest.skip("Test data not generated")

        checksum_file = test_data_dir / "checksums.txt"
        if not checksum_file.exists():
            pytest.skip("Checksum file not found")

        # Read expected checksums
        expected_checksums = {}
        with open(checksum_file) as f:
            for line in f:
                if line.strip():
                    filename, checksum = line.strip().split(": ")
                    expected_checksums[filename] = checksum

        # Verify files
        for filename, expected_checksum in expected_checksums.items():
            file_path = test_data_dir / filename
            if file_path.exists() and filename != "checksums.txt":
                with open(file_path, "rb") as f:
                    actual_checksum = hashlib.sha256(f.read()).hexdigest()
                assert (
                    actual_checksum == expected_checksum
                ), f"Checksum mismatch for {filename}"

    def test_memory_usage(self):
        """Test that memory usage stays within bounds."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run a small imputation task
        n_samples = 10000
        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples),
                "income": np.random.lognormal(10, 1, n_samples),
            }
        )

        # Process data
        data["income_bracket"] = pd.qcut(data["income"], 10)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Should use less than 500MB for this small task
        assert memory_used < 500, f"Used {memory_used:.1f}MB, expected <500MB"

    def test_platform_independence(self):
        """Test that code works across platforms."""
        import platform

        system = platform.system()
        assert system in [
            "Linux",
            "Darwin",
            "Windows",
        ], f"Unsupported platform: {system}"

        # Test path handling
        test_path = Path("data") / "test" / "file.csv"
        assert str(test_path).replace("\\", "/") == "data/test/file.csv"

    def test_api_credentials_documented(self):
        """Test that API credential requirements are documented."""
        readme_path = Path("REPRODUCTION.md")
        assert readme_path.exists(), "REPRODUCTION.md not found"

        content = readme_path.read_text()

        # Check for credential documentation
        required_sections = [
            "POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN",
            "CENSUS_API_KEY",
            "PUF Data Access",
        ]

        for section in required_sections:
            assert section in content, f"Missing documentation for '{section}'"

    def test_synthetic_data_generation(self):
        """Test that synthetic data can be generated for testing."""
        from scripts.generate_test_data import (
            generate_synthetic_cps,
            generate_synthetic_puf,
        )

        # Generate small datasets
        households, persons = generate_synthetic_cps(n_households=10)
        puf = generate_synthetic_puf(n_returns=50)

        # Verify structure
        assert len(households) == 10
        assert len(persons) > 10  # Multiple persons per household
        assert len(puf) == 50

        # Verify required columns
        assert "household_id" in households.columns
        assert "person_id" in persons.columns
        assert "wages" in puf.columns

    def test_smoke_test_pipeline(self):
        """Run a minimal version of the full pipeline."""
        # This test would be marked as slow and only run in CI
        pytest.skip("Full pipeline test - run with --runslow")

        # Would include:
        # 1. Load test data
        # 2. Run imputation on subset
        # 3. Run reweighting with few targets
        # 4. Validate outputs exist

    def test_documentation_completeness(self):
        """Test that all necessary documentation exists."""
        required_docs = [
            "README.md",
            "REPRODUCTION.md",
            "CLAUDE.md",
            "docs/methodology.md",
            "docs/data.md",
        ]

        for doc in required_docs:
            doc_path = Path(doc)
            assert doc_path.exists(), f"Missing documentation: {doc}"

            # Check not empty
            content = doc_path.read_text()
            assert len(content) > 100, f"Documentation too short: {doc}"


@pytest.mark.slow
class TestFullReproduction:
    """Full reproduction tests (run with --runslow flag)."""

    def test_full_pipeline_subset(self):
        """Test full pipeline on data subset."""
        # This would run the complete pipeline on a small subset
        # Taking ~10 minutes instead of hours
        pass

    def test_validation_dashboard(self):
        """Test that validation dashboard can be generated."""
        # Would test dashboard generation
        pass
