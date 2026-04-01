from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd

from policyengine_core.data import Dataset

from policyengine_us_data.datasets.scf.fed_scf import SummarizedFedSCF


def test_fed_scf_load_waits_for_concurrent_generation(tmp_path):
    class TestFedSCF(SummarizedFedSCF):
        time_period = 2099
        name = "test_fed_scf"
        label = "Test Federal Reserve SCF"
        file_path = tmp_path / "test_fed_scf.h5"
        data_format = Dataset.TABLES

    generation_calls = 0
    expected = pd.DataFrame({"value": [1, 2, 3]})

    def fake_generate(self):
        nonlocal generation_calls
        generation_calls += 1
        with pd.HDFStore(self.file_path, mode="w") as storage:
            storage["data"] = expected
            time.sleep(0.2)

    first = TestFedSCF()
    second = TestFedSCF()
    first._generate_unlocked = fake_generate.__get__(first, TestFedSCF)
    second._generate_unlocked = fake_generate.__get__(second, TestFedSCF)

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(first.load)
        right = executor.submit(second.load)
        left_result = left.result()
        right_result = right.result()

    assert generation_calls == 1
    pd.testing.assert_frame_equal(left_result, expected)
    pd.testing.assert_frame_equal(right_result, expected)
