import pytest


@pytest.mark.skip(reason="This test requires private data.")
@pytest.mark.parametrize("year", [2015])
def test_irs_puf_generates(year: int):
    from policyengine_us_data.datasets.puf.irs_puf import IRS_PUF_2015

    dataset_by_year = {
        2015: IRS_PUF_2015,
    }

    dataset_by_year[year](require=True)
