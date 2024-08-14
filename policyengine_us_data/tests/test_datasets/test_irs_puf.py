import pytest


@pytest.mark.parametrize("year", [2015])
def test_irs_puf_generates(year: int):
    from policyengine_us_data.irs_puf import IRS_PUF_2015

    dataset_by_year = {
        2015: IRS_PUF_2015,
    }

    dataset_by_year[year](require=True)