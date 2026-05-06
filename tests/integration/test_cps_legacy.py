"""Legacy CPS artifact checks that are not part of current Stage 1 output."""

import numpy as np
import pytest


def test_cps_2022_has_net_worth():
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import CPS_2022

    if not CPS_2022.file_path.exists():
        pytest.skip("cps_2022_v1_6_1.h5 not built locally")

    sim = Microsimulation(dataset=CPS_2022)
    net_worth_target = 160e12
    relative_tolerance = 0.25
    np.random.seed(42)

    assert (
        abs(sim.calculate("net_worth").sum() / net_worth_target - 1)
        < relative_tolerance
    )
