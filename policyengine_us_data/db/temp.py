# ucgid_str converts the household’s ucgid enumeration into a comma‑separated string of all hierarchical UCGID codes.
from policyengine_us import Simulation
from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import (
    UCGID,
)

# Minimal one-household simulation
sim = Simulation(
    situation={
        "people": {"p1": {}},
        "households": {"h1": {"members": ["p1"]}},
    }
)

# Assign a specific UCGID (California district 23)
sim.set_input("ucgid", 2024, UCGID.CA_23)

# Use the ucgid_str Variable's formula
ucgid_str_val = sim.calculate("ucgid_str", 2024)
print(ucgid_str_val)
# ['5001800US0623,0400000US06,0100000US']


# First, let's explore UCGID, the enum, and how it can create the hierarchy

import pandas as pd
from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import (
    UCGID,
)

rows = []
for node in UCGID:
    codes = node.get_hierarchical_codes()
    rows.append(
        {
            "name": node.name,
            "code": codes[0],
            "parent": codes[1] if len(codes) > 1 else None,
        }
    )

hierarchy_df = (
    pd.DataFrame(rows)
    .sort_values(["parent", "code"], na_position="first")
    .reset_index(drop=True)
)

print(hierarchy_df)
# Out[262]:
#      name           code       parent
# 0       US      0100000US         None
# 1       AL    0400000US01    0100000US
# 2       AK    0400000US02    0100000US
# 3       AZ    0400000US04    0100000US
# 4       AR    0400000US05    0100000US
# ..     ...            ...          ...
# 483  WI_05  5001800US5505  0400000US55
# 484  WI_06  5001800US5506  0400000US55
# 485  WI_07  5001800US5507  0400000US55
# 486  WI_08  5001800US5508  0400000US55
# 487  WY_01  5001800US5600  0400000US56
#
# [488 rows x 3 columns]
