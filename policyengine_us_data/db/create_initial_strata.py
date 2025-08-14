from typing import Dict

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, Session, select


from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import UCGID
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
)



def main():
    # Get the implied hierarchy by the UCGID enum -------- 
    rows = []
    for node in UCGID:
        codes = node.get_hierarchical_codes()
        rows.append({
            "name":   node.name,
            "code":   codes[0],
            "parent": codes[1] if len(codes) > 1 else None
        })
    
    hierarchy_df = (
        pd.DataFrame(rows)
          .sort_values(["parent", "code"], na_position="first")
          .reset_index(drop=True)
    )
    

    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    session = Session()

    # map the ucgid_str 'code' to auto-generated 'stratum_id'
    code_to_stratum_id: Dict[str, int] = {}
    
    for _, row in hierarchy_df.iterrows():
        parent_code = row["parent"]
        
        parent_id = code_to_stratum_id.get(parent_code) if parent_code else None

        new_stratum = Stratum(
            parent_stratum_id=parent_id,
            notes=f'{row["name"]} (ucgid {row["code"]})',
            stratum_group_id=1,
        )

        new_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="in",
                value=row["code"],
            )
        ]
        
        session.add(new_stratum)
        
        session.flush()
        
        code_to_stratum_id[row["code"]] = new_stratum.stratum_id

    session.commit()

if __name__ == "__main__":
    main()
