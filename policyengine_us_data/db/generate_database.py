import io

import pandas as pd
from sqlalchemy import create_engine

from policyengine_us_data.db.data_models import Base

def main():

    #DATABASE_URL = "sqlite:///policy_data.db"
    #engine = create_engine(DATABASE_URL)
    #
    #print("Creating initial tables...")
    #Base.metadata.create_all(engine)
    print("Tables created.")

if __name__ == "__main__":
    main()
