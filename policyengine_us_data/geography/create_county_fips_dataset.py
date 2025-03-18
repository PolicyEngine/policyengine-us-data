import pandas as pd

def create_county_fips_dataset():
  # Map county FIPS codes to county names and states
  # Downloaded from https://www.census.gov/library/reference/code-lists/ansi.html#cou

  # STATE - 2-digit state postal code (e.g., "AL")
  # STATEFP - State FIPS code (01 for AL)
  # COUNTYFP - Three-digit county portion of FIPS (001 for Autauga County, AL, if STATEFP is 01)
  # COUNTYNAME - County name
  county_fips = pd.read_csv(
      "county_fips.txt",
      delimiter="|",
      usecols=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"],
      dtype={
        "STATE": str, 
        "STATEFP": str, 
        "COUNTYFP": str, 
        "COUNTYNAME": str
      }
  )

  county_fips = county_fips.rename(columns={
      "STATE": "state", 
      "STATEFP": "state_fips_segment", 
      "COUNTYFP": "county_fips_segment", 
      "COUNTYNAME": "county_name"
  })


  # Create composite county FIPS code, then drop segment columns; 
  # note that the FIPS code is a 5-char str of digits
  county_fips["county_fips"] = county_fips["state_fips_segment"] + county_fips["county_fips_segment"]
  county_fips.drop(columns=["state_fips_segment", "county_fips_segment"], inplace=True)
  print(county_fips.head(8))

  county_fips.to_csv("county_fips.csv.gz", compression="gzip")

if __name__ == "__main__": 
  create_county_fips_dataset()