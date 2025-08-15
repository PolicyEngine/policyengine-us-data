# storage/ datasets 

- **aca_spending_and_enrollment_2024.csv**  
  • Source: CMS Marketplace Public Use File, 2024 open-enrollment  
  • Date: 2024  
  • Location: https://www.cms.gov/files/document/health-insurance-exchanges-2024-open-enrollment-report-final.pdf

- **medicaid_enrollment_2024.csv**  
  • Source: MACPAC Enrollment Tables, FFY 2024  
  • Date: 2024  
  • Location: https://www.medicaid.gov/resources-for-states/downloads/eligib-oper-and-enrol-snap-december2024.pdf#page=26

- **district_mapping.csv**  
  • Source: created by the script `policyengine_us/storage/calibration_targets/make_district_mapping.py`
  • Notes: this script is not part of `make data` because of the length of time it takes to run and the
    likelhood of timeout errors. See the script for more notes, including an alternative source. Also,
    once the IRS SOI updates their data in 2026, this mapping will likely be unncessesary. 
