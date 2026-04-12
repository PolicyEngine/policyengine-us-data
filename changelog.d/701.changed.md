Add SSTB QBI split inputs to `policyengine-us-data` by exposing
`sstb_self_employment_income`, `sstb_w2_wages_from_qualified_business`, and
`sstb_unadjusted_basis_qualified_property` from the existing PUF/calibration
pipeline. The current split follows the legacy all-or-nothing
`business_is_sstb` flag, so mixed SSTB/non-SSTB allocations remain approximate
until more granular source data or imputation is added.
