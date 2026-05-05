`build_release_manifest` now accepts an
`additional_compatible_specifiers` parameter that extends the
`compatible_model_packages` list with arbitrary PEP 440 specifiers
(e.g. `">=1.637.0,<2.0.0"`). Use this when the data build fingerprint
is known to be stable across a range of `policyengine-us` versions so
downstream consumers do not have to regenerate the dataset for every
model patch release.
