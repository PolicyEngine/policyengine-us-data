Require `policyengine-us>=1.637.0`, which ships the SSTB QBI split inputs
and formulas natively, and remove the in-package compat shim that
backfilled those variables against older `policyengine-us` releases.
