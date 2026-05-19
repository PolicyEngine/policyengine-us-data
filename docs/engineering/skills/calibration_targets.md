# Calibration Target Changes

Use this workflow when adding, changing, or reviewing manually sourced
calibration targets.

## Dual Registration

New targets must be registered in both active target systems:

- `policyengine_us_data/utils/loss.py` for the ECPS `build_loss_matrix()` path.
- The appropriate `policyengine_us_data/db/etl_*.py` loader for
  `policy_data.db`, local H5 outputs, and validation inputs. National targets
  usually belong in `etl_national_targets.py`; state and local targets should
  use a state/local ETL module and must still be present in this DB path.

If the default calibration path uses `policyengine_us_data/calibration/target_config.yaml`
with an `include:` list, also add the matching include rule there. A target can
exist in `policy_data.db` and still be ignored by calibration if it is missing
from `target_config.yaml`.

## Tests

Every target change should add or update tests that prove the target is wired
through every active path. For manually sourced targets, cover:

- the ECPS loss matrix registration in `tests/unit/calibration/test_loss_targets.py`;
- the DB ETL row in the matching `tests/unit/test_etl_*.py` file;
- the default calibration include rule in
  `tests/unit/calibration/test_target_config.py`;
- any publication guard in `tests/unit/test_upload_completed_datasets.py` when
  a missing target would make a released dataset materially wrong.

Do not use a successful DB ETL test as a substitute for a calibration-selection
test.
