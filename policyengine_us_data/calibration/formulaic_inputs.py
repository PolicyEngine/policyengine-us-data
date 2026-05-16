"""SPM formula/output aggregates that must not be persisted as leaf inputs."""

FORMULAIC_SPM_INPUTS_TO_DROP = frozenset(
    {
        "person_in_poverty",
        "in_poverty",
        "in_deep_poverty",
        "spm_unit_is_in_spm_poverty",
        "spm_unit_is_in_deep_spm_poverty",
        "spm_unit_spm_threshold",
        "spm_unit_geographic_adjustment",
        "spm_unit_total_income_reported",
        "spm_unit_net_income_reported",
    }
)


def drop_formulaic_spm_inputs(variable_names: set[str]) -> None:
    """Remove SPM formula/output aggregates from a mutable variable-name set."""

    variable_names.difference_update(FORMULAIC_SPM_INPUTS_TO_DROP)
