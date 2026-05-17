def target_variable_components(variable: str) -> list[str]:
    """Return component variables for a target expression.

    Calibration targets normally name a single policyengine-us variable, but
    some primary-source aggregates map to a sum of variables.  We support only
    additive expressions so target values remain linear in survey weights.
    """
    return [part.strip() for part in variable.split("+")]


def target_variable_is_valid(variable: str, valid_variables) -> bool:
    return all(
        component and component in valid_variables
        for component in target_variable_components(variable)
    )
