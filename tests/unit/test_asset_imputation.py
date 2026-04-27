import numpy as np
import pandas as pd

from policyengine_us_data.utils.asset_imputation import (
    NET_WORTH_COMPONENTS_ARE_COMPLETE,
    add_scf_financial_asset_targets,
    add_scf_net_worth_component_targets,
    aggregate_person_values_to_reference_households,
    align_household_values_to_reference_households,
    build_household_vehicle_receiver,
    check_household_net_worth_reconciliation,
    combine_sipp_and_scf_financial_assets,
    compute_net_worth_from_components,
    financial_asset_source_is_scf,
)


def test_build_household_vehicle_receiver_aggregates_person_inputs():
    person_df = pd.DataFrame(
        {
            "household_id": [10, 10, 20],
            "employment_income": [20_000.0, 5_000.0, 30_000.0],
            "interest_income": [100.0, 50.0, 25.0],
            "dividend_income": [10.0, 0.0, 5.0],
            "rental_income": [0.0, 200.0, 0.0],
            "age": [42, 12, 35],
            "is_female": [1.0, 1.0, 0.0],
            "is_married": [1.0, 0.0, 0.0],
            "is_household_head": [True, False, True],
        }
    )

    receiver = build_household_vehicle_receiver(
        person_df,
        tenure_type=np.array([b"OWNED_WITH_MORTGAGE", b"RENTED"]),
    )

    assert receiver["household_id"].tolist() == [10, 20]
    assert receiver["household_employment_income"].tolist() == [25_000.0, 30_000.0]
    assert receiver["household_interest_income"].tolist() == [150.0, 25.0]
    assert receiver["household_dividend_income"].tolist() == [10.0, 5.0]
    assert receiver["household_rental_income"].tolist() == [200.0, 0.0]
    assert receiver["count_under_18"].tolist() == [1.0, 0.0]
    assert receiver["household_size"].tolist() == [2.0, 1.0]
    assert receiver["reference_age"].tolist() == [42.0, 35.0]
    assert receiver["reference_is_female"].tolist() == [1.0, 0.0]
    assert receiver["reference_is_married"].tolist() == [1.0, 0.0]
    assert receiver["is_homeowner"].tolist() == [1.0, 0.0]


def test_net_worth_reconciliation_can_be_explicitly_skipped():
    data = {
        "net_worth": np.array([500_000.0]),
        "bank_account_assets": np.array([10_000.0]),
        "stock_assets": np.array([5_000.0]),
        "bond_assets": np.array([1_000.0]),
        "household_vehicles_value": np.array([15_000.0]),
        "auto_loan_balance": np.array([2_000.0]),
    }

    report = check_household_net_worth_reconciliation(
        data,
        components_are_complete=False,
    )

    assert NET_WORTH_COMPONENTS_ARE_COMPLETE is True
    assert report.components_are_complete is False
    assert report.is_reconciled is None
    assert report.max_abs_difference is None
    assert "marked incomplete" in report.message


def test_net_worth_reconciliation_checks_complete_household_components():
    data = {
        "net_worth": np.array([125.0, -10.0]),
        "bank_account_assets": np.array([100.0, 10.0]),
        "stock_assets": np.array([50.0, 0.0]),
        "auto_loan_balance": np.array([25.0, 20.0]),
    }

    report = check_household_net_worth_reconciliation(
        data,
        component_variables=(
            "bank_account_assets",
            "stock_assets",
            "auto_loan_balance",
        ),
        components_are_complete=True,
        atol=0.0,
    )

    assert report.components_are_complete is True
    assert report.is_reconciled is True
    assert report.max_abs_difference == 0.0


def test_net_worth_reconciliation_reports_complete_component_mismatch():
    data = {
        "net_worth": np.array([126.0]),
        "bank_account_assets": np.array([100.0]),
        "stock_assets": np.array([50.0]),
        "auto_loan_balance": np.array([25.0]),
    }

    report = check_household_net_worth_reconciliation(
        data,
        component_variables=(
            "bank_account_assets",
            "stock_assets",
            "auto_loan_balance",
        ),
        components_are_complete=True,
        atol=0.0,
    )

    assert report.is_reconciled is False
    assert report.max_abs_difference == 1.0


def test_add_scf_financial_asset_targets_builds_sipp_comparable_columns():
    scf = pd.DataFrame(
        {
            "liq": [100.0, 200.0],
            "stocks": [10.0, 20.0],
            "nmmf": [1.0, 2.0],
            "bond": [5.0, 6.0],
        }
    )

    targets = add_scf_financial_asset_targets(scf)

    assert targets == (
        "scf_bank_account_assets",
        "scf_stock_assets",
        "scf_bond_assets",
    )
    assert scf["scf_bank_account_assets"].tolist() == [100.0, 200.0]
    assert scf["scf_stock_assets"].tolist() == [11.0, 22.0]
    assert scf["scf_bond_assets"].tolist() == [5.0, 6.0]


def test_add_scf_net_worth_component_targets_builds_formula_columns():
    scf = pd.DataFrame(
        {
            "cds": [1.0],
            "savbnd": [1.5],
            "retqliq": [2.0],
            "cashli": [3.0],
            "othma": [4.0],
            "othfin": [5.0],
            "houses": [100.0],
            "oresre": [20.0],
            "nnresre": [30.0],
            "bus": [40.0],
            "othnfin": [6.0],
            "mrthel": [50.0],
            "resdbt": [7.0],
            "othloc": [8.0],
            "ccbal": [9.0],
            "veh_inst": [9.5],
            "edn_inst": [10.0],
            "oth_inst": [11.0],
            "odebt": [13.0],
        }
    )

    targets = add_scf_net_worth_component_targets(scf)

    assert "scf_savings_bonds" in targets
    assert "scf_retirement_assets" in targets
    assert "scf_vehicle_installment_debt" in targets
    assert "scf_mortgage_debt" in targets
    assert "scf_buy_now_pay_later_debt" not in targets
    assert scf["scf_savings_bonds"].tolist() == [1.5]
    assert scf["scf_retirement_assets"].tolist() == [2.0]
    assert scf["scf_vehicle_installment_debt"].tolist() == [9.5]
    assert scf["scf_mortgage_debt"].tolist() == [50.0]


def test_financial_asset_source_draw_is_household_stable():
    household_ids = np.array([10, 10, 20, 30])

    first = financial_asset_source_is_scf(household_ids, time_period=2024)
    second = financial_asset_source_is_scf(household_ids, time_period=2024)

    assert first.tolist() == second.tolist()
    assert first[0] == first[1]


def test_combine_sipp_and_scf_financial_assets_preserves_household_scf_total():
    person_household_ids = np.array([10, 10, 20, 20])
    reference_person_mask = np.array([True, False, True, False])
    use_scf = financial_asset_source_is_scf(
        person_household_ids,
        time_period=2024,
    )

    combined = combine_sipp_and_scf_financial_assets(
        sipp_values=np.array([1.0, 2.0, 3.0, 4.0]),
        scf_household_values=np.array([100.0, 200.0]),
        person_household_ids=person_household_ids,
        reference_person_mask=reference_person_mask,
        time_period=2024,
    )

    for household_id, scf_total in [(10, 100.0), (20, 200.0)]:
        household_mask = person_household_ids == household_id
        if use_scf[household_mask][0]:
            assert combined[household_mask].sum() == scf_total
            assert combined[household_mask & ~reference_person_mask].sum() == 0.0
        else:
            np.testing.assert_array_equal(
                combined[household_mask],
                np.array([1.0, 2.0, 3.0, 4.0])[household_mask],
            )


def test_aggregate_and_align_household_components():
    person_household_ids = np.array([20, 10, 20, 10])
    reference_person_mask = np.array([True, True, False, False])

    aggregated = aggregate_person_values_to_reference_households(
        [1.0, 2.0, 3.0, 4.0],
        person_household_ids,
        reference_person_mask,
    )
    aligned = align_household_values_to_reference_households(
        household_values=[100.0, 200.0],
        household_ids=np.array([10, 20]),
        reference_household_ids=person_household_ids[reference_person_mask],
    )

    assert aggregated.tolist() == [4.0, 6.0]
    assert aligned.tolist() == [200.0, 100.0]


def test_compute_net_worth_from_components_applies_signs():
    net_worth = compute_net_worth_from_components(
        components={
            "bank_account_assets": np.array([100.0]),
            "scf_retirement_assets": np.array([300.0]),
            "auto_loan_balance": np.array([50.0]),
            "scf_mortgage_debt": np.array([200.0]),
        },
    )

    assert net_worth.tolist() == [150.0]
