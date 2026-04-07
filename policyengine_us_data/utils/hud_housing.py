HUD_USER_HOUSING_ASSISTANCE_BENCHMARKS = {
    2022: {
        "reported_households": 4_537_614,
        "average_monthly_spending_per_unit": 899,
    },
    2023: {
        "reported_households": 4_569_973,
        "average_monthly_spending_per_unit": 989,
    },
    2024: {
        "reported_households": 4_584_691,
        "average_monthly_spending_per_unit": 1_067,
    },
    2025: {
        "reported_households": 4_519_561,
        "average_monthly_spending_per_unit": 1_135,
    },
}


def get_hud_user_housing_benchmark(year: int) -> dict:
    if year not in HUD_USER_HOUSING_ASSISTANCE_BENCHMARKS:
        raise ValueError(f"No HUD USER housing benchmark for {year}.")

    benchmark = dict(HUD_USER_HOUSING_ASSISTANCE_BENCHMARKS[year])
    benchmark["annual_spending_total"] = (
        benchmark["reported_households"]
        * benchmark["average_monthly_spending_per_unit"]
        * 12
    )
    return benchmark


def build_hud_user_housing_assistance_benchmark(year: int) -> dict:
    benchmark = get_hud_user_housing_benchmark(year)
    return {
        "variable": "housing_assistance",
        "annual_spending_total": benchmark["annual_spending_total"],
        "reported_households": benchmark["reported_households"],
        "average_monthly_spending_per_unit": benchmark[
            "average_monthly_spending_per_unit"
        ],
        "source": "HUD USER Picture of Subsidized Households",
        "notes": (
            "Annual federal spending and occupied assisted households from "
            "HUD USER; not Census SPM capped subsidy."
        ),
        "year": year,
    }
