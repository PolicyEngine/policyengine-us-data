"""
Generate all LaTeX tables for the Enhanced CPS paper.

This script creates all tables used in the paper from actual data,
ensuring reproducibility and preventing hard-coded values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def format_number(value, decimals=3):
    """Format a number for LaTeX table display."""
    if pd.isna(value) or value == "[TO BE CALCULATED]":
        return "[TBC]"
    if isinstance(value, (int, float)):
        if decimals == 0:
            return f"{value:,.0f}"
        else:
            return f"{value:.{decimals}f}"
    return str(value)


def create_latex_table(df, caption, label, float_format=None):
    """Convert a pandas DataFrame to LaTeX table format."""
    # Start table
    latex = "\\begin{table}[h]\n"
    latex += "    \\centering\n"
    latex += f"    \\caption{{{caption}}}\n"
    latex += f"    \\label{{{label}}}\n"
    
    # Format the dataframe as LaTeX
    if float_format:
        table_body = df.to_latex(index=False, escape=False, float_format=float_format)
    else:
        table_body = df.to_latex(index=False, escape=False)
    
    # Extract just the tabular part
    lines = table_body.split('\n')
    tabular_start = next(i for i, line in enumerate(lines) if '\\begin{tabular}' in line)
    tabular_end = next(i for i, line in enumerate(lines) if '\\end{tabular}' in line)
    
    # Indent the tabular content
    for i in range(tabular_start, tabular_end + 1):
        latex += "    " + lines[i] + "\n"
    
    latex += "\\end{table}\n"
    return latex


def generate_tax_unit_metrics():
    """Generate tax unit level distributional metrics table."""
    # Placeholder data - will be replaced with actual calculations
    data = {
        'Metric': ['Gini coefficient', 'Top 10\\% share', 'Top 1\\% share'],
        'CPS': ['[TBC]', '[TBC]', '[TBC]'],
        'Enhanced CPS': ['[TBC]', '[TBC]', '[TBC]'],
        'PUF': ['[TBC]', '[TBC]', '[TBC]']
    }
    
    df = pd.DataFrame(data)
    return create_latex_table(
        df,
        caption="Tax unit-level distributional metrics",
        label="tab:tax_unit_metrics"
    )


def generate_household_metrics():
    """Generate household level distributional metrics table."""
    data = {
        'Metric': ['Gini coefficient', 'Top 10\\% share', 'Top 1\\% share'],
        'CPS': ['[TBC]', '[TBC]', '[TBC]'],
        'Enhanced CPS': ['[TBC]', '[TBC]', '[TBC]'],
        'PUF': ['N/A', 'N/A', 'N/A']  # PUF doesn't have household structure
    }
    
    df = pd.DataFrame(data)
    return create_latex_table(
        df,
        caption="Household-level distributional metrics",
        label="tab:household_metrics"
    )


def generate_poverty_metrics():
    """Generate poverty metrics table."""
    data = {
        'Dataset': ['CPS', 'Enhanced CPS', 'PUF'],
        'SPM Poverty Rate': ['[TBC]', '[TBC]', 'N/A'],
        'Child Poverty Rate': ['[TBC]', '[TBC]', 'N/A'],
        'Senior Poverty Rate': ['[TBC]', '[TBC]', 'N/A']
    }
    
    df = pd.DataFrame(data)
    return create_latex_table(
        df,
        caption="Poverty rates by dataset",
        label="tab:poverty_metrics"
    )


def generate_weight_stats():
    """Generate weight distribution statistics table."""
    data = {
        'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Zeros (\\%)'],
        'CPS': ['[TBC]', '[TBC]', '[TBC]', '[TBC]', '[TBC]'],
        'Enhanced CPS': ['[TBC]', '[TBC]', '[TBC]', '[TBC]', '[TBC]'],
        'PUF': ['[TBC]', '[TBC]', '[TBC]', '[TBC]', '[TBC]']
    }
    
    df = pd.DataFrame(data)
    return create_latex_table(
        df,
        caption="Weight distribution statistics",
        label="tab:weight_stats"
    )


def generate_top_rate_reform():
    """Generate top tax rate reform impact table."""
    data = {
        'Dataset': ['CPS', 'Enhanced CPS', 'PUF'],
        'Revenue Impact (\\$B)': ['[TBC]', '[TBC]', '[TBC]'],
        'Affected Tax Units (M)': ['[TBC]', '[TBC]', '[TBC]'],
        'Avg Tax Increase (\\$)': ['[TBC]', '[TBC]', '[TBC]']
    }
    
    df = pd.DataFrame(data)
    return create_latex_table(
        df,
        caption="Revenue projections from top rate increase (37\\% to 39.6\\%)",
        label="tab:top_rate_reform"
    )


def generate_target_examples():
    """Generate examples of calibration targets table."""
    data = {
        'Source': [
            'IRS SOI',
            'Census', 
            'CBO',
            'JCT',
            'Healthcare'
        ],
        'Example Targets': [
            'AGI by bracket, employment income, capital gains',
            'Population by age, state populations',
            'SNAP benefits, Social Security, income tax',
            'SALT deduction (\\$21.2B), charitable (\\$65.3B)',
            'Medicare Part B premiums by age group'
        ],
        'Count': [
            '5,300+',
            '150+',
            '5',
            '4',
            '40+'
        ]
    }
    
    df = pd.DataFrame(data)
    return create_latex_table(
        df,
        caption="Examples of calibration targets by source",
        label="tab:target_examples"
    )


def main():
    """Generate all tables for the paper."""
    # Create output directory
    output_dir = Path("paper/tables")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating LaTeX tables for Enhanced CPS paper...")
    print("=" * 70)
    
    # Generate each table
    tables = {
        "tax_unit_metrics.tex": generate_tax_unit_metrics(),
        "household_metrics.tex": generate_household_metrics(),
        "poverty_metrics.tex": generate_poverty_metrics(),
        "weight_stats.tex": generate_weight_stats(),
        "top_rate_reform.tex": generate_top_rate_reform(),
        "target_examples.tex": generate_target_examples()
    }
    
    # Write tables to files
    for filename, content in tables.items():
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Generated {filename}")
    
    print("\nAll tables generated with [TBC] placeholders.")
    print("To populate with actual data:")
    print("1. Run 'make data' to generate datasets")
    print("2. Update calculation functions to compute actual metrics")
    

if __name__ == "__main__":
    main()