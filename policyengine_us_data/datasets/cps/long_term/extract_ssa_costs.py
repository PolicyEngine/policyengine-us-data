import pandas as pd
import numpy as np

# Read the file
df = pd.read_excel('SingleYearTRTables_TR2025.xlsx', sheet_name='VI.G9', header=None)

print("DataFrame shape:", df.shape)
print("\nChecking data types around row 66-70:")
for i in range(66, 71):
    year_val = df.iloc[i, 0]
    cost_val = df.iloc[i, 2]
    print(f"Row {i}: Year={year_val} (type: {type(year_val)}), Cost={cost_val} (type: {type(cost_val)})")

# Extract OASDI costs more carefully
oasdi_costs_2025_dollars = {}
for i in range(66, min(142, len(df))):
    year_val = df.iloc[i, 0]
    cost_val = df.iloc[i, 2]

    if pd.notna(year_val) and pd.notna(cost_val):
        try:
            year = int(year_val)
            cost = float(cost_val)
            oasdi_costs_2025_dollars[year] = cost
            if year <= 2030:
                print(f"Extracted: {year} -> ${cost}B")
        except Exception as e:
            print(f"Error at row {i}: {e}")
            break

print(f"\nTotal years extracted: {len(oasdi_costs_2025_dollars)}")

# Show the dictionary
print("\nFirst 10 years:")
for year in sorted(oasdi_costs_2025_dollars.keys())[:10]:
    print(f"  {year}: ${oasdi_costs_2025_dollars[year]:.1f}B")