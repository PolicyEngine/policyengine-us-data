# IRS SOI Data Inconsistency: A59664 Units Issue

## Summary
The IRS Statistics of Income (SOI) Congressional District data file has an undocumented data inconsistency where column A59664 (EITC amount for 3+ children) is reported in **dollars** instead of **thousands of dollars** like all other monetary columns.

## Discovery Date
December 2024

## Affected Data
- **File**: https://www.irs.gov/pub/irs-soi/22incd.csv (and likely other years)
- **Column**: A59664 - "Earned income credit with three qualifying children amount"
- **Issue**: Value is in dollars, not thousands of dollars

## Evidence

### 1. Documentation States All Money in Thousands
From the IRS SOI documentation: "For all the files, the money amounts are reported in thousands of dollars."

### 2. Data Analysis Shows Inconsistency
California example from 2022 data:
```
A59661 (EITC 0 children):    284,115  (thousands) = $284M ✓
A59662 (EITC 1 child):     2,086,260  (thousands) = $2.1B ✓
A59663 (EITC 2 children):  2,067,922  (thousands) = $2.1B ✓
A59664 (EITC 3+ children): 1,248,669,042  (if thousands) = $1.25 TRILLION ✗
```

### 3. Total EITC Confirms the Issue
```
A59660 (Total EITC): 5,687,167 (thousands) = $5.69B

Sum with A59664 as dollars: $5.69B ✓ (matches!)
Sum with A59664 as thousands: $1.25T ✗ (way off!)
```

### 4. Pattern Across All States
The ratio of A59664 to A59663 is consistently ~600x across all states:
- California: 603.8x
- North Carolina: 598.9x  
- New York: 594.2x
- Texas: 691.5x

If both were in the same units, this ratio should be 0.5-2x.

## Additional Finding: "Three" Means "Three or More"

The documentation says "three qualifying children" but the data shows this represents "three or more":
- Sum of N59661 + N59662 + N59663 + N59664 = 23,261,270
- N59660 (Total EITC recipients) = 23,266,630
- Difference: 5,360 (0.02% - essentially equal)

This confirms that category 4 represents families with 3+ children, not exactly 3.

## Fix Applied

In `etl_irs_soi.py`, we now divide A59664 by 1000 before applying the standard multiplier:

```python
if amount_col == 'A59664':
    # Convert from dollars to thousands to match other columns
    rec_amounts["target_value"] /= 1_000
```

## Impact Before Fix
- EITC calibration targets for 3+ children were 1000x too high
- California target: $1.25 trillion instead of $1.25 billion
- Made calibration impossible to converge for EITC

## Verification Steps
1. Download IRS SOI data for any year
2. Check A59660 (total EITC) value
3. Sum A59661-A59664 with A59664 divided by 1000
4. Confirm sum matches A59660

## Recommendation for IRS
The IRS should either:
1. Fix the data to report A59664 in thousands like other columns
2. Document this exception clearly in their documentation

## Verification Code

To verify this issue or check if the IRS has fixed it:

```python
import pandas as pd

# Load IRS data
df = pd.read_csv('https://www.irs.gov/pub/irs-soi/22incd.csv')
us_data = df[(df['STATE'] == 'US') & (df['agi_stub'] == 0)]

# Get EITC values
a61 = us_data['A59661'].values[0] * 1000  # 0 children (convert from thousands)
a62 = us_data['A59662'].values[0] * 1000  # 1 child
a63 = us_data['A59663'].values[0] * 1000  # 2 children  
a64 = us_data['A59664'].values[0]         # 3+ children (already in dollars!)
total = us_data['A59660'].values[0] * 1000  # Total EITC

print(f'Sum with A59664 as dollars: ${(a61 + a62 + a63 + a64):,.0f}')
print(f'Total EITC (A59660):        ${total:,.0f}')
print(f'Match: {abs(total - (a61 + a62 + a63 + a64)) < 1e6}')

# Check ratio to confirm inconsistency
ratio = us_data['A59664'].values[0] / us_data['A59663'].values[0]
print(f'\nA59664/A59663 ratio: {ratio:.1f}x')
print('(Should be ~0.5-2x if same units, but is ~600x)')
```

## Related Files
- `/home/baogorek/devl/policyengine-us-data/policyengine_us_data/db/etl_irs_soi.py` - ETL script with fix and auto-detection