"""
Run this against your real josaa_data.csv to find the exact
whitespace variant causing the IIT/GFTI + duplicate-prediction bug.
Usage: python diagnose_institute_names.py /path/to/josaa_data.csv
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else 'data/josaa_data.csv'
df = pd.read_csv(path)

# Look at every raw variant of institute names containing "Bombay"
# (swap "Bombay" for "Delhi"/"Madras" etc. to check those too)
variants = df.loc[df['Institute'].str.contains('Bombay', na=False), 'Institute'].unique()

print(f"Found {len(variants)} distinct raw string(s) for institutes containing 'Bombay':\n")
for v in variants:
    print(repr(v))   # repr() exposes double spaces, \xa0, trailing spaces, etc.