# src/data_validation.py 
import pandas as pd 
import pandera.pandas as pa 

# ---------------------------- 
# Load CSV 
# ---------------------------- 
df = pd.read_csv("data/bank.csv") 

# ---------------------------- 
# Convert columns to proper types 
# ---------------------------- 

# Numeric columns
for col in ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Binary / categorical encoded fields
df["credit_card"] = pd.to_numeric(df["credit_card"], errors="coerce").fillna(0)
df["active_member"] = pd.to_numeric(df["active_member"], errors="coerce").fillna(0)

# Target conversion
df["churn"] = pd.to_numeric(df["churn"], errors="coerce").fillna(0).astype(int)

# ---------------------------- 
# Define schema 
# ---------------------------- 
schema = pa.DataFrameSchema({
    "credit_score": pa.Column(int, checks=pa.Check.in_range(300, 900)),
    "age": pa.Column(int, checks=pa.Check.ge(18)),
    "tenure": pa.Column(int, checks=pa.Check.ge(0)),
    "balance": pa.Column(float, checks=pa.Check.ge(0)),
    "products_number": pa.Column(int, checks=pa.Check.ge(1)),
    "estimated_salary": pa.Column(float, checks=pa.Check.ge(0)),

    "credit_card": pa.Column(int, checks=pa.Check.isin([0, 1])),
    "active_member": pa.Column(int, checks=pa.Check.isin([0, 1])),

    "churn": pa.Column(int, checks=pa.Check.isin([0, 1]))
})

# ---------------------------- 
# Validate 
# ---------------------------- 
validated_df = schema.validate(df) 

print("✅ Data validation passed!")