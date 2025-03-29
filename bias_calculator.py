import pandas as pd
import os

# ===== Step 1: Load Dataset =====
df = pd.read_csv('data/bigdata.csv')

print("✅ Dataset loaded.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ===== Step 2: Create loan_approved column =====
df['loan_approved'] = df['action_taken'].apply(lambda x: 1 if x == 1 else 0)

# ===== Step 3: Approval Rate by Race =====
print("\n📊 Approval Rate by Race:")
approval_by_race = df.groupby('derived_race')['loan_approved'].mean().sort_values(ascending=False)
print(approval_by_race)

# ===== Step 4: Approval Rate by Gender =====
print("\n📊 Approval Rate by Gender:")
approval_by_gender = df.groupby('derived_sex')['loan_approved'].mean().sort_values(ascending=False)
print(approval_by_gender)

# ===== Step 5: Approval Rate by Income Group =====
df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
print("\n📊 Approval Rate by Income Group:")
approval_by_income = df.groupby('income_group')['loan_approved'].mean()
print(approval_by_income)

# ===== Step 6: Save Summary CSVs =====
if not os.path.exists('output'):
    os.makedirs('output')

approval_by_race.to_csv('output/approval_by_race.csv')
approval_by_gender.to_csv('output/approval_by_gender.csv')
approval_by_income.to_csv('output/approval_by_income.csv')

print("\n✅ Bias calculation done. Results saved in 'output/' folder.")
