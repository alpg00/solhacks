import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================
# PART 1: Basic DTI-Based Approval Calculation
# ===================================================

# Create output folder if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Step 1: Load the CSV File
try:
    # Let Pandas infer the delimiter, use the Python engine, and skip malformed lines.
    df = pd.read_csv('data/bigdata.csv', engine='python', on_bad_lines='skip', sep=None)
    print("✅ CSV loaded successfully. Shape:", df.shape)
except Exception as e:
    print("Error loading CSV:", e)
    exit(1)

# Debug: Print available columns
print("Columns in CSV:", df.columns.tolist())

# Verify expected column exists
expected_dti_column = 'debt_to_income_ratio'
if expected_dti_column not in df.columns:
    print(f"Error: Expected column '{expected_dti_column}' not found. Please check your CSV file header.")
    exit(1)

# Step 2: Compute the DTI-Based Approval Rating
def dti_approval_rating(dti):
    """
    Converts a debt-to-income ratio (DTI) into an approval rating on a scale of 0 to 1.
    Formula: rating = 1 - (DTI / 50)
      - DTI = 0   -> rating = 1.0 (best)
      - DTI = 25  -> rating = 0.5
      - DTI = 50  -> rating = 0.0 (worst)
    The result is clamped between 0 and 1.
    """
    try:
        dti_value = float(dti)
    except:
        return None
    rating = 1 - (dti_value / 50.0)
    return max(min(rating, 1.0), 0.0)

df['approval_rating'] = df[expected_dti_column].apply(dti_approval_rating)
df = df.dropna(subset=['approval_rating'])

# Step 3: Determine Predicted Approval Decision (Threshold = 0.5)
threshold = 0.5
df['predicted_approval'] = df['approval_rating'].apply(lambda x: 1 if x >= threshold else 0)

# ===================================================
# PART 2: Analysis and Output Files
# ===================================================

# --- Output File 1: Histogram of Approval Ratings ---
plt.figure(figsize=(8, 5))
plt.hist(df['approval_rating'], bins=20, edgecolor='black')
plt.title("Histogram of Approval Ratings")
plt.xlabel("Approval Rating")
plt.ylabel("Frequency")
histogram_path = "output/approval_rating_histogram.png"
plt.savefig(histogram_path)
plt.close()

# --- Output File 2: Bar Graph of Average Approval Rating by Income Group ---
if 'income' in df.columns:
    try:
        df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    except Exception as e:
        print("Error creating income groups:", e)
        df['income_group'] = "Unknown"
else:
    df['income_group'] = "Unknown"

grouped_income = df.groupby('income_group')['approval_rating'].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(grouped_income['income_group'].astype(str), grouped_income['approval_rating'], color='skyblue')
plt.title("Average Approval Rating by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Average Approval Rating")
bargraph_income_path = "output/approval_rating_bargraph_income.png"
plt.savefig(bargraph_income_path)
plt.close()

# --- Output File 3: Bar Graph of Average Approval Rating by Gender ---
if 'derived_sex' in df.columns:
    grouped_gender = df.groupby('derived_sex')['approval_rating'].mean().reset_index()
else:
    grouped_gender = pd.DataFrame({'derived_sex': ['Unknown'], 'approval_rating': [np.nan]})

plt.figure(figsize=(8, 5))
plt.bar(grouped_gender['derived_sex'].astype(str), grouped_gender['approval_rating'], color='lightgreen')
plt.title("Average Approval Rating by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Approval Rating")
bargraph_gender_path = "output/approval_rating_bargraph_gender.png"
plt.savefig(bargraph_gender_path)
plt.close()

# --- Output File 4: Bar Graph of Average Approval Rating by Race ---
if 'derived_race' in df.columns:
    grouped_race = df.groupby('derived_race')['approval_rating'].mean().reset_index()
else:
    grouped_race = pd.DataFrame({'derived_race': ['Unknown'], 'approval_rating': [np.nan]})

plt.figure(figsize=(10, 5))
plt.bar(grouped_race['derived_race'].astype(str), grouped_race['approval_rating'], color='salmon')
plt.title("Average Approval Rating by Race")
plt.xlabel("Race")
plt.ylabel("Average Approval Rating")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
bargraph_race_path = "output/approval_rating_bargraph_race.png"
plt.savefig(bargraph_race_path)
plt.close()

# --- Output File 5: Bar Graph of Average Approval Rating by Race & Gender ---
# Create a combined category column (e.g., "Black Male", "White Female", etc.)
if 'derived_race' in df.columns and 'derived_sex' in df.columns:
    df['race_gender'] = df['derived_race'] + " " + df['derived_sex']
    grouped_rg = df.groupby('race_gender')['approval_rating'].mean().reset_index()
else:
    grouped_rg = pd.DataFrame({'race_gender': ['Unknown'], 'approval_rating': [np.nan]})

plt.figure(figsize=(10, 5))
plt.bar(grouped_rg['race_gender'].astype(str), grouped_rg['approval_rating'], color='orchid')
plt.title("Average Approval Rating by Race & Gender")
plt.xlabel("Race & Gender")
plt.ylabel("Average Approval Rating")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
bargraph_rg_path = "output/approval_rating_bargraph_race_gender.png"
plt.savefig(bargraph_rg_path)
plt.close()

# --- Output File 6: Analysis Summary Document ---
total_apps = len(df)
overall_avg_rating = df['approval_rating'].mean()
approval_counts = df['predicted_approval'].value_counts().to_dict()

summary_lines = []
summary_lines.append("DTI-Based Mortgage Approval Analysis Summary")
summary_lines.append("=============================================")
summary_lines.append(f"Total Applications Analyzed: {total_apps}")
summary_lines.append(f"Overall Average Approval Rating: {overall_avg_rating:.3f}")
summary_lines.append(f"Predicted Approvals (rating >= {threshold}): {approval_counts.get(1, 0)}")
summary_lines.append(f"Predicted Denials (rating < {threshold}): {approval_counts.get(0, 0)}")
summary_lines.append("\nAverage Approval Rating by Income Group:")
for idx, row in grouped_income.iterrows():
    summary_lines.append(f"  {row['income_group']}: {row['approval_rating']:.3f}")
summary_lines.append("\nAverage Approval Rating by Gender:")
for idx, row in grouped_gender.iterrows():
    summary_lines.append(f"  {row['derived_sex']}: {row['approval_rating']:.3f}")
summary_lines.append("\nAverage Approval Rating by Race:")
for idx, row in grouped_race.iterrows():
    summary_lines.append(f"  {row['derived_race']}: {row['approval_rating']:.3f}")
summary_lines.append("\nAverage Approval Rating by Race & Gender:")
for idx, row in grouped_rg.iterrows():
    summary_lines.append(f"  {row['race_gender']}: {row['approval_rating']:.3f}")
summary_lines.append("\nOutput Files:")
summary_lines.append(f"  Histogram: {histogram_path}")
summary_lines.append(f"  Income Group Bar Graph: {bargraph_income_path}")
summary_lines.append(f"  Gender Bar Graph: {bargraph_gender_path}")
summary_lines.append(f"  Race Bar Graph: {bargraph_race_path}")
summary_lines.append(f"  Race & Gender Bar Graph: {bargraph_rg_path}")
summary_lines.append("  Summary Document: output/analysis_summary.txt")
summary_lines.append("\nNote: This analysis ignores sensitive attributes in decision-making, but breakdowns by gender and race are provided for transparency.")

summary_text = "\n".join(summary_lines)
summary_path = "output/analysis_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print("✅ Analysis complete. Output files created:")
print("   Histogram:", histogram_path)
print("   Income Group Bar Graph:", bargraph_income_path)
print("   Gender Bar Graph:", bargraph_gender_path)
print("   Race Bar Graph:", bargraph_race_path)
print("   Race & Gender Bar Graph:", bargraph_rg_path)
print("   Summary Document:", summary_path)
