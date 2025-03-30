import pandas as pd
import os

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

# Check if required columns exist
required_columns = ['debt_to_income_ratio', 'derived_ethnicity']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Error: Missing required columns: {missing}. Please check your CSV header.")
    exit(1)

# Step 2: Define the DTI-Based Approval Rating Function
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

# Step 3: Compute the Approval Rating for Each Application
df['approval_rating'] = df['debt_to_income_ratio'].apply(dti_approval_rating)
df = df.dropna(subset=['approval_rating'])

# Step 4: Compute Average Approval Ratings by Ethnicity
# Group by 'derived_ethnicity'
grouped_ethnicity = df.groupby('derived_ethnicity')['approval_rating'].mean().reset_index()
min_avg = grouped_ethnicity['approval_rating'].min()
max_avg = grouped_ethnicity['approval_rating'].max()
if max_avg > 0:
    clear_score = min_avg / max_avg
else:
    clear_score = None

# Step 5: Write the ClearScore Summary to a Text File
summary_lines = []
summary_lines.append("ClearScore Summary")
summary_lines.append("==================")
summary_lines.append("")
summary_lines.append("ClearScore is defined as the ratio of the minimum average approval rating to the maximum average approval rating across ethnic groups.")
summary_lines.append("")
summary_lines.append("Formula: ClearScore = (Minimum Average Approval Rating) / (Maximum Average Approval Rating)")
summary_lines.append("")
summary_lines.append(f"Minimum Average Approval Rating among ethnic groups: {min_avg:.3f}")
summary_lines.append(f"Maximum Average Approval Rating among ethnic groups: {max_avg:.3f}")
if clear_score is not None:
    summary_lines.append(f"Computed ClearScore: {clear_score:.3f}")
else:
    summary_lines.append("Computed ClearScore: N/A (Maximum average approval rating is 0)")
summary_lines.append("")
summary_lines.append("Interpretation:")
summary_lines.append("- A ClearScore of 1.0 indicates perfect equality: all ethnic groups have the same average approval rating.")
summary_lines.append("- A ClearScore less than 1.0 indicates disparity among groups, with lower values signaling greater inequality in loan approval fairness.")
summary_lines.append("")
summary_lines.append("This metric helps evaluate how evenly the loan approval process treats applicants across different ethnic groups when using a DTI-based approval rating.")
summary_text = "\n".join(summary_lines)

summary_path = "output/clear_score_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print("✅ ClearScore summary has been saved to", summary_path)
