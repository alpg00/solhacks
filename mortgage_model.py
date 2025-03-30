import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_mortgage_model():
    if not os.path.exists('output'):
        os.makedirs('output')

    try:
        df = pd.read_csv('data/bigdata.csv', engine='python', on_bad_lines='skip', sep=None)
        print("✅ CSV loaded successfully. Shape:", df.shape)
    except Exception as e:
        print("Error loading CSV:", e)
        return {}

    expected_dti_column = 'debt_to_income_ratio'
    if expected_dti_column not in df.columns:
        print(f"Error: Expected column '{expected_dti_column}' not found.")
        return {}

    def dti_approval_rating(dti):
        try:
            dti_value = float(dti)
        except:
            return None
        rating = 1 - (dti_value / 50.0)
        return max(min(rating, 1.0), 0.0)

    df['approval_rating'] = df[expected_dti_column].apply(dti_approval_rating)
    df = df.dropna(subset=['approval_rating'])

    threshold = 0.5
    df['predicted_approval'] = df['approval_rating'].apply(lambda x: 1 if x >= threshold else 0)

    if 'income' in df.columns:
        try:
            df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        except:
            df['income_group'] = "Unknown"
    else:
        df['income_group'] = "Unknown"

    grouped_income = df.groupby('income_group')['approval_rating'].mean().reset_index()
    income_bar_path = "output/approval_rating_bargraph_income.png"
    plt.figure(figsize=(8, 5))
    plt.bar(grouped_income['income_group'].astype(str), grouped_income['approval_rating'], color='skyblue')
    plt.title("Average Approval Rating by Income Group")
    plt.xlabel("Income Group")
    plt.ylabel("Average Approval Rating")
    plt.savefig(income_bar_path)
    plt.close()

    if 'derived_sex' in df.columns:
        grouped_gender = df.groupby('derived_sex')['approval_rating'].mean().reset_index()
    else:
        grouped_gender = pd.DataFrame({'derived_sex': ['Unknown'], 'approval_rating': [np.nan]})
    gender_bar_path = "output/approval_rating_bargraph_gender.png"
    plt.figure(figsize=(8, 5))
    plt.bar(grouped_gender['derived_sex'].astype(str), grouped_gender['approval_rating'], color='lightgreen')
    plt.title("Average Approval Rating by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average Approval Rating")
    plt.savefig(gender_bar_path)
    plt.close()

    if 'derived_race' in df.columns:
        grouped_race = df.groupby('derived_race')['approval_rating'].mean().reset_index()
    else:
        grouped_race = pd.DataFrame({'derived_race': ['Unknown'], 'approval_rating': [np.nan]})
    race_bar_path = "output/approval_rating_bargraph_race.png"
    plt.figure(figsize=(10, 5))
    plt.bar(grouped_race['derived_race'].astype(str), grouped_race['approval_rating'], color='salmon')
    plt.title("Average Approval Rating by Race")
    plt.xlabel("Race")
    plt.ylabel("Average Approval Rating")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(race_bar_path)
    plt.close()

    if 'derived_race' in df.columns and 'derived_sex' in df.columns:
        df['race_gender'] = df['derived_race'] + " " + df['derived_sex']
        grouped_rg = df.groupby('race_gender')['approval_rating'].mean().reset_index()
    else:
        grouped_rg = pd.DataFrame({'race_gender': ['Unknown'], 'approval_rating': [np.nan]})
    race_gender_bar_path = "output/approval_rating_bargraph_race_gender.png"
    plt.figure(figsize=(10, 5))
    plt.bar(grouped_rg['race_gender'].astype(str), grouped_rg['approval_rating'], color='orchid')
    plt.title("Average Approval Rating by Race & Gender")
    plt.xlabel("Race & Gender")
    plt.ylabel("Average Approval Rating")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(race_gender_bar_path)
    plt.close()

    summary_path = "output/analysis_summary.txt"
    summary_lines = [
        "DTI-Based Mortgage Approval Analysis Summary\n",
        f"Total Applications Analyzed: {len(df)}",
        f"Overall Average Approval Rating: {df['approval_rating'].mean():.3f}",
        f"Predicted Approvals (rating >= {threshold}): {df['predicted_approval'].sum()}",
        f"Predicted Denials (rating < {threshold}): {(df['predicted_approval'] == 0).sum()}"
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return {
        "income_bar": income_bar_path,
        "gender_bar": gender_bar_path,
        "race_bar": race_bar_path,
        "race_gender_bar": race_gender_bar_path,
        "summary": summary_path
    }

# Do NOT call run_mortgage_model() here unless testing manually
if __name__ == "__main__":
    run_mortgage_model()