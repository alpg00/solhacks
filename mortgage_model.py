import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_mortgage_model():
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
        return {}

    # Debug: Print available columns
    print("Columns in CSV:", df.columns.tolist())

    expected_dti_column = 'debt_to_income_ratio'
    if expected_dti_column not in df.columns:
        print(f"Error: Expected column '{expected_dti_column}' not found.")
        return {}

    # Step 2: Compute the DTI-Based Approval Rating
    def dti_approval_rating(dti):
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

    # ---------------------------
    # Income Group Analysis Graph
    # ---------------------------
    if 'income' in df.columns:
        try:
            df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        except Exception as e:
            print("Error creating income groups:", e)
            df['income_group'] = "Unknown"
    else:
        df['income_group'] = "Unknown"
    grouped_income = df.groupby('income_group', observed=False)['approval_rating'].mean().reset_index()
    income_bar_path = "output/approval_rating_bargraph_income.png"
    plt.figure(figsize=(8, 5))
    plt.bar(grouped_income['income_group'].astype(str), grouped_income['approval_rating'], color='skyblue')
    plt.title("Average Approval Rating by Income Group")
    plt.xlabel("Income Group")
    plt.ylabel("Average Approval Rating")
    plt.savefig(income_bar_path)
    plt.close()

    # ---------------------------
    # Gender Analysis Graph
    # ---------------------------
    if 'derived_sex' in df.columns:
        grouped_gender = df.groupby('derived_sex', observed=False)['approval_rating'].mean().reset_index()
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

    # ---------------------------
    # Race Analysis Graph
    # ---------------------------
    if 'derived_race' in df.columns:
        grouped_race = df.groupby('derived_race', observed=False)['approval_rating'].mean().reset_index()
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

    # ---------------------------
    # Combined Race & Gender Analysis Graph
    # ---------------------------
    if 'derived_race' in df.columns and 'derived_sex' in df.columns:
        df['race_gender'] = df['derived_race'] + " " + df['derived_sex']
        grouped_rg = df.groupby('race_gender', observed=False)['approval_rating'].mean().reset_index()
    else:
        grouped_rg = pd.DataFrame({'race_gender': ['Unknown'], 'approval_rating': [np.nan]})
    race_gender_graph_path = "output/multicategorical_graph.png"
    plt.figure(figsize=(10, 5))
    plt.bar(grouped_rg['race_gender'].astype(str), grouped_rg['approval_rating'], color='orchid')
    plt.title("Average Approval Rating by Race & Gender")
    plt.xlabel("Race & Gender")
    plt.ylabel("Average Approval Rating")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(race_gender_graph_path)
    plt.close()

    # ---------------------------
    # Combined Race & Gender Table (Separate PNG)
    # ---------------------------
    if 'derived_race' in df.columns and 'derived_sex' in df.columns:
        df['race_gender'] = df['derived_race'] + " " + df['derived_sex']
        grouped_rg_table = df.groupby('race_gender', observed=False).agg(
            Applicants=("race_gender", "count"),
            Approved=("predicted_approval", "sum")
        ).reset_index()
        grouped_rg_table["Approval Rate"] = (grouped_rg_table["Approved"] / grouped_rg_table["Applicants"]) * 100
    else:
        grouped_rg_table = pd.DataFrame({'race_gender': ['Unknown'], 'Applicants': [np.nan], 'Approved': [np.nan], 'Approval Rate': [np.nan]})
    
    # Create a separate figure for the multicategorical table with larger formatting
    fig_table = plt.figure(figsize=(14, 8))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis("off")
    table_data = grouped_rg_table.values.tolist()
    col_labels = list(grouped_rg_table.columns)
    table = ax_table.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    table.set_fontsize(12)
    table.scale(1, 3)  # Increase vertical scale for a bigger table
    plt.tight_layout()
    multicat_table_path = "output/multicategorical_table.png"
    plt.savefig(multicat_table_path)
    plt.close()

    # ---------------------------
    # ClearScore Calculation (Ethnicity-based)
    # ---------------------------
    if 'derived_ethnicity' in df.columns:
        grouped_ethnicity = df.groupby('derived_ethnicity', observed=False)['approval_rating'].mean().reset_index()
        min_avg = grouped_ethnicity['approval_rating'].min()
        max_avg = grouped_ethnicity['approval_rating'].max()
        clear_score = min_avg / max_avg if max_avg > 0 else None
    else:
        clear_score = None

    # ---------------------------
    # Analysis Summary Document
    # ---------------------------
    summary_path = "output/analysis_summary.txt"
    summary_lines = [
        "ClearScore - Understanding Loan Fairness Using ML",
        "=================================================",
        f"Total Applications Analyzed: {len(df)}",
        f"Overall Average Approval Rating: {df['approval_rating'].mean():.3f}",
        f"Predicted Approvals (rating >= {threshold}): {df['predicted_approval'].sum()}",
        f"Predicted Denials (rating < {threshold}): {(df['predicted_approval'] == 0).sum()}",
        "",
        "Average Approval Rating by Income Group:"
    ]
    for idx, row in grouped_income.iterrows():
        summary_lines.append(f"  {row['income_group']}: {row['approval_rating']:.3f}")
    summary_lines.append("")
    if 'derived_sex' in df.columns:
        summary_lines.append("Average Approval Rating by Gender:")
        for idx, row in grouped_gender.iterrows():
            summary_lines.append(f"  {row['derived_sex']}: {row['approval_rating']:.3f}")
        summary_lines.append("")
    if 'derived_race' in df.columns:
        summary_lines.append("Average Approval Rating by Race:")
        for idx, row in grouped_race.iterrows():
            summary_lines.append(f"  {row['derived_race']}: {row['approval_rating']:.3f}")
        summary_lines.append("")
    summary_lines.append("Approval Rate by Combined Race & Gender:")
    for idx, row in grouped_rg_table.iterrows():
        summary_lines.append(f"  {row['race_gender']}: Applicants={int(row['Applicants'])}, Approved={int(row['Approved'])}, Approval Rate={row['Approval Rate']:.1f}%")
    summary_lines.append("")
    summary_lines.append("ClearScore Calculation (Ethnicity-based):")
    summary_lines.append("ClearScore = (Minimum Avg Approval Rating among Ethnic Groups) / (Maximum Avg Approval Rating among Ethnic Groups)")
    if 'derived_ethnicity' in df.columns:
        summary_lines.append(f"Minimum Avg Approval Rating (Ethnicity): {min_avg:.3f}")
        summary_lines.append(f"Maximum Avg Approval Rating (Ethnicity): {max_avg:.3f}")
        if clear_score is not None:
            summary_lines.append(f"Computed ClearScore: {clear_score:.3f}")
        else:
            summary_lines.append("Computed ClearScore: N/A")
    else:
        summary_lines.append("ClearScore: N/A (derived_ethnicity column not found)")
    summary_lines.append("")
    summary_lines.append("Output Files:")
    summary_lines.append(f"  Income Group Bar Graph: {income_bar_path}")
    summary_lines.append(f"  Gender Bar Graph: {gender_bar_path}")
    summary_lines.append(f"  Race Bar Graph: {race_bar_path}")
    summary_lines.append(f"  Combined Race & Gender Graph: {race_gender_graph_path}")
    summary_lines.append(f"  Combined Race & Gender Table: {multicat_table_path}")
    summary_lines.append(f"  Analysis Summary: {summary_path}")
    summary_lines.append("")
    summary_lines.append("Note: This analysis ignores sensitive attributes in decision-making, but provides breakdowns for transparency.")

    summary_text = "\n".join(summary_lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return {
        "income_bar": income_bar_path,
        "gender_bar": gender_bar_path,
        "race_bar": race_bar_path,
        "race_gender_bar": race_gender_graph_path,
        "multicategorical_table": multicat_table_path,
        "summary": summary_path
    }

if __name__ == "__main__":
    run_mortgage_model()
