import pandas as pd
import matplotlib.pyplot as plt

def get_grouped_applicants(csv_file_path):
    """
    Reads the CSV file and processes applicant data for the equality of outcomes algorithm.
    
    Steps:
    1. Load the CSV data into a DataFrame.
    2. Normalize the column names (lowercase and strip whitespace).
    3. Check for required columns: 'derived_ethnicity', 'derived_race', and 'derived_sex'.
    4. Convert 'derived_ethnicity' to lowercase for reliable matching.
    5. If 'derived_ethnicity' equals "hispanic or latino", override 'derived_race' with "Hispanic".
    6. Standardize 'derived_race' and 'derived_sex' to title case.
    7. Ensure an 'id' column exists (auto-generate if missing).
    8. Convert 'debt_to_income_ratio' to numeric (coerce errors to NaN and drop them).
    9. Create a 'group' column combining 'derived_race' and 'derived_sex'.
    10. Group the applicants by this new 'group' value.
    
    Returns:
        A dictionary mapping each group (e.g., "White Female", "Hispanic Male") to a list of applicant dictionaries.
    """
    # Load CSV data into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Normalize column names: lowercase and strip whitespace
    df.columns = df.columns.str.strip().str.lower()
    
    # Check for required columns
    required_columns = ['derived_ethnicity', 'derived_race', 'derived_sex']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The CSV file must contain the column '{col}'. Available columns: {df.columns.tolist()}")
    
    # Ensure 'id' column exists; auto-generate if missing
    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df) + 1))
    
    # Convert 'derived_ethnicity' to string, lowercase, and strip whitespace for matching
    df['derived_ethnicity'] = df['derived_ethnicity'].astype(str).str.lower().str.strip()
    
    # Override 'derived_race' with "Hispanic" if 'derived_ethnicity' exactly equals "hispanic or latino"
    df['derived_race'] = df.apply(
        lambda row: "Hispanic" if row['derived_ethnicity'] == "hispanic or latino" else row['derived_race'],
        axis=1
    )
    
    # Standardize 'derived_race' and 'derived_sex' to title case
    df['derived_race'] = df['derived_race'].astype(str).str.title()
    df['derived_sex'] = df['derived_sex'].astype(str).str.title()
    
    # Convert 'debt_to_income_ratio' to numeric, coercing errors to NaN, then drop NaNs
    if 'debt_to_income_ratio' in df.columns:
        df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
        df = df.dropna(subset=['debt_to_income_ratio'])
    
    # Debugging: Print unique values
    print("Unique Values in 'derived_race':", df['derived_race'].unique())
    print("Unique Values in 'derived_ethnicity':", df['derived_ethnicity'].unique())
    
    # Create a new 'group' column combining 'derived_race' and 'derived_sex'
    df['group'] = df['derived_race'] + " " + df['derived_sex']
    print("Unique Groups:", df['group'].unique())
    
    # Group the applicants by the 'group' column
    grouped_applicants = {}
    for group, group_df in df.groupby('group'):
        grouped_applicants[group] = group_df.to_dict(orient='records')
    
    return grouped_applicants

def display_statistics_table_and_graph(grouped_applicants, thresholds, ratio_field='debt_to_income_ratio'):
    """
    Constructs a DataFrame of approval statistics and displays them as a table and as a bar chart.
    
    The statistics include:
      - Group name,
      - Total number of applicants,
      - The max ratio (i.e., the threshold) used for that group,
      - The number of approved applicants (those with a ratio <= threshold),
      - The approval rate.
    
    Approval rate is computed as: approved / total.
    
    Parameters:
      - grouped_applicants: dict mapping group names to a list of applicant dictionaries.
      - thresholds: dict mapping group names to the debt-to-income ratio threshold.
      - ratio_field: the field name in each applicant's dict representing the debt-to-income ratio.
    """
    data = []
    for group, apps in grouped_applicants.items():
        total = len(apps)
        if total == 0:
            continue
        threshold = thresholds.get(group, float('-inf'))
        approved = sum(1 for app in apps if float(app[ratio_field]) <= threshold)
        rate = approved / total
        data.append({
            "Group": group,
            "Applicants": total,
            "Max Ratio": threshold,
            "Approved": approved,
            "Rate (%)": round(rate * 100, 1)
        })
    
    df_stats = pd.DataFrame(data)
    df_stats = df_stats.sort_values("Group")
    
    # Create figure with two subplots: bar chart and table.
    fig, (ax_bar, ax_table) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bar Chart
    ax_bar.bar(df_stats["Group"], df_stats["Rate (%)"], color='skyblue')
    ax_bar.set_ylabel("Approval Rate (%)")
    ax_bar.set_title("Approval Rate by Group")
    ax_bar.set_ylim(0, 100)
    ax_bar.tick_params(axis='x', rotation=45)
    
    # Table
    ax_table.axis('off')
    ax_table.axis('tight')
    table = ax_table.table(cellText=df_stats.values,
                           colLabels=df_stats.columns,
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # For testing purposes, you can run:
    csv_file = "bigdata.csv"  # Adjust path as necessary
    try:
        groups = get_grouped_applicants(csv_file)
        for group, applicants in groups.items():
            print(f"Group: {group}, Count: {len(applicants)}")
    except ValueError as e:
        print(e)
