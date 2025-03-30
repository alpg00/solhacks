import pandas as pd

def get_grouped_applicants(csv_file_path):
    """
    Reads the CSV file and processes applicant data for the equality of outcomes algorithm.

    Steps:
    1. Load the CSV data into a DataFrame.
    2. Normalize the column names (lowercase and strip whitespace).
    3. Check for required columns: 'derived_ethnicity', 'derived_race', and 'derived_sex'.
    4. Convert 'derived_ethnicity' to lowercase for reliable matching.
    5. If 'derived_ethnicity' indicates Hispanic, override 'derived_race' with "Hispanic".
    6. Standardize 'derived_race' and 'derived_sex' to title case.
    7. Ensure an 'id' column exists (auto-generate if missing).
    8. Create a 'group' column combining 'derived_race' and 'derived_sex'.
    9. Group the applicants by this new 'group' value.

    Returns:
        A dictionary mapping each group (e.g., "White Female", "Hispanic Male") to a list of applicant dictionaries.
    """
    # Load the CSV data into a DataFrame
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
        df.insert(0, 'id', range(1, len(df) + 1))  # Assigns IDs from 1 to N
    
    # Convert 'derived_ethnicity' to string and lowercase it for matching
    df['derived_ethnicity'] = df['derived_ethnicity'].astype(str).str.lower()
    
    # Override 'derived_race' with "Hispanic" if 'derived_ethnicity' indicates Hispanic
    df['derived_race'] = df.apply(
        lambda row: "Hispanic" if "hispanic" in row['derived_ethnicity'] else row['derived_race'],
        axis=1
    )
    
    # Standardize 'derived_race' and 'derived_sex' to title case
    df['derived_race'] = df['derived_race'].astype(str).str.title()
    df['derived_sex'] = df['derived_sex'].astype(str).str.title()

    # Convert 'debt_to_income_ratio' to numeric, coercing errors to NaN
    df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
    
    # Remove rows with NaN values in debt-to-income ratio
    df = df.dropna(subset=['debt_to_income_ratio'])
    
    # Debugging: Print the unique values in 'derived_race' and 'derived_ethnicity'
    print("Unique Values in 'derived_race':", df['derived_race'].unique())
    print("Unique Values in 'derived_ethnicity':", df['derived_ethnicity'].unique())

    # Create a new 'group' column combining 'derived_race' and 'derived_sex' (e.g., "White Female")
    df['group'] = df['derived_race'] + " " + df['derived_sex']
    
    # Debugging: Check the unique values in the 'group' column
    print("Unique Groups:", df['group'].unique())
    
    # Group the applicants by the 'group' column
    grouped_applicants = {}
    for group, group_df in df.groupby('group'):
        grouped_applicants[group] = group_df.to_dict(orient='records')
    
    return grouped_applicants

if __name__ == "__main__":
    csv_file = "bigdata.csv"  # Adjust path as necessary
    try:
        groups = get_grouped_applicants(csv_file)
        # Print out a summary: each group and the count of applicants in that group
        for group, applicants in groups.items():
            print(f"Group: {group}, Count: {len(applicants)}")
    except ValueError as e:
        print(e)
