

import pandas as pd
import matplotlib.pyplot as plt

def get_grouped_applicants(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = ['derived_ethnicity', 'derived_race', 'derived_sex']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df) + 1))

    df['derived_ethnicity'] = df['derived_ethnicity'].astype(str).str.lower().str.strip()
    df['derived_race'] = df.apply(
        lambda row: "Hispanic" if row['derived_ethnicity'] == "hispanic or latino" else row['derived_race'], axis=1
    )

    df['derived_race'] = df['derived_race'].astype(str).str.title()
    df['derived_sex'] = df['derived_sex'].astype(str).str.title()

    if 'debt_to_income_ratio' in df.columns:
        df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
        df = df.dropna(subset=['debt_to_income_ratio'])

    df['group'] = df['derived_race'] + " " + df['derived_sex']

    grouped_applicants = {}
    for group, group_df in df.groupby('group'):
        grouped_applicants[group] = group_df.to_dict(orient='records')

    return grouped_applicants

def display_statistics_table_and_graph(grouped_applicants, thresholds, ratio_field='debt_to_income_ratio', save_path=None):
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

    df_stats = pd.DataFrame(data).sort_values("Group")

    fig, (ax_bar, ax_table) = plt.subplots(2, 1, figsize=(12, 10))

    ax_bar.bar(df_stats["Group"], df_stats["Rate (%)"], color='skyblue')
    ax_bar.set_ylabel("Approval Rate (%)")
    ax_bar.set_title("Approval Rate by Group")
    ax_bar.set_ylim(0, 100)
    ax_bar.tick_params(axis='x', rotation=45)

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

    if save_path:
        plt.savefig(save_path)

    return fig
