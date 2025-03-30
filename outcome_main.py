
import json
from outcome_data_processing import get_grouped_applicants, display_statistics_table_and_graph
from outcome_thresholds import calculate_approvals, print_statistics

def run_pipeline(input_csv, approval_rate=0.5, output_json="decisions.json"):
    try:
        print("🔍 Processing applicant data...")
        grouped_applicants = get_grouped_applicants(input_csv)

        print("⚖️  Calculating approval thresholds...")
        decisions, thresholds = calculate_approvals(
            grouped_applicants,
            approval_rate=approval_rate,
            ratio_field='debt_to_income_ratio'
        )

        print("\n📊 Approval Statistics:")
        print_statistics(grouped_applicants, thresholds, ratio_field='debt_to_income_ratio')

        # Save graph to file
        output_path = "output/ml_outcome_graph.png"
        display_statistics_table_and_graph(
            grouped_applicants,
            thresholds,
            ratio_field='debt_to_income_ratio',
            save_path=output_path
        )

        with open(output_json, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\n✅ Saved decisions to {output_json}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None
    return output_path  # only return string path
