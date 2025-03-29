import json
from outcome_data_processing import get_grouped_applicants
from outcome_thresholds import calculate_approvals, print_statistics

def run_pipeline(input_csv, approval_rate=0.5, output_json="decisions.json"):
    try:
        print("🔍 Processing applicant data...")
        grouped_applicants = get_grouped_applicants(input_csv)
        
        print("⚖️  Calculating approval thresholds...")
        decisions, thresholds = calculate_approvals(
            grouped_applicants,
            approval_rate=approval_rate,
            ratio_field='debt_to_income_ratio'  # Explicitly specify the field
        )
        
        print("\n📊 Approval Statistics:")
        print_statistics(grouped_applicants, thresholds, ratio_field='debt_to_income_ratio')
        
        with open(output_json, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\n✅ Saved decisions to {output_json}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--rate", type=float, default=0.5, help="Target approval rate (0.0-1.0)")
    parser.add_argument("--output", default="decisions.json", help="Output JSON file path")
    args = parser.parse_args()
    
    run_pipeline(args.input, args.rate, args.output)