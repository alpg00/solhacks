import numpy as np

def calculate_approvals(grouped_applicants, approval_rate=0.5, ratio_field='debt_to_income_ratio', id_field='id'):
    """
    Enforces equal approval rates using debt-to-income ratio (lower is better)
    
    Parameters:
      - grouped_applicants: dict mapping group names to lists of applicant dicts.
      - approval_rate: target percentage (0 to 1) of approvals per group.
      - ratio_field: field in applicant dict holding the debt-to-income ratio.
      - id_field: field for the applicant's unique ID.
      
    Returns:
      - decisions: dict mapping applicant IDs to "approved" or "denied".
      - group_thresholds: dict mapping group names to the threshold value.
    """
    if not 0 <= approval_rate <= 1:
        raise ValueError(f"approval_rate must be between 0 and 1, got {approval_rate}")
    
    decisions = {}
    group_thresholds = {}
    
    for group, applicants in grouped_applicants.items():
        if not applicants:
            group_thresholds[group] = float('-inf')
            continue
        
        try:
            ratios = np.array([float(app[ratio_field]) for app in applicants])
            threshold = np.percentile(ratios, 100 * approval_rate)
            group_thresholds[group] = threshold
            
            for app in applicants:
                app_ratio = float(app[ratio_field])
                decisions[app[id_field]] = "approved" if app_ratio <= threshold else "denied"
        except KeyError as e:
            raise KeyError(f"Missing required field {str(e)} in applicant data")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid ratio data in group '{group}': {str(e)}")
    
    return decisions, group_thresholds

def print_statistics(grouped_applicants, thresholds, ratio_field='debt_to_income_ratio'):
    """Prints approval statistics for each group."""
    print("\n" + "="*60)
    print("Approval Statistics (Lower Ratios Are Better)".center(60))
    print("="*60)
    print(f"{'Group':<40} | {'Applicants':>10} | {'Max Ratio':>10} | {'Approved':>10} | {'Rate':>6}")
    print("-"*60)
    
    for group, apps in grouped_applicants.items():
        if not apps:
            print(f"{group:<40} | {'0':>10} | {'N/A':>10} | {'0':>10} | {'0%':>6}")
            continue
        
        threshold = thresholds.get(group, float('-inf'))
        approved = sum(1 for app in apps if float(app[ratio_field]) <= threshold)
        rate = approved / len(apps)
        print(f"{group:<40} | {len(apps):>10} | {threshold:>10.2f} | {approved:>10} | {rate:>6.1%}")