import numpy as np

def calculate_approvals(
    grouped_applicants,
    approval_rate=0.5,
    ratio_field='debt_to_income_ratio',  
    id_field='id'
):
    """
    Enforces equal approval rates using debt-to-income ratio (lower is better)
    """
    if not 0 <= approval_rate <= 1:
        raise ValueError(f"approval_rate must be between 0 and 1, got {approval_rate}")
    
    decisions = {}
    group_thresholds = {}

    for group, applicants in grouped_applicants.items():
        if not applicants:
            group_thresholds[group] = float('-inf')  # Now using -inf since lower ratios are better
            continue

        try:
            # Extract debt-to-income ratios (lower values are better)
            ratios = np.array([float(app[ratio_field]) for app in applicants])
            
            # Calculate threshold (percentile inverted since lower ratios are better)
            threshold = np.percentile(ratios, 100 * approval_rate)  # Changed calculation
            group_thresholds[group] = threshold
            
            # Approve applicants BELOW the threshold (unlike credit scores where higher was better)
            for app in applicants:
                app_ratio = float(app[ratio_field])
                decisions[app[id_field]] = "approved" if app_ratio <= threshold else "denied"
                
        except KeyError as e:
            raise KeyError(f"Missing required field {str(e)} in applicant data")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid ratio data in group '{group}': {str(e)}")

    return decisions, group_thresholds

def print_statistics(grouped_applicants, thresholds, ratio_field='debt_to_income_ratio'):
    """Print statistics with debt-to-income ratio logic"""
    print("\n" + "="*60)
    print("Approval Statistics (Lower Ratios Are Better)".center(60))
    print("="*60)
    print(f"{'Group':<20} | {'Applicants':>10} | {'Max Ratio':>10} | {'Approved':>10} | {'Rate':>6}")
    print("-"*60)
    
    for group, apps in grouped_applicants.items():
        if not apps:
            print(f"{group:<20} | {'0':>10} | {'N/A':>10} | {'0':>10} | {'0%':>6}")
            continue
            
        threshold = thresholds.get(group, float('-inf'))
        approved = sum(1 for app in apps if float(app[ratio_field]) <= threshold)
        rate = approved / len(apps)
        
        print(f"{group:<20} | {len(apps):>10} | {threshold:>10.2f} | "
              f"{approved:>10} | {rate:>6.1%}")