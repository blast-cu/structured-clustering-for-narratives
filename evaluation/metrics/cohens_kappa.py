import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy import stats


def compute_kappa_statistics(y1, y2):
    """
    Compute Cohen's kappa with p-value and confidence interval.
    
    Based on the asymptotic distribution of kappa statistic.
    """
    n = len(y1)
    kappa = cohen_kappa_score(y1, y2)
    
    # Create confusion matrix
    cm = confusion_matrix(y1, y2)
    
    # Calculate observed agreement (Po)
    po = np.trace(cm) / n
    
    # Calculate marginal totals
    row_totals = np.sum(cm, axis=1) / n
    col_totals = np.sum(cm, axis=0) / n
    
    # Calculate expected agreement (Pe)
    pe = np.sum(row_totals * col_totals)
    
    # Calculate variance of kappa (Fleiss et al. 1969)
    # This is the asymptotic variance under the null hypothesis (kappa = 0)
    
    # Calculate variance components
    var_components = []
    for i in range(len(row_totals)):
        for j in range(len(col_totals)):
            if i == j:  # Agreement cell
                term = row_totals[i] * col_totals[j] * (1 - (row_totals[i] + col_totals[j]) * (1 - kappa))**2
            else:  # Disagreement cell
                term = row_totals[i] * col_totals[j] * ((row_totals[i] + col_totals[j]) * (1 - kappa))**2
            var_components.append(term)
    
    # Simplified variance calculation (conservative approach)
    # Under null hypothesis that kappa = 0
    var_kappa_null = pe / (n * (1 - pe)**2)
    
    # More accurate variance calculation (Fleiss et al.)
    # For confidence intervals when kappa != 0
    theta = 0
    for i in range(len(row_totals)):
        for j in range(len(col_totals)):
            theta += cm[i, j] * (row_totals[i] + col_totals[j])**2
    theta = theta / n**2
    
    var_kappa = (po * (1 - po)) / (n * (1 - pe)**2) + \
                (2 * (1 - po) * (2 * po * pe - theta)) / (n * (1 - pe)**3) + \
                ((1 - po)**2 * (theta - 4 * pe**2)) / (n * (1 - pe)**4)
    
    # Standard error
    se_kappa = np.sqrt(var_kappa)
    
    # Z-statistic for testing H0: kappa = 0
    z_stat = kappa / np.sqrt(var_kappa_null)
    
    # P-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # 95% Confidence interval
    z_critical = stats.norm.ppf(0.975)  # 1.96 for 95% CI
    ci_lower = kappa - z_critical * se_kappa
    ci_upper = kappa + z_critical * se_kappa
    
    return kappa, p_value, (ci_lower, ci_upper), se_kappa, z_stat


def compute_cohens_kappa():
    """
    Compute unweighted Cohen's Kappa for n-label classification annotation task
    with p-value and confidence interval.
    
    Input: TSV file with two columns representing annotations from two annotators.
    Each row is one annotation example with labels from both annotators.
    """
    # Read the TSV file
    input_file = "./data/mfc/immigration/annotations.tsv"
    data = pd.read_csv(input_file, sep='\t', header=None, names=['annotator1', 'annotator2'])
    
    print(f"Loaded {len(data)} annotations from {input_file}")
    
    # Extract annotations
    annotator1_labels = data['annotator1'].values
    annotator2_labels = data['annotator2'].values
    
    # Get unique labels from both annotators
    all_labels = sorted(set(annotator1_labels) | set(annotator2_labels))
    print(f"Found {len(all_labels)} unique labels: {all_labels}")
    
    # Compute Cohen's Kappa with statistics
    kappa, p_value, (ci_lower, ci_upper), se, z_stat = compute_kappa_statistics(
        annotator1_labels, annotator2_labels
    )
    
    # Pretty print results (multiplied by 100, 2 decimal places)
    print(f"\nCohen's Kappa Results:")
    print(f"Kappa: {kappa * 100:.2f}")
    print(f"Standard Error: {se * 100:.2f}")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_value:.6f}")
    print(f"95% Confidence Interval: [{ci_lower * 100:.2f}, {ci_upper * 100:.2f}]")
    
    # Statistical significance
    if p_value < 0.001:
        sig_level = "p < 0.001"
    elif p_value < 0.01:
        sig_level = "p < 0.01"
    elif p_value < 0.05:
        sig_level = "p < 0.05"
    else:
        sig_level = "not significant"
    
    print(f"Significance: {sig_level}")
    
    # Print results in tab-separated format (multiplied by 100, 2 decimal places)
    print(f"\nTab-separated results:")
    print(f"{kappa * 100:.2f}\t{se * 100:.2f}\t{p_value:.6f}\t{ci_lower * 100:.2f}\t{ci_upper * 100:.2f}")
    
    return kappa, p_value, (ci_lower, ci_upper)


def main():
    compute_cohens_kappa()


if __name__ == "__main__":
    main()