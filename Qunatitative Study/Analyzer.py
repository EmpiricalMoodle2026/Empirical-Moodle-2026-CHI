import os
import random
from collections import Counter
from statistics import mean, median, stdev
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import ranksums, mannwhitneyu, shapiro
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import numpy as np
import logging
from datetime import datetime
import json

# File paths
audit_issues_file = "Data/audit_issues.json"
mdl_a11y_issues_file = "Data/MDL_a11y_issues.json"
combined_a11y_file = "Data/combined_a11y_issues.json"
non_a11y_issues_file = "Data/non_a11y_issues_high_level.json"
non_a11y_issue_file_jsonl = "Data/non_a11y_issues_high_level.jsonl"
output_file = "Data/combined_a11y_issues.json"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Extract all issues from audit_issues.json
def extract_audit_issues(audit_data):
    all_issues = []
    for key in ["First Audit", "Second Audit", "Third Audit", "Fourth Audit"]:
        if key in audit_data:
            all_issues.extend(audit_data[key])
    return all_issues

def combine_issues(audit_issues, mdl_issues):
    combined = {}

    # Add audit issues to the combined dictionary
    for issue in audit_issues:
        issue_key = issue.get("issue_key")
        if issue_key:
            combined[issue_key] = issue

    # Add mdl_a11y issues to the combined dictionary
    for issue in mdl_issues:
        issue_key = issue.get("issue_key")
        if issue_key:
            combined[issue_key] = issue

    return list(combined.values())


def filter_resolved_issues(issues):
    """Filter issues that are resolved and exclude 'Epic' or 'Task'."""
    return [
        issue for issue in issues
        if issue.get('resolution_type') == 'Fixed' and issue.get('issue_type') not in ['Epic', 'Task']
    ]

def filter_by_date_range(issues, start_date, end_date):
    """Filter issues based on their creation date."""
    start_date = start_date.replace(tzinfo=None)  # Ensure offset-naive
    end_date = end_date.replace(tzinfo=None)      # Ensure offset-naive
    return [
        issue for issue in issues
        if start_date <= datetime.fromisoformat(issue['created_date']).replace(tzinfo=None) <= end_date
    ]


def calculate_numeric_stats(values):
    """Calculate Avg, Median, Min, Max, and Standard Deviation for a list of numbers."""
    if not values:
        return 0, 0, 0, 0, 0
    return mean(values), median(values), min(values), max(values), stdev(values), len(values) if len(values) > 1 else 0

def calculate_time_to_resolve(issues):
    """Calculate time-to-resolve statistics."""
    time_deltas = []
    for issue in issues:
        created_date = issue.get('created_date')
        resolved_date = issue.get('resolved_date')
        if created_date and resolved_date:
            delta = datetime.fromisoformat(resolved_date) - datetime.fromisoformat(created_date)
            time_deltas.append(delta.days)
    return calculate_numeric_stats(time_deltas)

def calculate_time_to_resolve_pure(issues):
    """Calculate time-to-resolve statistics."""
    time_deltas = []
    for issue in issues:
        created_date = issue.get('created_date')
        resolved_date = issue.get('resolved_date')
        if created_date and resolved_date:
            delta = datetime.fromisoformat(resolved_date) - datetime.fromisoformat(created_date)
            time_deltas.append(delta.days)
    return time_deltas

def summarize_versions(issues, key):
    """Summarize version-related attributes (fix_versions or affected_versions)."""
    version_counts = [len(issue.get(key, [])) for issue in issues]
    version_counts = [count for count in version_counts if count > 0]
    return calculate_numeric_stats(version_counts)

def summarize_description_words(issues, key):
    """Summarize version-related attributes (fix_versions or affected_versions)."""
    description_counts = [len(issue.get(key, []) or []) for issue in issues]
    description_counts = [count for count in description_counts if count > 0]
    return calculate_numeric_stats(description_counts)

def summarize_numeric_attribute(issues, key, zero_value = True):
    """Summarize a numeric attribute with Avg, Median, Min, Max, and Std Dev."""
    if zero_value:
        values = [issue[key] for issue in issues if issue.get(key) is not None]
    else:
        values = [issue[key] for issue in issues if issue.get(key) is not None and issue[key] > 0]
    return calculate_numeric_stats(values)


def perform_wilcoxon_test(metric_audit, metric_non_audit, metric_name):
    """Perform Wilcoxon Rank-Sum test and log results."""
    test_name = None
    if abs(len(metric_non_audit)-len(metric_audit)) > 2000:
        stat, p_value = mannwhitneyu(metric_audit, metric_non_audit, alternative="two-sided")
        test_name = "Mann-Whitney U Test"
    else:
        stat, p_value = ranksums(metric_audit, metric_non_audit)
        test_name = "Wilcoxon Rank-Sum Test"
    logging.info(f"\n{metric_name} - {test_name}:")
    logging.info(f"Test Statistic: {stat:.30f}, P-value: {p_value:.30f}")
    if p_value < 0.05:
        logging.info(f"Significant difference found (P < 0.05) for {metric_name}.")
    else:
        logging.info(f"No significant difference (P >= 0.05) for {metric_name}.")


def RQ2_2():
    """
    Analyzes and compares attributes of audit and non-audit issues.
    Performs appropriate statistical tests based on data type and distribution.
    Includes effect size calculations and multiple comparison correction.
    """
    # Load and prepare data
    audit_issues = load_and_filter_audit_issues()
    non_audit_issues = load_and_filter_non_audit_issues()

    logging.info(f"Number of audit issues: {len(audit_issues)}")
    logging.info(f"Number of non-audit issues: {len(non_audit_issues)}")

    # Define attributes to compare with their display names and data types
    attributes = [
        ('time_logged', "Time Logged", "continuous"),
        ('resolution_time', "Resolution Time", "continuous"),
        ('fix_versions', "Fix Versions", "count"),
        ('affected_versions', "Affected Versions", "count"),
        ('num_participants', "Participants", "count"),
        ('num_comments', "Comments", "count"),
        ('num_commits', "Commits", "count"),
        ('num_watchers', "Watchers", "count"),
        ('num_votes', "Votes", "count"),
        ('num_attachments', "Attachments", "count"),
        ('description', "Description Words", "count"),
        ('priority', "Priority", "ordinal")
    ]

    # Compare all attributes and log descriptive results
    compare_and_log_attributes(audit_issues, non_audit_issues, attributes)

    # Perform advanced statistical tests with appropriate effect sizes
    results = perform_statistical_analysis(audit_issues, non_audit_issues, attributes)

    # Apply Benjamini-Hochberg correction for multiple comparisons
    apply_multiple_testing_correction(results, attributes)

    # Save results to file
    save_results_to_file(results, attributes, audit_issues, non_audit_issues, "RQ/RQ2")


def load_and_filter_audit_issues():
    """Load audit issues from file and filter to only resolved issues."""
    audit_issues_data = load_json(audit_issues_file)
    audit_issues = []

    for audit_key in ["First Audit", "Second Audit", "Third Audit", "Fourth Audit"]:
        audit_issues.extend(audit_issues_data.get(audit_key, []))

    return filter_resolved_issues(audit_issues)


def load_and_filter_non_audit_issues():
    """Load non-audit issues from file and filter by date range and uniqueness."""
    non_audit_issues = load_json(mdl_a11y_issues_file)

    # Filter by date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 11, 16)
    non_audit_issues = filter_by_date_range(non_audit_issues, start_date, end_date)

    # Filter by uniqueness (compared to audit issues)
    resolved_audit_issues = load_and_filter_audit_issues()
    unique_non_audit_issues = filter_by_issue_key(resolved_audit_issues, non_audit_issues)

    return filter_resolved_issues(unique_non_audit_issues)


def compare_and_log_attributes(audit_issues, non_audit_issues, attributes):
    """Compare and log statistics for all attributes between audit and non-audit issues."""
    results = {}

    for attr_key, display_name, data_type in attributes:
        if attr_key == 'resolution_time':
            audit_stats = calculate_time_to_resolve(audit_issues)
            non_audit_stats = calculate_time_to_resolve(non_audit_issues)
        elif attr_key in ['fix_versions', 'affected_versions']:
            audit_stats = summarize_versions(audit_issues, attr_key)
            non_audit_stats = summarize_versions(non_audit_issues, attr_key)
        elif attr_key == 'description':
            audit_stats = summarize_description_words(audit_issues, attr_key)
            non_audit_stats = summarize_description_words(non_audit_issues, attr_key)
        elif attr_key == 'priority':
            audit_values = extract_priority_values(audit_issues)
            non_audit_values = extract_priority_values(non_audit_issues)
            audit_stats = summarize_ordinal_attribute(audit_values)
            non_audit_stats = summarize_ordinal_attribute(non_audit_values)
        else:
            remove_outliers = attr_key != 'num_commits'
            audit_stats = summarize_numeric_attribute(audit_issues, attr_key, remove_outliers)
            non_audit_stats = summarize_numeric_attribute(non_audit_issues, attr_key, remove_outliers)

        results[attr_key] = (audit_stats, non_audit_stats)
        log_stats(display_name, audit_stats, non_audit_stats)

    return results


def log_stats(title, audit_stats, non_audit_stats):
    """Format and log the comparison statistics."""
    logging.info(f"\n{title} (Audit vs. Non-Audit):")
    logging.info(f"Audit - Avg: {audit_stats[0]:.2f}, Median: {audit_stats[1]:.2f}, "
                 f"Min: {audit_stats[2]}, Max: {audit_stats[3]}, "
                 f"Std Dev: {audit_stats[4]:.2f}, Count: {audit_stats[5]}")
    logging.info(f"Non-Audit - Avg: {non_audit_stats[0]:.2f}, Median: {non_audit_stats[1]:.2f}, "
                 f"Min: {non_audit_stats[2]}, Max: {non_audit_stats[3]}, "
                 f"Std Dev: {non_audit_stats[4]:.2f}, Count: {non_audit_stats[5]}")


def perform_statistical_analysis(audit_issues, non_audit_issues, attributes):
    """
    Perform appropriate statistical tests based on data type and distribution.
    Calculate relevant effect sizes for each attribute.

    Returns:
        Dict of results with p-values and effect sizes for each attribute
    """
    results = {}

    for attr_key, display_name, data_type in attributes:
        # Extract data
        if attr_key == 'resolution_time':
            audit_data = extract_resolution_times(audit_issues)
            non_audit_data = extract_resolution_times(non_audit_issues)
        elif attr_key in ['fix_versions', 'affected_versions']:
            filter_zeros = attr_key == 'fix_versions'
            audit_data = extract_version_counts(audit_issues, attr_key, filter_zeros)
            non_audit_data = extract_version_counts(non_audit_issues, attr_key, filter_zeros)
        elif attr_key == 'description':
            audit_data = extract_description_counts(audit_issues)
            non_audit_data = extract_description_counts(non_audit_issues)
        elif attr_key == 'priority':
            audit_data = extract_priority_values(audit_issues)
            non_audit_data = extract_priority_values(non_audit_issues)
        else:
            filter_zeros = attr_key == "num_commits"
            audit_data = extract_numeric_attribute(audit_issues, attr_key, filter_zeros)
            non_audit_data = extract_numeric_attribute(non_audit_issues, attr_key, filter_zeros)

        # Skip if not enough data
        if len(audit_data) < 5 or len(non_audit_data) < 5:
            logging.warning(f"Insufficient data for {display_name}, skipping statistical tests")
            continue

        # Perform statistical analysis based on data type
        if data_type == "continuous":
            result = analyze_continuous_variable(audit_data, non_audit_data, display_name)
        elif data_type == "count":
            result = analyze_count_variable(audit_data, non_audit_data, display_name)
        elif data_type == "binary":
            # Convert to binary format if needed
            result = analyze_binary_variable(audit_data, non_audit_data, display_name)
        elif data_type == "ordinal":
            result = analyze_ordinal_variable(audit_data, non_audit_data, display_name)
        else:
            # Default to non-parametric test
            result = analyze_continuous_variable(audit_data, non_audit_data, display_name, force_nonparametric=True)

        results[attr_key] = result

    return results


def analyze_continuous_variable(audit_data, non_audit_data, display_name, force_nonparametric=False):
    """
    Analyze continuous variables using appropriate statistical tests.
    Perform normality test to choose between parametric and non-parametric tests.

    Args:
        audit_data: List of values from audit issues
        non_audit_data: List of values from non-audit issues
        display_name: Name of the attribute for logging
        force_nonparametric: If True, use non-parametric test regardless of normality

    Returns:
        Dict with test results and effect sizes
    """
    # Check sample sizes
    if len(audit_data) < 8 or len(non_audit_data) < 8:
        # Sample too small for reliable normality test, use non-parametric
        is_normal = False
    else:
        # Check normality using Shapiro-Wilk test
        _, p_audit = shapiro(np.random.choice(audit_data, min(5000, len(audit_data)), replace=False))
        _, p_non_audit = shapiro(np.random.choice(non_audit_data, min(5000, len(non_audit_data)), replace=False))
        is_normal = (p_audit > 0.05) and (p_non_audit > 0.05) and not force_nonparametric

    audit_array = np.array(audit_data)
    non_audit_array = np.array(non_audit_data)

    if is_normal:
        # Use Welch's t-test for normal data (doesn't assume equal variances)
        t_stat, p_value = ttest_ind(audit_array, non_audit_array, equal_var=False)

        # Calculate Cohen's d effect size
        mean_diff = np.mean(audit_array) - np.mean(non_audit_array)
        pooled_std = np.sqrt(((len(audit_array) - 1) * np.var(audit_array, ddof=1) +
                              (len(non_audit_array) - 1) * np.var(non_audit_array, ddof=1)) /
                             (len(audit_array) + len(non_audit_array) - 2))
        effect_size = mean_diff / pooled_std
        effect_type = "Cohen's d"

        # Calculate 95% confidence interval for Cohen's d
        se = np.sqrt((len(audit_array) + len(non_audit_array)) /
                     (len(audit_array) * len(non_audit_array)) +
                     (effect_size ** 2) / (2 * (len(audit_array) + len(non_audit_array))))
        ci_lower = effect_size - 1.96 * se
        ci_upper = effect_size + 1.96 * se

        test_name = "Welch's t-test"
        logging.info(f"\n{display_name}: {test_name} - t={t_stat:.3f}, p={p_value:.5f}")

    else:
        # Use Mann-Whitney U test for non-normal data
        u_stat, p_value = mannwhitneyu(audit_array, non_audit_array, alternative='two-sided')

        # Calculate Cliff's Delta effect size for non-parametric data
        effect_size = calculate_cliffs_delta(audit_array, non_audit_array)
        effect_type = "Cliff's Delta"

        # Approximate 95% CI for Cliff's Delta
        n1, n2 = len(audit_array), len(non_audit_array)
        se = np.sqrt((2 * (n1 + n2 + 1)) / (3 * n1 * n2))
        ci_lower = max(-1, effect_size - 1.96 * se)
        ci_upper = min(1, effect_size + 1.96 * se)

        test_name = "Mann-Whitney U test"
        logging.info(f"\n{display_name}: {test_name} - U={u_stat}, p={p_value:.5f}")

    # Format and log effect size
    logging.info(f"{display_name}: {effect_type} = {effect_size:.3f} [95% CI: {ci_lower:.3f} to {ci_upper:.3f}]")

    # Determine practical significance based on effect size
    significance = interpret_effect_size(effect_size, effect_type)
    logging.info(f"{display_name}: Practical significance - {significance}")

    return {
        "test_name": test_name,
        "p_value": p_value,
        "effect_size": effect_size,
        "effect_type": effect_type,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "is_normal": is_normal,
        "practical_significance": significance
    }


def analyze_count_variable(audit_data, non_audit_data, display_name):
    """
    Analyze count variables using appropriate regression models.

    Args:
        audit_data: List of count values from audit issues
        non_audit_data: List of count values from non-audit issues
        display_name: Name of the attribute for logging

    Returns:
        Dict with test results and effect sizes
    """
    # Calculate zero percentages
    audit_zeros_pct = sum(1 for x in audit_data if x == 0) / len(audit_data) * 100
    non_audit_zeros_pct = sum(1 for x in non_audit_data if x == 0) / len(non_audit_data) * 100

    # Log zero percentages
    logging.info(
        f"{display_name}: {audit_zeros_pct:.1f}% zeros in audit, {non_audit_zeros_pct:.1f}% zeros in non-audit")

    # Define the threshold for excessive zeros (can be adjusted)
    zero_inflation_threshold = 50  # If more than 50% are zeros

    # Check if we have zero-inflation
    is_zero_inflated = (audit_zeros_pct > zero_inflation_threshold or
                        non_audit_zeros_pct > zero_inflation_threshold)

    if is_zero_inflated:
        # Use hurdle model approach for zero-inflated data
        return analyze_zero_inflated_count(audit_data, non_audit_data, display_name)


    # Check for overdispersion to decide between Poisson and Negative Binomial
    audit_mean = np.mean(audit_data)
    audit_var = np.var(audit_data)
    non_audit_mean = np.mean(non_audit_data)
    non_audit_var = np.var(non_audit_data)

    is_overdispersed = (audit_var > 1.5 * audit_mean) or (non_audit_var > 1.5 * non_audit_mean)

    # Create a combined dataset for regression
    group1 = np.ones(len(audit_data))  # 1 for audit group
    group2 = np.zeros(len(non_audit_data))  # 0 for non-audit group

    y = np.concatenate([audit_data, non_audit_data])
    X = np.concatenate([group1, group2]).reshape(-1, 1)

    # Add constant term for intercept
    X_with_const = sm.add_constant(X)

    if is_overdispersed:
        # Use Negative Binomial regression for overdispersed count data
        try:
            # Use NegativeBinomialP instead of GLM to properly estimate alpha
            model = sm.NegativeBinomial(y, X_with_const)
            result = model.fit(disp=0)  # disp=0 suppresses optimization output

            # Log the estimated alpha (dispersion parameter)
            alpha = result.params[-1]
            logging.info(f"{display_name}: Estimated alpha (dispersion) = {alpha:.4f}")

            # Calculate Incidence Rate Ratio (IRR) and confidence intervals
            coef = result.params[1]
            std_err = result.bse[1]
            irr = np.exp(coef)
            irr_ci_lower = np.exp(coef - 1.96 * std_err)
            irr_ci_upper = np.exp(coef + 1.96 * std_err)

            p_value = result.pvalues[1]
            test_name = "Negative Binomial Regression"
            effect_type = "IRR"

        except Exception as e:
            logging.warning(f"Error in Negative Binomial regression for {display_name}: {e}")
            # Fall back to non-parametric test
            return analyze_continuous_variable(audit_data, non_audit_data, display_name, force_nonparametric=True)
    else:
        # Use Poisson regression for non-overdispersed count data
        try:
            model = sm.GLM(y, X_with_const, family=sm.families.Poisson())
            result = model.fit()

            # Calculate Incidence Rate Ratio (IRR) and confidence intervals
            coef = result.params[1]
            std_err = result.bse[1]
            irr = np.exp(coef)
            irr_ci_lower = np.exp(coef - 1.96 * std_err)
            irr_ci_upper = np.exp(coef + 1.96 * std_err)

            p_value = result.pvalues[1]
            test_name = "Poisson Regression"
            effect_type = "IRR"

        except Exception as e:
            logging.warning(f"Error in Poisson regression for {display_name}: {e}")
            # Fall back to non-parametric test
            return analyze_continuous_variable(audit_data, non_audit_data, display_name, force_nonparametric=True)

    # Log results
    logging.info(f"\n{display_name}: {test_name} - p={p_value:.5f}")
    logging.info(f"{display_name}: {effect_type} = {irr:.3f} [95% CI: {irr_ci_lower:.3f} to {irr_ci_upper:.3f}]")

    # Interpret effect size
    if irr > 1:
        relative_change = f"{(irr - 1) * 100:.1f}% higher in audit issues"
    else:
        relative_change = f"{(1 - irr) * 100:.1f}% lower in audit issues"

    logging.info(f"{display_name}: Audit issues have {relative_change} compared to non-audit issues")

    return {
        "test_name": test_name,
        "p_value": p_value,
        "effect_size": irr,
        "effect_type": effect_type,
        "ci_lower": irr_ci_lower,
        "ci_upper": irr_ci_upper,
        "is_overdispersed": is_overdispersed,
        "relative_change": relative_change
    }


def analyze_zero_inflated_count(group1_data, group2_data, display_name):
    """
    Analyze zero-inflated count data using a two-part hurdle model approach.
    This handles count data with excessive zeros by:
    1. Analyzing presence/absence (binary component)
    2. Analyzing magnitude for non-zero values (count component)

    Args:
        group1_data: List of count values from first group (audit/a11y)
        group2_data: List of count values from second group (non-audit/non-a11y)
        display_name: Name of the attribute for logging

    Returns:
        Dict with test results from both components and combined effect
    """
    # Part 1: Analyze the binary presence/absence (logistic component)
    group1_binary = [1 if x > 0 else 0 for x in group1_data]
    group2_binary = [1 if x > 0 else 0 for x in group2_data]

    # Calculate zero percentages (for reference)
    group1_zeros_pct = (1 - np.mean(group1_binary)) * 100
    group2_zeros_pct = (1 - np.mean(group2_binary)) * 100

    # Log overall descriptive statistics (including zeros)
    logging.info(f"\n{display_name} - Overall Descriptive Statistics (including zeros):")
    group1_overall_stats = calculate_numeric_stats(group1_data)
    group2_overall_stats = calculate_numeric_stats(group2_data)
    
    logging.info(f"Group 1 (overall) - Mean: {group1_overall_stats[0]:.2f}, Median: {group1_overall_stats[1]:.2f}, "
                 f"Min: {group1_overall_stats[2]}, Max: {group1_overall_stats[3]}, "
                 f"Std Dev: {group1_overall_stats[4]:.2f}, Count: {group1_overall_stats[5]}")
    logging.info(f"Group 2 (overall) - Mean: {group2_overall_stats[0]:.2f}, Median: {group2_overall_stats[1]:.2f}, "
                 f"Min: {group2_overall_stats[2]}, Max: {group2_overall_stats[3]}, "
                 f"Std Dev: {group2_overall_stats[4]:.2f}, Count: {group2_overall_stats[5]}")

    # Log binary component statistics
    logging.info(f"\n{display_name} - Binary Component Statistics:")
    logging.info(f"Group 1: {group1_zeros_pct:.1f}% zeros, {100-group1_zeros_pct:.1f}% non-zeros")
    logging.info(f"Group 2: {group2_zeros_pct:.1f}% zeros, {100-group2_zeros_pct:.1f}% non-zeros")

    # Fisher's exact test for the binary component
    table = [
        [sum(group1_binary), len(group1_binary) - sum(group1_binary)],
        [sum(group2_binary), len(group2_binary) - sum(group2_binary)]
    ]

    odds_ratio, p_binary = stats.fisher_exact(table)

    logging.info(f"\n{display_name}: Binary component (has any vs. none) - p={p_binary:.5f}")
    logging.info(f"{display_name}: Odds ratio = {odds_ratio:.3f}")

    # Calculate relative likelihood of having non-zero values
    if odds_ratio > 1:
        binary_effect = f"{odds_ratio:.2f}x more likely to have any in group 1"
    else:
        binary_effect = f"{1 / odds_ratio:.2f}x less likely to have any in group 1"

    logging.info(f"{display_name}: {binary_effect}")

    # Part 2: Analyze the non-zero values (truncated count component)
    group1_nonzero = [x for x in group1_data if x > 0]
    group2_nonzero = [x for x in group2_data if x > 0]

    # Always calculate stats for both groups
    group1_nonzero_stats = calculate_numeric_stats(group1_nonzero)
    group2_nonzero_stats = calculate_numeric_stats(group2_nonzero)

    # Log descriptive statistics for non-zero values
    logging.info(f"\n{display_name} - Non-Zero Values Descriptive Statistics:")
    logging.info(f"Group 1 (non-zero only) - Mean: {group1_nonzero_stats[0]:.2f}, Median: {group1_nonzero_stats[1]:.2f}, "
                 f"Min: {group1_nonzero_stats[2]}, Max: {group1_nonzero_stats[3]}, "
                 f"Std Dev: {group1_nonzero_stats[4]:.2f}, Count: {group1_nonzero_stats[5]}")
    logging.info(f"Group 2 (non-zero only) - Mean: {group2_nonzero_stats[0]:.2f}, Median: {group2_nonzero_stats[1]:.2f}, "
                 f"Min: {group2_nonzero_stats[2]}, Max: {group2_nonzero_stats[3]}, "
                 f"Std Dev: {group2_nonzero_stats[4]:.2f}, Count: {group2_nonzero_stats[5]}")

    # Only perform this test if we have enough non-zero data points
    if len(group1_nonzero) >= 5 and len(group2_nonzero) >= 5:
        # Use non-parametric test for the truncated count component
        # This is more robust than parametric tests for this scenario
        stat, p_count = stats.mannwhitneyu(group1_nonzero, group2_nonzero, alternative='two-sided')

        # Calculate effect size (Cliff's delta) for the non-zero values
        try:
            from cliffs_delta import cliffs_delta
            d_effect, _ = cliffs_delta(group1_nonzero, group2_nonzero)
        except ImportError:
            # If cliffs_delta package is not available, calculate it manually
            d_effect = calculate_cliffs_delta(group1_nonzero, group2_nonzero)

        # Interpret magnitude direction
        if np.median(group1_nonzero) > np.median(group2_nonzero):
            count_effect = f"Higher magnitude in group 1 (effect size = {d_effect:.3f})"
        else:
            count_effect = f"Lower magnitude in group 1 (effect size = {d_effect:.3f})"

        logging.info(f"{display_name}: Count component (magnitude when present) - p={p_count:.5f}")
        logging.info(f"{display_name}: {count_effect}")

        has_count_component = True
    else:
        logging.warning(f"{display_name}: Insufficient non-zero data for count component analysis")
        p_count = None
        d_effect = None
        count_effect = "Insufficient data for analysis"
        has_count_component = False

    # Combine results - we need to synthesize the findings from both parts
    if has_count_component:
        # Combine p-values using Fisher's method
        combined_stat = -2 * (np.log(p_binary) + np.log(p_count))
        combined_p = 1 - stats.chi2.cdf(combined_stat, df=4)  # df = 2 * number of tests
        test_name = "Hurdle Model (Binary + Truncated Count)"
    else:
        combined_p = p_binary
        test_name = "Zero-Inflated Model (Binary Component Only)"

    logging.info(f"{display_name}: Combined analysis - p={combined_p:.5f}")

    # Create a descriptive summary of the overall effect
    summary = f"Zero-inflation: {group1_zeros_pct:.1f}% zeros in group 1 vs {group2_zeros_pct:.1f}% in group 2. "
    summary += f"{binary_effect}. "
    if has_count_component:
        summary += f"When present: {count_effect}."

    logging.info(f"{display_name}: Summary: {summary}")

    # Calculate a combined effect size
    # For zero-inflated data, we create a composite effect size that accounts for both
    # the binary component and the count component (when available)
    if has_count_component:
        # Weighted average of binary and count effects
        # Convert odds ratio to a [-1,1] scale for compatibility with Cliff's delta
        normalized_or = 2 * (odds_ratio / (1 + odds_ratio)) - 1
        # Average the binary and count effects (equal weights)
        composite_effect = (normalized_or + d_effect) / 2

        # Calculate confidence intervals for binary component
        # Manual calculation of odds ratio CI since older scipy versions don't support it directly
        a = table[0][0]  # Group 1 with non-zero values
        b = table[0][1]  # Group 1 with zero values
        c = table[1][0]  # Group 2 with non-zero values
        d = table[1][1]  # Group 2 with zero values

        # Calculate log odds ratio
        lor = np.log(odds_ratio)

        # Standard error of log odds ratio
        if a > 0 and b > 0 and c > 0 and d > 0:
            se_lor = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        else:
            # Add small constant to avoid division by zero
            se_lor = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))

        # 95% confidence interval for log odds ratio
        lor_ci_lower = lor - 1.96 * se_lor
        lor_ci_upper = lor + 1.96 * se_lor

        # Convert back to odds ratio scale
        or_ci_lower = np.exp(lor_ci_lower)
        or_ci_upper = np.exp(lor_ci_upper)

        # Normalize the CI bounds to [-1,1] scale
        normalized_or_lower = 2 * (or_ci_lower / (1 + or_ci_lower)) - 1
        normalized_or_upper = 2 * (or_ci_upper / (1 + or_ci_upper)) - 1

        # For the count component, use bootstrapping to estimate CI for Cliff's delta
        # For simplicity, we'll use an approximation based on the effect size
        count_ci_lower = d_effect - 0.2  # Approximate 95% CI
        count_ci_upper = d_effect + 0.2  # Approximate 95% CI

        # Combine the CIs using the same weighting as the effect size
        ci_lower = (normalized_or_lower + count_ci_lower) / 2
        ci_upper = (normalized_or_upper + count_ci_upper) / 2
    else:
        # If we only have the binary component, normalize the odds ratio to [-1,1]
        normalized_or = 2 * (odds_ratio / (1 + odds_ratio)) - 1
        composite_effect = normalized_or

        # Calculate confidence intervals for binary component only using manual calculation
        a = table[0][0]  # Group 1 with non-zero values
        b = table[0][1]  # Group 1 with zero values
        c = table[1][0]  # Group 2 with non-zero values
        d = table[1][1]  # Group 2 with zero values

        # Calculate log odds ratio
        lor = np.log(odds_ratio)

        # Standard error of log odds ratio
        if a > 0 and b > 0 and c > 0 and d > 0:
            se_lor = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        else:
            # Add small constant to avoid division by zero
            se_lor = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))

        # 95% confidence interval for log odds ratio
        lor_ci_lower = lor - 1.96 * se_lor
        lor_ci_upper = lor + 1.96 * se_lor

        # Convert back to odds ratio scale
        or_ci_lower = np.exp(lor_ci_lower)
        or_ci_upper = np.exp(lor_ci_upper)

        # Normalize the CI bounds to [-1,1] scale
        ci_lower = 2 * (or_ci_lower / (1 + or_ci_lower)) - 1
        ci_upper = 2 * (or_ci_upper / (1 + or_ci_upper)) - 1

    # Return comprehensive results including separate descriptive stats with flexible keys
    return {
        "test_name": test_name,
        "p_value": combined_p,
        "effect_size": composite_effect,  # Add the main effect_size key
        "ci_lower": ci_lower,  # Add lower confidence interval
        "ci_upper": ci_upper,  # Add upper confidence interval
        "binary_p_value": p_binary,
        "count_p_value": p_count if has_count_component else None,
        "effect_type": "Hurdle Model",
        "binary_effect_size": odds_ratio,
        "count_effect_size": d_effect if has_count_component else None,
        "binary_effect_desc": binary_effect,
        "count_effect_desc": count_effect if has_count_component else None,
        "audit_zeros_pct": group1_zeros_pct,  # Keep original keys for compatibility
        "non_audit_zeros_pct": group2_zeros_pct,
        "summary": summary,
        # Add separate descriptive statistics with flexible keys
        "overall_stats": {
            "audit": group1_overall_stats,
            "non_audit": group2_overall_stats,
            "a11y": group1_overall_stats,  # Also store with a11y key for RQ3
            "non_a11y": group2_overall_stats  # Also store with non_a11y key for RQ3
        },
        "nonzero_stats": {
            "audit": group1_nonzero_stats,
            "non_audit": group2_nonzero_stats,
            "a11y": group1_nonzero_stats,  # Also store with a11y key for RQ3
            "non_a11y": group2_nonzero_stats  # Also store with non_a11y key for RQ3
        }
    }
    
    
def analyze_binary_variable(audit_data, non_audit_data, display_name):
    """
    Analyze binary variables using chi-squared or Fisher's exact test.

    Args:
        audit_data: List of binary values (or values to be treated as binary) from audit issues
        non_audit_data: List of binary values from non-audit issues
        display_name: Name of the attribute for logging

    Returns:
        Dict with test results and effect sizes
    """
    # If data is not already binary, convert it (e.g., presence/absence)
    # Here we assume any non-zero value is treated as "present"
    audit_binary = [1 if x > 0 else 0 for x in audit_data]
    non_audit_binary = [1 if x > 0 else 0 for x in non_audit_data]

    # Create contingency table
    audit_present = sum(audit_binary)
    audit_absent = len(audit_binary) - audit_present
    non_audit_present = sum(non_audit_binary)
    non_audit_absent = len(non_audit_binary) - non_audit_present

    contingency_table = np.array([[audit_present, audit_absent],
                                  [non_audit_present, non_audit_absent]])

    # Determine which test to use based on expected frequencies
    use_fisher = np.min(contingency_table) < 5

    if use_fisher:
        # Use Fisher's exact test for small expected frequencies
        odds_ratio, p_value = fisher_exact(contingency_table)
        test_name = "Fisher's exact test"
    else:
        # Use chi-squared test for larger samples
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        odds_ratio = (audit_present * non_audit_absent) / (audit_absent * non_audit_present) if (
                                                                                                            audit_absent * non_audit_present) > 0 else np.inf
        test_name = "Chi-squared test"

    # Calculate phi coefficient for effect size
    n = np.sum(contingency_table)
    phi = np.sqrt(chi2 / n) if not use_fisher else np.sqrt((odds_ratio - 1) ** 2 / (odds_ratio + 1) ** 2)

    # Calculate 95% CI for odds ratio
    if odds_ratio < np.inf:
        log_odds = np.log(odds_ratio)
        se_log_odds = np.sqrt(1 / audit_present + 1 / audit_absent + 1 / non_audit_present + 1 / non_audit_absent)
        ci_lower = np.exp(log_odds - 1.96 * se_log_odds)
        ci_upper = np.exp(log_odds + 1.96 * se_log_odds)
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    # Log results
    logging.info(f"\n{display_name}: {test_name} - p={p_value:.5f}")
    logging.info(f"{display_name}: Odds Ratio = {odds_ratio:.3f} [95% CI: {ci_lower:.3f} to {ci_upper:.3f}]")
    logging.info(f"{display_name}: Phi Coefficient = {phi:.3f}")

    # Interpret practical significance
    significance = interpret_effect_size(phi, "Phi")
    logging.info(f"{display_name}: Practical significance - {significance}")

    # Calculate proportion in each group
    audit_proportion = audit_present / len(audit_binary)
    non_audit_proportion = non_audit_present / len(non_audit_binary)

    # Calculate absolute and relative differences
    abs_diff = audit_proportion - non_audit_proportion
    rel_diff = (audit_proportion / non_audit_proportion - 1) * 100 if non_audit_proportion > 0 else np.inf

    logging.info(
        f"{display_name}: Proportion in audit issues: {audit_proportion:.2f}, in non-audit issues: {non_audit_proportion:.2f}")
    logging.info(f"{display_name}: Absolute difference: {abs_diff:.2f}, Relative difference: {rel_diff:.1f}%")

    return {
        "test_name": test_name,
        "p_value": p_value,
        "effect_size": phi,
        "effect_type": "Phi Coefficient",
        "odds_ratio": odds_ratio,
        "or_ci_lower": ci_lower,
        "or_ci_upper": ci_upper,
        "practical_significance": significance,
        "audit_proportion": audit_proportion,
        "non_audit_proportion": non_audit_proportion,
        "absolute_difference": abs_diff,
        "relative_difference": rel_diff
    }


def analyze_ordinal_variable(audit_data, non_audit_data, display_name):
    """
    Analyze ordinal variables using appropriate non-parametric tests.

    Args:
        audit_data: List of ordinal values from audit issues
        non_audit_data: List of ordinal values from non-audit issues
        display_name: Name of the attribute for logging

    Returns:
        Dict with test results and effect sizes
    """
    # For ordinal data, use Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(audit_data, non_audit_data, alternative='two-sided')

    # Calculate Cliff's Delta effect size for ordinal data
    effect_size = calculate_cliffs_delta(audit_data, non_audit_data)

    # Approximate 95% CI for Cliff's Delta
    n1, n2 = len(audit_data), len(non_audit_data)
    se = np.sqrt((2 * (n1 + n2 + 1)) / (3 * n1 * n2))
    ci_lower = max(-1, effect_size - 1.96 * se)
    ci_upper = min(1, effect_size + 1.96 * se)

    # Log results
    test_name = "Mann-Whitney U test"
    effect_type = "Cliff's Delta"
    logging.info(f"\n{display_name}: {test_name} - U={u_stat}, p={p_value:.5f}")
    logging.info(f"{display_name}: {effect_type} = {effect_size:.3f} [95% CI: {ci_lower:.3f} to {ci_upper:.3f}]")

    # Interpret practical significance
    significance = interpret_effect_size(effect_size, effect_type)
    logging.info(f"{display_name}: Practical significance - {significance}")

    return {
        "test_name": test_name,
        "p_value": p_value,
        "effect_size": effect_size,
        "effect_type": effect_type,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "practical_significance": significance
    }


def calculate_cliffs_delta(x, y):
    """
    Calculate Cliff's Delta effect size for non-parametric data.

    Cliff's Delta measures the probability that a value from one group
    is greater than a value from another group, minus the reverse probability.

    Args:
        x: Values from first group
        y: Values from second group

    Returns:
        Cliff's Delta value in range [-1, 1]
    """
    # Count comparisons where x > y, x < y, and x == y
    greater = 0
    less = 0
    equal = 0

    for i in x:
        for j in y:
            if i > j:
                greater += 1
            elif i < j:
                less += 1
            else:
                equal += 1

    # Calculate Cliff's Delta
    total_comparisons = len(x) * len(y)
    delta = (greater - less) / total_comparisons

    return delta


def interpret_effect_size(effect_size, effect_type):
    """
    Interpret the practical significance of effect sizes.

    Args:
        effect_size: Calculated effect size value
        effect_type: Type of effect size measure

    Returns:
        String describing practical significance
    """
    effect_size = abs(effect_size)  # Use absolute value for interpretation

    if effect_type == "Cohen's d":
        if effect_size < 0.2:
            return "Negligible effect"
        elif effect_size < 0.5:
            return "Small effect"
        elif effect_size < 0.8:
            return "Medium effect"
        else:
            return "Large effect"

    elif effect_type == "Cliff's Delta":
        if effect_size < 0.147:
            return "Negligible effect"
        elif effect_size < 0.33:
            return "Small effect"
        elif effect_size < 0.474:
            return "Medium effect"
        else:
            return "Large effect"

    elif effect_type == "Phi Coefficient":
        if effect_size < 0.1:
            return "Negligible effect"
        elif effect_size < 0.3:
            return "Small effect"
        elif effect_size < 0.5:
            return "Medium effect"
        else:
            return "Large effect"

    elif effect_type == "IRR":
        # For IRR, we interpret in terms of % difference from 1
        percent_diff = abs(effect_size - 1) * 100
        if percent_diff < 10:
            return "Negligible effect"
        elif percent_diff < 50:
            return "Small effect"
        elif percent_diff < 100:
            return "Medium effect"
        else:
            return "Large effect"

    else:
        return "Unknown effect size measure"


def apply_multiple_testing_correction(results, attributes):
    """
    Apply Benjamini-Hochberg correction to control false discovery rate.

    Args:
        results: Dict of results from statistical tests
        attributes: List of attributes for reference
    """
    # Extract p-values and corresponding attribute names
    p_values = []
    attr_names = []

    for attr_tuple in attributes:
        # Adapt to either 3-value or 4-value tuples
        if len(attr_tuple) == 4:  # Format: (a11y_key, non_a11y_key, display_name, data_type)
            attr_key = attr_tuple[0]  # Use a11y_key as attr_key
            display_name = attr_tuple[2]
        else:  # Format: (attr_key, display_name, data_type)
            attr_key = attr_tuple[0]
            display_name = attr_tuple[1]

        if attr_key in results and "p_value" in results[attr_key]:
            p_values.append(results[attr_key]["p_value"])
            attr_names.append(display_name)

    # Apply BH correction
    if p_values:
        _, adjusted_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        # Log adjusted p-values and updated significance
        logging.info("\n\nMultiple Testing Correction (Benjamini-Hochberg):")
        logging.info("=" * 50)

        for i, (attr_name, p_orig, p_adj) in enumerate(zip(attr_names, p_values, adjusted_p)):
            significance = "Significant" if p_adj < 0.05 else "Not significant"
            logging.info(f"{attr_name}: Original p={p_orig:.5f}, Adjusted p={p_adj:.5f}, {significance}")

            # Update results with adjusted p-values - same pattern as above
            for attr_tuple in attributes:
                # Adapt to either 3-value or 4-value tuples
                if len(attr_tuple) == 4:  # Format: (a11y_key, non_a11y_key, display_name, data_type)
                    attr_key = attr_tuple[0]  # Use a11y_key as attr_key
                    disp_name = attr_tuple[2]
                else:  # Format: (attr_key, display_name, data_type)
                    attr_key = attr_tuple[0]
                    disp_name = attr_tuple[1]

                if disp_name == attr_name and attr_key in results:
                    results[attr_key]["adjusted_p_value"] = p_adj
                    results[attr_key]["significant_after_correction"] = (p_adj < 0.05)
    else:
        logging.warning("No p-values available for multiple testing correction")


def extract_resolution_times(issues):
    """Extract resolution times in days from issues."""
    return [
        (datetime.fromisoformat(issue['resolved_date']) - datetime.fromisoformat(issue['created_date'])).days
        for issue in issues
        if issue.get('created_date') and issue.get('resolved_date')
    ]


def extract_version_counts(issues, version_key, filter_zeros=False):
    """Extract version counts from issues."""
    counts = [len(issue.get(version_key, [])) for issue in issues]

    if filter_zeros:
        counts = [count for count in counts if count > 0]

    return counts


def extract_description_counts(issues):
    """Extract description word counts from issues."""
    counts = [len(issue.get("description", []) or []) for issue in issues]
    return [count for count in counts if count > 0]


def extract_numeric_attribute(issues, attr_key, filter_zeros=False):
    """Extract numeric attribute values from issues."""
    values = [issue[attr_key] for issue in issues if issue.get(attr_key) is not None]

    if filter_zeros:
        values = [value for value in values if value > 0]

    return values


def extract_priority_values(issues):
    """
    Extract priority values from issues and convert them to numeric values for analysis.

    Moodle uses the following priority levels (from highest to lowest):
    - Blocker: Blocks development or testing, prevents Moodle from running
    - Critical: Crashes server, loss of data, severe memory leak
    - Major: Major loss of function, incorrect output
    - Minor: Minor loss of function where workaround is possible
    - Trivial: Cosmetic problem like misspelt words or misaligned text

    Returns:
        List of numeric priority values (higher number = higher priority)
    """
    # Define mapping based on Moodle's specific priority system
    priority_mapping = {
        "blocker": 5,
        "critical": 4,
        "major": 3,
        "minor": 2,
        "trivial": 1
    }

    # Extract priority values
    values = []
    for issue in issues:
        priority = issue.get('priority')
        if priority is not None:
            # If priority is a string, convert to lowercase for mapping
            if isinstance(priority, str):
                priority_value = priority_mapping.get(priority.lower(), None)
                if priority_value is not None:
                    values.append(priority_value)
            # If priority is already numeric (1-5), use as is
            elif isinstance(priority, (int, float)) and 1 <= priority <= 5:
                values.append(priority)

    return values


def summarize_ordinal_attribute(values):
    """Summarize an ordinal attribute with basic statistics."""
    if not values:
        return (0, 0, 0, 0, 0, 0)  # Empty stats

    mean = np.mean(values)
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    std_dev = np.std(values, ddof=1)
    count = len(values)

    return (mean, median, min_val, max_val, std_dev, count)


def RQ3_2(last_five_years=False):
    """
    Analyzes and compares attributes of a11y and non-a11y issues with balanced sampling.
    Performs appropriate statistical tests based on data type and distribution.
    Includes effect size calculations, multiple testing correction, and bootstrapping for imbalanced datasets.
    """
    # Load and prepare data
    combined_a11y_issues = load_json(combined_a11y_file)
    resolved_a11y_issues = filter_resolved_issues(combined_a11y_issues)
    cutoff_date = datetime(2019, 11, 19)

    if last_five_years:
        # Filter a11y issues by creation date >= 2019-11-19
        resolved_a11y_issues = filter_by_date_range_start_only(resolved_a11y_issues, cutoff_date)

    non_a11y_issues = load_jsonl(non_a11y_issue_file_jsonl)
    resolved_non_a11y_issues = filter_resolved_issues(non_a11y_issues)

    if last_five_years:
        # Filter non-a11y issues by creation date >= 2019-11-19
        resolved_non_a11y_issues = filter_by_date_range_start_only(resolved_non_a11y_issues, cutoff_date)

    # Log original dataset sizes after filtering
    logging.info(f"Number of a11y issues (after date filter): {len(resolved_a11y_issues)}")
    logging.info(f"Number of non-a11y issues (after date filter): {len(resolved_non_a11y_issues)}")

    # Define attributes to compare with their display names and data types
    # Format: (a11y_key, non_a11y_key, display_name, data_type)
    attributes = [
        ('time_logged', 'logged_time', "Time Logged", "continuous"),
        ('resolution_time', 'resolution_time', "Resolution Time", "continuous"),
        ('fix_versions', 'fix_versions', "Fix Versions", "count"),
        ('affected_versions', 'affected_versions', "Affected Versions", "count"),
        ('num_participants', 'num_participants', "Participants", "count"),
        ('num_comments', 'num_comments', "Comments", "count"),
        ('num_commits', 'num_commits', "Commits", "count"),
        ('num_watchers', 'num_watchers', "Watchers", "count"),
        ('num_votes', 'num_votes', "Votes", "count"),
        ('num_attachments', 'num_attachments', "Attachments", "count"),
        ('description', 'num_description', "Description Words", "count"),
        ('priority', 'priority', "Priority", "ordinal")
    ]

    # Compare basic descriptive statistics
    compare_descriptive_stats(resolved_a11y_issues, resolved_non_a11y_issues, attributes)

    # Perform bootstrapped balanced statistical analysis
    results = perform_balanced_statistical_analysis(resolved_a11y_issues, resolved_non_a11y_issues, attributes)

    # Apply multiple testing correction
    apply_multiple_testing_correction(results, attributes)

    # Create visualizations
    create_visualizations(resolved_a11y_issues, resolved_non_a11y_issues, attributes)

    # Save results to RQ3 directory
    save_results_to_file(results, attributes, resolved_a11y_issues, resolved_non_a11y_issues, directory="RQ/RQ3")


def compare_descriptive_stats(a11y_issues, non_a11y_issues, attributes):
    """
    Compare and log basic descriptive statistics for all attributes.
    """
    for a11y_key, non_a11y_key, display_name, _ in attributes:
        # Extract data for both groups based on attribute type
        if a11y_key == 'resolution_time':
            a11y_stats = calculate_time_to_resolve(a11y_issues)
            non_a11y_stats = calculate_time_to_resolve(non_a11y_issues)
        elif a11y_key in ['fix_versions', 'affected_versions']:
            a11y_stats = summarize_versions(a11y_issues, a11y_key)
            non_a11y_stats = summarize_versions(non_a11y_issues, non_a11y_key)
        elif a11y_key == 'description':
            a11y_stats = summarize_description_words(a11y_issues, a11y_key)
            non_a11y_stats = summarize_numeric_attribute(non_a11y_issues, non_a11y_key, zero_value=False)
        elif a11y_key == 'priority':
            a11y_values = extract_priority_values(a11y_issues)
            non_a11y_values = extract_priority_values(non_a11y_issues)
            a11y_stats = summarize_ordinal_attribute(a11y_values)
            non_a11y_stats = summarize_ordinal_attribute(non_a11y_values)
        elif a11y_key == 'time_logged':
            a11y_stats = summarize_numeric_attribute(a11y_issues, a11y_key)
            non_a11y_stats = summarize_numeric_attribute(non_a11y_issues, non_a11y_key)
        else:
            remove_outliers = a11y_key != 'num_commits'
            a11y_stats = summarize_numeric_attribute(a11y_issues, a11y_key, remove_outliers)
            non_a11y_stats = summarize_numeric_attribute(non_a11y_issues, non_a11y_key, remove_outliers)

        # Log the statistics
        log_stats(display_name, a11y_stats, non_a11y_stats)


def perform_balanced_statistical_analysis(a11y_issues, non_a11y_issues, attributes, num_bootstraps=1000):
    """
    Perform statistical analysis using bootstrapping to balance sample sizes.

    Args:
        a11y_issues: List of a11y issues
        non_a11y_issues: List of non-a11y issues
        attributes: List of attributes to compare
        num_bootstraps: Number of bootstrap iterations

    Returns:
        Dict of results with p-values and effect sizes
    """
    logging.info(f"\nPerforming balanced statistical analysis with {num_bootstraps} bootstrap iterations")

    # Determine sample sizes
    a11y_size = len(a11y_issues)

    # Limit non-a11y sample size for computational efficiency if very large
    max_non_a11y_sample = min(len(non_a11y_issues), 10000)
    if len(non_a11y_issues) > max_non_a11y_sample:
        logging.info(f"Using a random sample of {max_non_a11y_sample} non-a11y issues for computational efficiency")
        non_a11y_sample = random.sample(non_a11y_issues, max_non_a11y_sample)
    else:
        non_a11y_sample = non_a11y_issues

    # Initialize storage for bootstrap results
    results = {}

    # Perform bootstrap analysis for each attribute
    for a11y_key, non_a11y_key, display_name, data_type in attributes:
        bootstrap_results = []

        for i in range(num_bootstraps):
            # Use all a11y issues
            bootstrap_a11y = a11y_issues

            # Random sample of non-a11y issues to match a11y count
            balanced_non_a11y = random.sample(non_a11y_sample, min(a11y_size, len(non_a11y_sample)))

            # Extract attribute data based on data type
            a11y_data, non_a11y_data = extract_attribute_data(
                bootstrap_a11y, balanced_non_a11y, a11y_key, non_a11y_key, data_type)

            # Skip if not enough data
            if len(a11y_data) < 5 or len(non_a11y_data) < 5:
                continue

            # Perform statistical test based on data type
            result = analyze_attribute(a11y_data, non_a11y_data, display_name, data_type)

            # Store result for this bootstrap iteration
            bootstrap_results.append(result)

        # Skip if no valid results
        if not bootstrap_results:
            logging.warning(f"Insufficient data for {display_name}, skipping analysis")
            continue

        # Average the results across all bootstrap iterations
        avg_result = average_bootstrap_results(bootstrap_results, data_type)
        avg_result["bootstrap_iterations"] = len(bootstrap_results)

        # Log and store results
        log_statistical_result(display_name, avg_result)
        results[a11y_key] = avg_result

    return results


def extract_attribute_data(a11y_issues, non_a11y_issues, a11y_key, non_a11y_key, data_type):
    """
    Extract attribute data for analysis based on the attribute type.

    Args:
        a11y_issues: List of a11y issues
        non_a11y_issues: List of non-a11y issues
        a11y_key: Key for the attribute in a11y issues
        non_a11y_key: Key for the attribute in non-a11y issues
        data_type: Type of data (continuous, count, binary, ordinal)

    Returns:
        Tuple of (a11y_data, non_a11y_data)
    """
    if a11y_key == 'resolution_time':
        a11y_data = extract_resolution_times(a11y_issues)
        non_a11y_data = extract_resolution_times(non_a11y_issues)

    elif a11y_key in ['fix_versions', 'affected_versions']:
        filter_zeros = a11y_key == 'fix_versions'
        a11y_data = extract_version_counts(a11y_issues, a11y_key, filter_zeros)
        non_a11y_data = extract_version_counts(non_a11y_issues, non_a11y_key, filter_zeros)

    elif a11y_key == 'description':
        a11y_data = extract_description_counts(a11y_issues)
        non_a11y_data = [issue.get(non_a11y_key, 0) for issue in non_a11y_issues
                      if issue.get(non_a11y_key) is not None and issue.get(non_a11y_key) > 0]

    elif a11y_key == 'priority':
        a11y_data = extract_priority_values(a11y_issues)
        non_a11y_data = extract_priority_values(non_a11y_issues)

    elif a11y_key == 'time_logged':
        a11y_data = [issue.get(a11y_key, 0) for issue in a11y_issues if issue.get(a11y_key) is not None]
        non_a11y_data = [issue.get(non_a11y_key, 0) for issue in non_a11y_issues
                       if issue.get(non_a11y_key) is not None]

    else:
        filter_zeros = a11y_key == "num_commits"
        a11y_data = extract_numeric_attribute(a11y_issues, a11y_key, filter_zeros)
        non_a11y_data = extract_numeric_attribute(non_a11y_issues, non_a11y_key, filter_zeros)

    return a11y_data, non_a11y_data


def analyze_attribute(a11y_data, non_a11y_data, display_name, data_type):
    """
    Analyze attribute data using the appropriate statistical test.

    Args:
        a11y_data: List of values from a11y issues
        non_a11y_data: List of values from non-a11y issues
        display_name: Name of the attribute for logging
        data_type: Type of data (continuous, count, binary, ordinal)

    Returns:
        Dict with test results and effect sizes
    """
    if data_type == "continuous":
        return analyze_continuous_variable(a11y_data, non_a11y_data, display_name)
    elif data_type == "count":
        return analyze_count_variable(a11y_data, non_a11y_data, display_name)
    elif data_type == "binary":
        return analyze_binary_variable(a11y_data, non_a11y_data, display_name)
    elif data_type == "ordinal":
        return analyze_ordinal_variable(a11y_data, non_a11y_data, display_name)
    else:
        return analyze_continuous_variable(a11y_data, non_a11y_data, display_name, force_nonparametric=True)


def average_bootstrap_results(bootstrap_results, data_type):
    """
    Average the results from multiple bootstrap iterations.

    Args:
        bootstrap_results: List of result dictionaries from bootstrap iterations
        data_type: Type of data being analyzed (continuous, count, etc.)

    Returns:
        Dict with averaged values
    """
    # Initialize result with consistent keys
    avg_result = {
        "test_name": bootstrap_results[0]["test_name"],
        "p_value": np.mean([r["p_value"] for r in bootstrap_results]),
        "effect_type": bootstrap_results[0]["effect_type"],
        "data_type": data_type
    }

    # Add keys specific to each data type
    if data_type == "continuous" or data_type == "ordinal":
        avg_result["effect_size"] = np.mean([r["effect_size"] for r in bootstrap_results])
        avg_result["ci_lower"] = np.mean([r["ci_lower"] for r in bootstrap_results])
        avg_result["ci_upper"] = np.mean([r["ci_upper"] for r in bootstrap_results])

        if "practical_significance" in bootstrap_results[0]:
            # Count frequency of each significance level
            significance_counts = {}
            for r in bootstrap_results:
                sig = r["practical_significance"]
                significance_counts[sig] = significance_counts.get(sig, 0) + 1

            # Use the most common significance level
            avg_result["practical_significance"] = max(significance_counts.items(), key=lambda x: x[1])[0]

    elif data_type == "count":
        avg_result["effect_size"] = np.mean([r["effect_size"] for r in bootstrap_results])
        avg_result["ci_lower"] = np.mean([r["ci_lower"] for r in bootstrap_results])
        avg_result["ci_upper"] = np.mean([r["ci_upper"] for r in bootstrap_results])

        # Check if this is a hurdle model (has binary_p_value)
        if "binary_p_value" in bootstrap_results[0]:
            # This is a hurdle model - preserve the detailed components
            avg_result["binary_p_value"] = np.mean([r["binary_p_value"] for r in bootstrap_results])
            
            # Handle count_p_value (might be None for some iterations)
            count_p_values = [r["count_p_value"] for r in bootstrap_results if r["count_p_value"] is not None]
            if count_p_values:
                avg_result["count_p_value"] = np.mean(count_p_values)
            else:
                avg_result["count_p_value"] = None
            
            # Average binary and count effect sizes
            avg_result["binary_effect_size"] = np.mean([r["binary_effect_size"] for r in bootstrap_results])
            
            count_effect_sizes = [r["count_effect_size"] for r in bootstrap_results if r["count_effect_size"] is not None]
            if count_effect_sizes:
                avg_result["count_effect_size"] = np.mean(count_effect_sizes)
            else:
                avg_result["count_effect_size"] = None
            
            # Use the most common descriptions (take from first result for simplicity)
            avg_result["binary_effect_desc"] = bootstrap_results[0]["binary_effect_desc"]
            avg_result["count_effect_desc"] = bootstrap_results[0]["count_effect_desc"]
            
            # Average zero percentages
            avg_result["audit_zeros_pct"] = np.mean([r["audit_zeros_pct"] for r in bootstrap_results])
            avg_result["non_audit_zeros_pct"] = np.mean([r["non_audit_zeros_pct"] for r in bootstrap_results])
            
            # Use the first summary as template (they should be similar)
            avg_result["summary"] = bootstrap_results[0]["summary"]
            
        else:
            # Regular count model (Poisson/Negative Binomial)
            # Determine relative change direction based on mean effect size
            mean_effect = avg_result["effect_size"]
            if mean_effect > 1:
                relative_change = f"{(mean_effect - 1) * 100:.1f}% higher in a11y issues"
            else:
                relative_change = f"{(1 - mean_effect) * 100:.1f}% lower in a11y issues"

            avg_result["relative_change"] = relative_change

    elif data_type == "binary":
        avg_result["effect_size"] = np.mean([r["effect_size"] for r in bootstrap_results])

        # Handle potential infinite values in odds ratios
        valid_ors = [r["odds_ratio"] for r in bootstrap_results
                     if not np.isnan(r["odds_ratio"]) and not np.isinf(r["odds_ratio"])]
        valid_ci_lowers = [r["or_ci_lower"] for r in bootstrap_results
                        if not np.isnan(r["or_ci_lower"]) and not np.isinf(r["or_ci_lower"])]
        valid_ci_uppers = [r["or_ci_upper"] for r in bootstrap_results
                        if not np.isnan(r["or_ci_upper"]) and not np.isinf(r["or_ci_upper"])]

        if valid_ors:
            avg_result["odds_ratio"] = np.mean(valid_ors)
        else:
            avg_result["odds_ratio"] = float('inf')

        if valid_ci_lowers:
            avg_result["or_ci_lower"] = np.mean(valid_ci_lowers)
        else:
            avg_result["or_ci_lower"] = np.nan

        if valid_ci_uppers:
            avg_result["or_ci_upper"] = np.mean(valid_ci_uppers)
        else:
            avg_result["or_ci_upper"] = np.nan

    return avg_result


def log_statistical_result(display_name, result):
    """Log the statistical test result in a readable format."""
    logging.info(f"\n{display_name} - Bootstrap Analysis Result:")
    logging.info(f"  Test: {result['test_name']}")
    logging.info(f"  p-value: {result['p_value']:.5f}")

    if result["data_type"] == "continuous" or result["data_type"] == "ordinal":
        logging.info(f"  {result['effect_type']}: {result['effect_size']:.3f} [95% CI: {result['ci_lower']:.3f} to {result['ci_upper']:.3f}]")
        if "practical_significance" in result:
            logging.info(f"  Practical significance: {result['practical_significance']}")

    elif result["data_type"] == "count":
        logging.info(f"  {result['effect_type']}: {result['effect_size']:.3f} [95% CI: {result['ci_lower']:.3f} to {result['ci_upper']:.3f}]")
        
        # Check if this is a hurdle model
        if "binary_p_value" in result:
            # This is a hurdle model - log the detailed components
            logging.info(f"  Binary component p-value: {result['binary_p_value']:.5f}")
            if result.get('count_p_value') is not None:
                logging.info(f"  Count component p-value: {result['count_p_value']:.5f}")
            logging.info(f"  Binary effect: {result.get('binary_effect_desc', 'N/A')}")
            logging.info(f"  Count effect: {result.get('count_effect_desc', 'N/A')}")
            if "summary" in result:
                logging.info(f"  Summary: {result['summary']}")
        else:
            # Regular count model (Poisson/Negative Binomial)
            if "relative_change" in result:
                logging.info(f"  {result['relative_change']}")

    elif result["data_type"] == "binary":
        logging.info(f"  {result['effect_type']}: {result['effect_size']:.3f}")
        if not np.isinf(result.get('odds_ratio', float('inf'))):
            logging.info(f"  Odds Ratio: {result['odds_ratio']:.3f} [95% CI: {result['or_ci_lower']:.3f} to {result['or_ci_upper']:.3f}]")
        else:
            logging.info(f"  Odds Ratio: Infinite (perfect separation in data)")


def save_results_to_file(results, attributes, a11y_issues, non_a11y_issues, directory="RQ/RQ3"):
    """
    Save statistical analysis results and descriptive statistics to file in specified directory.
    Creates directories if they don't exist.

    Args:
        results: Dict of results from statistical tests
        attributes: List of attributes to compare
        a11y_issues: List of a11y issues
        non_a11y_issues: List of non-a11y issues
        directory: Target directory for saving results
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Prepare results for serialization
    serializable_results = {}
    descriptive_stats = {}

    # Process statistical test results
    for attr_key, result in results.items():
        # Convert numpy values to Python native types for JSON serialization
        cleaned_result = {}
        for k, v in result.items():
            if isinstance(v, (np.integer, np.floating)):
                cleaned_result[k] = float(v)
            elif isinstance(v, np.ndarray):
                cleaned_result[k] = v.tolist()
            elif isinstance(v, (bool, str, int, float, list, dict, type(None))):
                cleaned_result[k] = v
            else:
                cleaned_result[k] = str(v)
        serializable_results[attr_key] = cleaned_result

    # Calculate and add descriptive statistics for each attribute
    for attr_tuple in attributes:
        # Determine attribute structure
        if len(attr_tuple) == 4:  # Format: (a11y_key, non_a11y_key, display_name, data_type)
            a11y_key = attr_tuple[0]
            non_a11y_key = attr_tuple[1]
            display_name = attr_tuple[2]
            data_type = attr_tuple[3]
        else:  # Format: (attr_key, display_name, data_type)
            a11y_key = attr_tuple[0]
            non_a11y_key = attr_tuple[0]
            display_name = attr_tuple[1]
            data_type = attr_tuple[2]

        # Check if this variable used hurdle model
        is_hurdle_model = (a11y_key in results and 
                          results[a11y_key].get("test_name", "").startswith("Hurdle Model"))

        # Extract statistics based on attribute type
        if a11y_key == 'resolution_time':
            a11y_stats = calculate_time_to_resolve(a11y_issues)
            non_a11y_stats = calculate_time_to_resolve(non_a11y_issues)
        elif a11y_key in ['fix_versions', 'affected_versions']:
            a11y_stats = summarize_versions(a11y_issues, a11y_key)
            non_a11y_stats = summarize_versions(non_a11y_issues, non_a11y_key)
        elif a11y_key == 'description':
            a11y_stats = summarize_description_words(a11y_issues, a11y_key)
            if "3" in directory:
                non_a11y_stats = summarize_numeric_attribute(non_a11y_issues, non_a11y_key, zero_value=False)
            else:
                non_a11y_stats = summarize_description_words(non_a11y_issues, non_a11y_key)
        elif a11y_key == 'priority':
            a11y_values = extract_priority_values(a11y_issues)
            non_a11y_values = extract_priority_values(non_a11y_issues)
            a11y_stats = summarize_ordinal_attribute(a11y_values)
            non_a11y_stats = summarize_ordinal_attribute(non_a11y_values)
        elif a11y_key == 'time_logged':
            a11y_stats = summarize_numeric_attribute(a11y_issues, a11y_key)
            non_a11y_stats = summarize_numeric_attribute(non_a11y_issues, non_a11y_key)
        else:
            a11y_stats = summarize_numeric_attribute(a11y_issues, a11y_key)
            non_a11y_stats = summarize_numeric_attribute(non_a11y_issues, non_a11y_key)

        # Format stats for JSON serialization
        if isinstance(a11y_stats, tuple) and len(a11y_stats) >= 6:
            a11y_formatted = {
                "mean": float(a11y_stats[0]),
                "median": float(a11y_stats[1]),
                "min": float(a11y_stats[2]),
                "max": float(a11y_stats[3]),
                "std_dev": float(a11y_stats[4]),
                "count": int(a11y_stats[5])
            }
        else:
            a11y_formatted = {"error": "Statistics not available"}

        if isinstance(non_a11y_stats, tuple) and len(non_a11y_stats) >= 6:
            non_a11y_formatted = {
                "mean": float(non_a11y_stats[0]),
                "median": float(non_a11y_stats[1]),
                "min": float(non_a11y_stats[2]),
                "max": float(non_a11y_stats[3]),
                "std_dev": float(non_a11y_stats[4]),
                "count": int(non_a11y_stats[5])
            }
        else:
            non_a11y_formatted = {"error": "Statistics not available"}

        # For hurdle model variables, add separate overall and non-zero statistics
        if is_hurdle_model:
            # Get the detailed stats from the hurdle model results
            overall_stats = results[a11y_key].get("overall_stats", {})
            nonzero_stats = results[a11y_key].get("nonzero_stats", {})
            
            # Format overall stats
            if "audit" in overall_stats and len(overall_stats["audit"]) >= 6:
                audit_overall_formatted = {
                    "mean": float(overall_stats["audit"][0]),
                    "median": float(overall_stats["audit"][1]),
                    "min": float(overall_stats["audit"][2]),
                    "max": float(overall_stats["audit"][3]),
                    "std_dev": float(overall_stats["audit"][4]),
                    "count": int(overall_stats["audit"][5])
                }
            else:
                audit_overall_formatted = a11y_formatted
                
            if "non_audit" in overall_stats and len(overall_stats["non_audit"]) >= 6:
                non_audit_overall_formatted = {
                    "mean": float(overall_stats["non_audit"][0]),
                    "median": float(overall_stats["non_audit"][1]),
                    "min": float(overall_stats["non_audit"][2]),
                    "max": float(overall_stats["non_audit"][3]),
                    "std_dev": float(overall_stats["non_audit"][4]),
                    "count": int(overall_stats["non_audit"][5])
                }
            else:
                non_audit_overall_formatted = non_a11y_formatted
            
            # Format non-zero stats
            if "audit" in nonzero_stats and len(nonzero_stats["audit"]) >= 6:
                audit_nonzero_formatted = {
                    "mean": float(nonzero_stats["audit"][0]),
                    "median": float(nonzero_stats["audit"][1]),
                    "min": float(nonzero_stats["audit"][2]),
                    "max": float(nonzero_stats["audit"][3]),
                    "std_dev": float(nonzero_stats["audit"][4]),
                    "count": int(nonzero_stats["audit"][5])
                }
            else:
                audit_nonzero_formatted = {"error": "Statistics not available"}
                
            if "non_audit" in nonzero_stats and len(nonzero_stats["non_audit"]) >= 6:
                non_audit_nonzero_formatted = {
                    "mean": float(nonzero_stats["non_audit"][0]),
                    "median": float(nonzero_stats["non_audit"][1]),
                    "min": float(nonzero_stats["non_audit"][2]),
                    "max": float(nonzero_stats["non_audit"][3]),
                    "std_dev": float(nonzero_stats["non_audit"][4]),
                    "count": int(nonzero_stats["non_audit"][5])
                }
            else:
                non_audit_nonzero_formatted = {"error": "Statistics not available"}

            descriptive_stats[display_name] = {
                "overall": {
                    "a11y": audit_overall_formatted,
                    "non_a11y": non_audit_overall_formatted
                },
                "non_zero_only": {
                    "a11y": audit_nonzero_formatted,
                    "non_a11y": non_audit_nonzero_formatted
                },
                "data_type": data_type,
                "model_type": "hurdle_model"
            }
        else:
            # Regular format for non-hurdle model variables
            descriptive_stats[display_name] = {
                "a11y": a11y_formatted,
                "non_a11y": non_a11y_formatted,
                "data_type": data_type
            }

    # Create a results summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attributes_analyzed": [attr_tuple[2] if len(attr_tuple) == 4 else attr_tuple[1] for attr_tuple in attributes],
        "a11y_issues_count": len(a11y_issues),
        "non_a11y_issues_count": len(non_a11y_issues),
        "bootstrap_iterations": results[list(results.keys())[0]].get("bootstrap_iterations",
                                                                     "N/A") if results else "N/A",
        "statistical_results": serializable_results,
        "descriptive_statistics": descriptive_stats
    }

    # Save to file with timestamp
    filename = f"{directory}/a11y_vs_nona11y_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Results saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving results to file: {e}")

        # Try an alternative location if the directory might be the issue
        alt_filename = f"a11y_vs_nona11y_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(alt_filename, 'w') as f:
                json.dump(summary, f, indent=2)
            logging.info(f"Results saved to alternative location: {alt_filename}")
        except Exception as e2:
            logging.error(f"Failed to save results to alternative location: {e2}")


def create_visualizations(a11y_issues, non_a11y_issues, attributes):
    """
    Create visualizations comparing a11y and non-a11y issues.
    Reuses existing visualization function.
    """
    # Create directory structure
    os.makedirs('RQ/RQ3/Figures', exist_ok=True)

    # # Use existing visualization function for resolution time comparisons
    # draw_combined_histogram_and_boxplot(a11y_issues, non_a11y_issues)

    # Create priority distribution visualization if priority data exists
    create_priority_distribution_chart(a11y_issues, non_a11y_issues)


def create_priority_distribution_chart(a11y_issues, non_a11y_issues):
    """
    Create a bar chart showing the distribution of priority levels.
    """
    # Extract priority values
    a11y_data = extract_priority_values(a11y_issues)
    non_a11y_data = extract_priority_values(non_a11y_issues)

    if not a11y_data or not non_a11y_data:
        logging.info("Insufficient priority data for visualization")
        return

    # Count frequency of each priority level
    a11y_counts = [0, 0, 0, 0, 0]  # For priorities 1-5
    for p in a11y_data:
        if 1 <= p <= 5:
            a11y_counts[p-1] += 1

    non_a11y_counts = [0, 0, 0, 0, 0]  # For priorities 1-5
    for p in non_a11y_data:
        if 1 <= p <= 5:
            non_a11y_counts[p-1] += 1

    # Convert to percentages
    a11y_pct = [100 * count / len(a11y_data) for count in a11y_counts]
    non_a11y_pct = [100 * count / len(non_a11y_data) for count in non_a11y_counts]

    # Create bar chart
    plt.figure(figsize=(12, 6))
    priority_labels = ['Trivial', 'Minor', 'Major', 'Critical', 'Blocker']
    x = np.arange(len(priority_labels))
    width = 0.35

    plt.bar(x - width/2, a11y_pct, width, label='A11y')
    plt.bar(x + width/2, non_a11y_pct, width, label='Non-A11y')

    plt.xlabel('Priority Level')
    plt.ylabel('Percentage (%)')
    plt.title('Priority Distribution: A11y vs. Non-A11y Issues')
    plt.xticks(x, priority_labels)
    plt.legend()

    priority_chart_path = "RQ/RQ3/Figures/priority_distribution.png"
    plt.savefig(priority_chart_path)
    plt.close()

    logging.info(f"Priority distribution chart saved to {priority_chart_path}")


def log_percentile_breakdown(times, label):
    """
    Compute and log a set of percentiles for the given resolution times.
    """
    if not times:
        logging.info(f"No data available for {label} issues.")
        return

    percentiles_to_check = [25, 50, 75, 90, 95, 99]
    calculated_percentiles = np.percentile(times, percentiles_to_check)

    logging.info(f"{label} Resolution Time - Percentile Breakdown:")
    for p, val in zip(percentiles_to_check, calculated_percentiles):
        logging.info(f"  {p}th percentile: {val:.2f} days")


def filter_by_issue_key(resolved_audit_issues, non_audit_issues):
    """
    Filters out entries in non_audit_issues that have issue_keys found in resolved_audit_issues.

    Args:
        resolved_audit_issues (list[dict]): A list of dictionaries, each containing an "issue_key" field.
        non_audit_issues (list[dict]): A list of dictionaries, each containing an "issue_key" field.

    Returns:
        list[dict]: Filtered list of dictionaries from non_audit_issues where issue_keys
                    are not found in resolved_audit_issues.
    """
    # Extract the set of issue_keys from resolved_audit_issues
    resolved_keys = {entry["issue_key"] for entry in resolved_audit_issues}

    # Filter out entries in non_audit_issues whose issue_key is in resolved_keys
    filtered_non_audit_issues = [
        issue for issue in non_audit_issues if issue["issue_key"] not in resolved_keys
    ]

    return filtered_non_audit_issues


def RQ1_non_a11y_analysis(input_file):
    """
    Analyze non-accessibility issues and create trend figures
    
    Args:
        input_file (str): Path to the input JSONL file
    """
    logger = logging.getLogger(__name__)
    
    # Load issues from JSONL file
    issues = load_jsonl(input_file)
    
    if not issues:
        logger.error("No issues loaded. Exiting analysis.")
        return
    
    # Filter out Epics and Tasks (using 'type' field from JSONL structure)
    filtered_issues = [
        issue for issue in issues
        if issue.get("type") not in ["Epic", "Task"]
    ]
    
    logger.info(f"Filtered non-a11y issues: {len(filtered_issues)} (excluded Epics and Tasks)")
    
    # Process creation trends
    creation_years = []
    for issue in filtered_issues:
        created_date = issue.get("created_date")
        if created_date:
            # Extract year from date string (format: "2024-12-13T07:07:51.000+0000")
            year = created_date[:4]
            creation_years.append(year)
    
    creation_counts = Counter(creation_years)
    
    # Process resolution trends (only for Fixed issues)
    resolution_years = []
    for issue in filtered_issues:
        if issue.get("resolution_type") == "Fixed" and issue.get("resolved_date"):
            resolved_date = issue.get("resolved_date")
            if resolved_date:  # Check if not null
                year = resolved_date[:4]
                resolution_years.append(year)
    
    resolution_counts = Counter(resolution_years)
    
    # Sort the data by year and convert years to integers
    creation_years_sorted = sorted(creation_counts.keys())
    creation_counts_sorted = [creation_counts[year] for year in creation_years_sorted]
    
    resolution_years_sorted = sorted(resolution_counts.keys())
    resolution_counts_sorted = [resolution_counts[year] for year in resolution_years_sorted]
    
    # Cast years to integers for numeric plotting
    creation_years_int = [int(year) for year in creation_years_sorted]
    resolution_years_int = [int(year) for year in resolution_years_sorted]
    
    # Log statistics
    total_issues_created = sum(creation_counts.values())
    total_issues_resolved = sum(resolution_counts.values())
    logger.info(f"Total non-a11y issues created across all years: {total_issues_created}")
    logger.info(f"Total non-a11y issues resolved (Fixed) across all years: {total_issues_resolved}")
    logger.info(f"Non-a11y creation years range: {min(creation_years_int) if creation_years_int else 'N/A'} - {max(creation_years_int) if creation_years_int else 'N/A'}")
    logger.info(f"Non-a11y resolution years range: {min(resolution_years_int) if resolution_years_int else 'N/A'} - {max(resolution_years_int) if resolution_years_int else 'N/A'}")
    
    # Set style with larger, bolder text
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    
    # Draw bar chart for creation trends with semantic colors
    if creation_years_int and creation_counts_sorted:
        plt.figure(figsize=(10, 6))
        # Red/orange bars for issue creation (problems arising)
        bars = plt.bar(creation_years_int, creation_counts_sorted, 
                      color='#E74C3C', alpha=0.8, width=0.8)
        # Dark brown trend line for contrast
        plt.plot(creation_years_int, creation_counts_sorted, 
                'o-', color='#8B4513', linewidth=2, markersize=4, label="Trend")
        
        plt.title("Trend of Non-Accessibility Issues Created", fontsize=16, pad=20, fontweight='bold')
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Issues Created", fontsize=14, fontweight='bold')
        plt.xticks(creation_years_int, rotation=45, fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(False)  # No grid to match reference
        
        # Set y-axis to start from 0
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig("RQ/RQ1/Figures/non_a11y_created_issues.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Non-a11y created issues trend chart saved to RQ/RQ1/Figures/non_a11y_created_issues.png")
    else:
        logger.warning("No non-a11y creation data available for plotting")
    
    # Draw bar chart for resolution trends with semantic colors
    if resolution_years_int and resolution_counts_sorted:
        plt.figure(figsize=(10, 6))
        # Green bars for issue resolution (problems solved)
        bars = plt.bar(resolution_years_int, resolution_counts_sorted, 
                      color='#27AE60', alpha=0.8, width=0.8)
        # Dark blue trend line for contrast
        plt.plot(resolution_years_int, resolution_counts_sorted, 
                'o-', color='#1F618D', linewidth=2, markersize=4, label="Trend")
        
        plt.title("Trend of Non-Accessibility Issues Resolved (Fixed)", fontsize=16, pad=20, fontweight='bold')
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Issues Resolved", fontsize=14, fontweight='bold')
        plt.xticks(resolution_years_int, rotation=45, fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(False)  # No grid to match reference
        
        # Set y-axis to start from 0
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig("RQ/RQ1/Figures/non_a11y_resolved_issues.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Non-a11y resolved issues trend chart saved to RQ/RQ1/Figures/non_a11y_resolved_issues.png")
    else:
        logger.warning("No non-a11y resolution data available for plotting")
    
    # Save summary statistics for non-a11y
    summary_stats = {
        'type': 'non_accessibility_issues',
        'total_issues_loaded': len(issues),
        'total_issues_filtered': len(filtered_issues),
        'total_issues_created': total_issues_created,
        'total_issues_resolved_fixed': total_issues_resolved,
        'creation_year_range': f"{min(creation_years_int) if creation_years_int else 'N/A'} - {max(creation_years_int) if creation_years_int else 'N/A'}",
        'resolution_year_range': f"{min(resolution_years_int) if resolution_years_int else 'N/A'} - {max(resolution_years_int) if resolution_years_int else 'N/A'}",
        'creation_by_year': dict(zip(creation_years_sorted, creation_counts_sorted)),
        'resolution_by_year': dict(zip(resolution_years_sorted, resolution_counts_sorted))
    }
    
    # Save summary to JSON file
    try:
        with open('RQ/RQ1/non_a11y_summary_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info("Non-a11y summary statistics saved to RQ/RQ1/non_a11y_summary_statistics.json")
    except Exception as e:
        logger.error(f"Error saving non-a11y summary statistics: {e}")


def RQ1_a11y_analysis(input_file):
    """
    Analyze accessibility issues and create trend figures
    
    Args:
        input_file (str): Path to the input JSONL file
    """
    logger = logging.getLogger(__name__)
    
    # Load issues from JSONL file
    issues = load_json(input_file)
    
    if not issues:
        logger.error("No issues loaded. Exiting analysis.")
        return
    
    # Filter for accessibility issues only (assuming accessibility issues are identified by specific criteria)
    # You may need to adjust this filter based on how accessibility issues are identified in your data
    # For example, if there's a specific field or keyword that identifies accessibility issues
    filtered_issues = [
        issue for issue in issues
        if issue.get("type") not in ["Epic", "Task"]  # Exclude Epics and Tasks like the original
        # Add additional filtering criteria for accessibility issues here if needed
        # e.g., if issue.get("labels") and any("accessibility" in label.lower() for label in issue.get("labels", []))
    ]
    
    logger.info(f"Filtered a11y issues: {len(filtered_issues)} (excluded Epics and Tasks)")
    
    # Process creation trends
    creation_years = []
    for issue in filtered_issues:
        created_date = issue.get("created_date")
        if created_date:
            # Extract year from date string (format: "2024-12-13T07:07:51.000+0000")
            year = created_date[:4]
            creation_years.append(year)
    
    creation_counts = Counter(creation_years)
    
    # Process resolution trends (only for Fixed issues)
    resolution_years = []
    for issue in filtered_issues:
        if issue.get("resolution_type") == "Fixed" and issue.get("resolved_date"):
            resolved_date = issue.get("resolved_date")
            if resolved_date:  # Check if not null
                year = resolved_date[:4]
                resolution_years.append(year)
    
    resolution_counts = Counter(resolution_years)
    
    # Sort the data by year and convert years to integers
    creation_years_sorted = sorted(creation_counts.keys())
    creation_counts_sorted = [creation_counts[year] for year in creation_years_sorted]
    
    resolution_years_sorted = sorted(resolution_counts.keys())
    resolution_counts_sorted = [resolution_counts[year] for year in resolution_years_sorted]
    
    # Cast years to integers for numeric plotting
    creation_years_int = [int(year) for year in creation_years_sorted]
    resolution_years_int = [int(year) for year in resolution_years_sorted]
    
    # Define accessibility audit periods
    audit_years = [2006, 2012]  # Single-year audits
    audit_period_start = 2020   # Multi-year audit start
    audit_period_end = 2023     # Multi-year audit end
    audit_period_years = list(range(audit_period_start, audit_period_end + 1))  # All years in the period
    
    # Log statistics
    total_issues_created = sum(creation_counts.values())
    total_issues_resolved = sum(resolution_counts.values())
    logger.info(f"Total a11y issues created across all years: {total_issues_created}")
    logger.info(f"Total a11y issues resolved (Fixed) across all years: {total_issues_resolved}")
    logger.info(f"A11y creation years range: {min(creation_years_int) if creation_years_int else 'N/A'} - {max(creation_years_int) if creation_years_int else 'N/A'}")
    logger.info(f"A11y resolution years range: {min(resolution_years_int) if resolution_years_int else 'N/A'} - {max(resolution_years_int) if resolution_years_int else 'N/A'}")
    
    # Set style with larger, bolder text
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    
    # Draw bar chart for creation trends with different colors for audit years
    if creation_years_int and creation_counts_sorted:
        plt.figure(figsize=(12, 6))
        
        # Create color list based on whether year is an audit year
        bar_colors = []
        for year in creation_years_int:
            if year in audit_years:
                bar_colors.append('#9B59B6')  # Purple for single-year audits
            elif year in audit_period_years:
                bar_colors.append('#8E44AD')  # Darker purple for multi-year audit period
            else:
                bar_colors.append('#E74C3C')  # Red/orange for regular years
        
        # Create bars with different colors
        bars = plt.bar(creation_years_int, creation_counts_sorted, 
                      color=bar_colors, alpha=0.8, width=0.8, edgecolor='black', linewidth=1)
        
        # Dark brown trend line
        plt.plot(creation_years_int, creation_counts_sorted, 
                'o-', color='#8B4513', linewidth=2, markersize=4)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', alpha=0.8, label='Regular Years'),
            Patch(facecolor='#9B59B6', alpha=0.8, label='Single-Year Audits (2006, 2012)'),
            Patch(facecolor='#8E44AD', alpha=0.8, label='Multi-Year Audit Period (2020-2023)'),
            plt.Line2D([0], [0], color='#8B4513', linewidth=2, marker='o', label='Trend')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.title("Trend of Accessibility Issues Created", fontsize=16, pad=20, fontweight='bold')
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Issues Created", fontsize=14, fontweight='bold')
        plt.xticks(creation_years_int, rotation=45, fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(False)
        
        # Set y-axis to start from 0
        plt.ylim(bottom=0, top=max(creation_counts_sorted) * 1.1 if creation_counts_sorted else 1)
        
        plt.tight_layout()
        plt.savefig("RQ/RQ1/Figures/a11y_created_issues.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("A11y created issues trend chart saved to RQ/RQ1/Figures/a11y_created_issues.png")
    else:
        logger.warning("No a11y creation data available for plotting")
    
    # Draw bar chart for resolution trends with different colors for audit years
    if resolution_years_int and resolution_counts_sorted:
        plt.figure(figsize=(12, 6))
        
        # Create color list based on whether year is an audit year
        bar_colors = []
        for year in resolution_years_int:
            if year in audit_years:
                bar_colors.append('#9B59B6')  # Purple for single-year audits
            elif year in audit_period_years:
                bar_colors.append('#8E44AD')  # Darker purple for multi-year audit period
            else:
                bar_colors.append('#27AE60')  # Green for regular years
        
        # Create bars with different colors
        bars = plt.bar(resolution_years_int, resolution_counts_sorted, 
                      color=bar_colors, alpha=0.8, width=0.8, edgecolor='black', linewidth=1)
        
        # Dark blue trend line
        plt.plot(resolution_years_int, resolution_counts_sorted, 
                'o-', color='#1F618D', linewidth=2, markersize=4)
        
        # Add legend - positioned in upper right instead of outside
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27AE60', alpha=0.8, label='Regular Years'),
            Patch(facecolor='#9B59B6', alpha=0.8, label='Single-Year Audits (2006, 2012)'),
            Patch(facecolor='#8E44AD', alpha=0.8, label='Multi-Year Audit Period (2020-2023)'),
            plt.Line2D([0], [0], color='#1F618D', linewidth=2, marker='o', label='Trend')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.title("Trend of Accessibility Issues Resolved", fontsize=16, pad=20, fontweight='bold')
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Issues Resolved", fontsize=14, fontweight='bold')
        plt.xticks(resolution_years_int, rotation=45, fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(False)
        
        # Set y-axis to start from 0
        plt.ylim(bottom=0, top=max(resolution_counts_sorted) * 1.1 if resolution_counts_sorted else 1)
        
        plt.tight_layout()
        plt.savefig("RQ/RQ1/Figures/a11y_resolved_issues.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("A11y resolved issues trend chart saved to RQ/RQ1/Figures/a11y_resolved_issues.png")
    else:
        logger.warning("No a11y resolution data available for plotting")
    
    # Save summary statistics for a11y (updated to include audit information)
    summary_stats = {
        'type': 'accessibility_issues',
        'total_issues_loaded': len(issues),
        'total_issues_filtered': len(filtered_issues),
        'total_issues_created': total_issues_created,
        'total_issues_resolved_fixed': total_issues_resolved,
        'creation_year_range': f"{min(creation_years_int) if creation_years_int else 'N/A'} - {max(creation_years_int) if creation_years_int else 'N/A'}",
        'resolution_year_range': f"{min(resolution_years_int) if resolution_years_int else 'N/A'} - {max(resolution_years_int) if resolution_years_int else 'N/A'}",
        'creation_by_year': dict(zip(creation_years_sorted, creation_counts_sorted)),
        'resolution_by_year': dict(zip(resolution_years_sorted, resolution_counts_sorted)),
        'accessibility_audits': {
            'single_year_audits': audit_years,
            'multi_year_audit_period': f"{audit_period_start}-{audit_period_end}"
        }
    }
    
    # Save summary to JSON file
    try:
        with open('RQ/RQ1/a11y_summary_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info("A11y summary statistics saved to RQ/RQ1/a11y_summary_statistics.json")
    except Exception as e:
        logger.error(f"Error saving a11y summary statistics: {e}")
        
        

def filter_by_date_range_start_only(issues, start_date):
    """Filter issues based on their creation date being >= start_date."""
    start_date = start_date.replace(tzinfo=None)  # Ensure offset-naive
    return [
        issue for issue in issues
        if datetime.fromisoformat(issue['created_date']).replace(tzinfo=None) >= start_date
    ]
    

def calculate_hurdle_descriptive_stats(group1_data, group2_data, variable_name):
    """
    Calculate descriptive statistics for hurdle model variables.
    Returns both overall stats (including zeros) and non-zero only stats.
    
    Args:
        group1_data: List of values from first group (audit/a11y)
        group2_data: List of values from second group (non-audit/non-a11y) 
        variable_name: Name of the variable for logging
        
    Returns:
        Dict with formatted statistics for JSON output
    """
    # Calculate overall stats (including zeros)
    group1_overall = calculate_numeric_stats(group1_data)
    group2_overall = calculate_numeric_stats(group2_data)
    
    # Extract non-zero values
    group1_nonzero = [x for x in group1_data if x > 0]
    group2_nonzero = [x for x in group2_data if x > 0]
    
    # Calculate non-zero stats
    group1_nonzero_stats = calculate_numeric_stats(group1_nonzero)
    group2_nonzero_stats = calculate_numeric_stats(group2_nonzero)
    
    # Format for JSON output
    def format_stats(stats_tuple):
        if len(stats_tuple) >= 6:
            return {
                "mean": float(stats_tuple[0]),
                "median": float(stats_tuple[1]), 
                "min": float(stats_tuple[2]),
                "max": float(stats_tuple[3]),
                "std_dev": float(stats_tuple[4]),
                "count": int(stats_tuple[5])
            }
        else:
            return {"error": "Statistics calculation failed"}
    
    # Return formatted results
    return {
        "overall": {
            "group1": format_stats(group1_overall),
            "group2": format_stats(group2_overall)
        },
        "non_zero_only": {
            "group1": format_stats(group1_nonzero_stats),
            "group2": format_stats(group2_nonzero_stats)
        },
        "zero_percentages": {
            "group1": (len(group1_data) - len(group1_nonzero)) / len(group1_data) * 100 if group1_data else 0,
            "group2": (len(group2_data) - len(group2_nonzero)) / len(group2_data) * 100 if group2_data else 0
        }
    }


def add_hurdle_stats_to_json(results_dict, issues_group1, issues_group2, variable_key, variable_display_name):
    """
    Add hurdle model descriptive statistics to an existing results dictionary.
    
    Args:
        results_dict: The dictionary where results should be added
        issues_group1: List of issues from first group
        issues_group2: List of issues from second group  
        variable_key: The key to extract data (e.g., 'num_votes')
        variable_display_name: Display name for the variable (e.g., 'Votes')
    """
    # Extract the raw data
    group1_data = [issue.get(variable_key, 0) for issue in issues_group1 if issue.get(variable_key) is not None]
    group2_data = [issue.get(variable_key, 0) for issue in issues_group2 if issue.get(variable_key) is not None]
    
    # Calculate hurdle stats
    hurdle_stats = calculate_hurdle_descriptive_stats(group1_data, group2_data, variable_display_name)
    
    # Add to results dictionary
    results_dict[variable_display_name] = {
        "overall": {
            "a11y": hurdle_stats["overall"]["group1"],
            "non_a11y": hurdle_stats["overall"]["group2"]
        },
        "non_zero_only": {
            "a11y": hurdle_stats["non_zero_only"]["group1"], 
            "non_a11y": hurdle_stats["non_zero_only"]["group2"]
        },
        "data_type": "count",
        "model_type": "hurdle_model",
        "zero_info": hurdle_stats["zero_percentages"]
    }
    
    return results_dict




if __name__ == "__main__":
    RQ1_a11y_analysis(output_file)
    RQ1_non_a11y_analysis(non_a11y_issue_file_jsonl)
    RQ2_2()
    RQ3_2(False)
    RQ3_2(True)