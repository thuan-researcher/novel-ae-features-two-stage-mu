import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp, ks_2samp, mannwhitneyu

# Function for Anderson-Darling, Kolmogorov-Smirnov, and Mann-Whitney U tests
def perform_statistical_tests(reference_segment, segments):
    stats_results = {}
    for feature in reference_segment.keys():
        ref_segment = np.array(reference_segment[feature])
        results = []
        for i in range(len(segments[feature])):
            seg_data = np.array(segments[feature][i])
            if len(ref_segment) > 0 and len(seg_data) > 0:
                # Anderson-Darling Test
                ad_stat, _, ad_crit_values = anderson_ksamp([ref_segment, seg_data])
                
                # Kolmogorov-Smirnov Test
                ks_stat, ks_p_value = ks_2samp(ref_segment, seg_data)
                
                # Mann-Whitney U Test
                u_stat, u_p_value = mannwhitneyu(ref_segment, seg_data, alternative='two-sided')
                
                results.append({
                    'Anderson-Darling Statistic': ad_stat,
                    'Kolmogorov-Smirnov Statistic': ks_stat,
                    'Kolmogorov-Smirnov P-Value': ks_p_value,
                    'Mann-Whitney U Statistic': u_stat,
                    'Mann-Whitney U P-Value': u_p_value
                })
            else:
                results.append({
                    'Anderson-Darling Statistic': None,
                    'Kolmogorov-Smirnov Statistic': None,
                    'Kolmogorov-Smirnov P-Value': None,
                    'Mann-Whitney U Statistic': None,
                    'Mann-Whitney U P-Value': None
                })
        stats_results[feature] = results
    return stats_results

# Perform statistical tests
stats_results = perform_statistical_tests(reference_segment, segments)

# Function for smoothing using a moving window with overlap
def smooth_values(values, window=150, overlap=0.75):
    step = int(window * (1 - overlap))
    smoothed = [np.mean(values[i:i + window]) for i in range(0, len(values) - window + 1, step)]
    return smoothed
