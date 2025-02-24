from scipy.stats import mannwhitneyu

# Function to perform statistical tests using the Mann-Whitney U test
def perform_statistical_tests_mannwhitney(reference_segment, segments):
    stats_results = {}
    for feature in reference_segment.keys():
        ref_segment = np.array(reference_segment[feature])
        results = []
        for i in range(len(segments[feature])):
            seg_data = np.array(segments[feature][i])
            # Check if either of the arrays is empty
            if len(ref_segment) > 0 and len(seg_data) > 0:
                u_stat, p_value = mannwhitneyu(ref_segment, seg_data, alternative='two-sided')
                results.append({
                    'U-Statistic': u_stat,
                    'P-Value': p_value
                })
            else:
                results.append({
                    'U-Statistic': None,
                    'P-Value': None
                })
        stats_results[feature] = results
    return stats_results

# STAGE 1
# Segment length
segment_length = 150
num_segments = len(features['Mean']) // segment_length*4

# Split features into segments
segmented_features = {}
for feature in features.keys():
    segmented_features[feature] = [features[feature][i * segment_length//4: i * segment_length//4 + segment_length]
                                   for i in range(num_segments)]

# Perform statistical tests
reference_segment = {feature: segmented_features[feature][0] for feature in segmented_features.keys()}
segments = {feature: segmented_features[feature][1:] for feature in segmented_features.keys()}

stats_results = perform_statistical_tests_mannwhitney(reference_segment, segments)

# STAGE 2
# Define segment length and initialize lists
segment_length = 150
num_features = len(stats_results.keys())

# Reference segment (first segment)
ref_segment = {feature: np.array([result['U-Statistic'] for result in stats_results[feature][0:segment_length]]) for feature in stats_results.keys()}

# Initialize a list to store U-values for each segment
all_u_values = []

# Iterate over the segments starting from the second segment
for i in range(80):
    # Initialize list to store U-values for each feature
    feature_u_values = []

    # Perform Mann-Whitney U test for each feature separately
    for feature in stats_results.keys():
        ref_feature_data = ref_segment[feature]
        current_feature_data = np.array([result['U-Statistic'] for result in stats_results[feature][i*segment_length//4:i*segment_length//4+segment_length]])

        # Perform Mann-Whitney U test and get the U-value
        u_stat, _ = mannwhitneyu(ref_feature_data, current_feature_data, alternative='two-sided')
        feature_u_values.append(u_stat)

    # Store the U-values for this segment
    all_u_values.append(feature_u_values)

# Convert results to a numpy array for easier handling, handling None values
all_u_values = np.array(all_u_values, dtype=np.float64)
all_u_values[np.isnan(all_u_values)] = 0  # Replace None values with 0 for simplicity

# Normalize U-values for each feature across all segments
u_values_normalized = np.zeros_like(all_u_values)
for feature_idx in range(num_features):
    feature_data = all_u_values[:, feature_idx]
    if np.max(feature_data) - np.min(feature_data) != 0:  # Avoid division by zero
        u_values_normalized[:, feature_idx] = (feature_data - np.min(feature_data)) / (np.max(feature_data) - np.min(feature_data))
