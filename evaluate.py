from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Ground truth labels (assuming first half is healthy, second half is faulty)
true_labels = np.concatenate([np.zeros(len(u_values_smoothed) // 2), np.ones(len(u_values_smoothed) // 2)])

# Threshold the U-values to get predicted labels
threshold = 0.5
predicted_labels_proposal = (u_values_smoothed < threshold).astype(int)

# Calculate metrics for the proposal
auc_proposal = roc_auc_score(true_labels, u_values_smoothed)
accuracy_proposal = accuracy_score(true_labels, predicted_labels_proposal)
precision_proposal = precision_score(true_labels, predicted_labels_proposal)
recall_proposal = recall_score(true_labels, predicted_labels_proposal)
f1_proposal = f1_score(true_labels, predicted_labels_proposal)

print("Proposal Metrics:")
print(f"AUC: {auc_proposal:.3f}")
print(f"Accuracy: {accuracy_proposal:.3f}")
print(f"Precision: {precision_proposal:.3f}")
print(f"Recall: {recall_proposal:.3f}")
print(f"F1-Score: {f1_proposal:.3f}")

# Calculate metrics for each individual feature in u_values_normalized
for feature_idx in range(u_values_normalized.shape[1]):
    predicted_labels_feature = (u_values_normalized[:, feature_idx] > threshold).astype(int)
    auc_feature = roc_auc_score(true_labels, u_values_normalized[:, feature_idx])
    accuracy_feature = accuracy_score(true_labels, predicted_labels_feature)
    precision_feature = precision_score(true_labels, predicted_labels_feature)
    recall_feature = recall_score(true_labels, predicted_labels_feature)
    f1_feature = f1_score(true_labels, predicted_labels_feature)

    print(f"\nFeature {feature_idx + 1} Metrics:")
    print(f"AUC: {auc_feature:.3f}")
    print(f"Accuracy: {accuracy_feature:.3f}")
    print(f"Precision: {precision_feature:.3f}")
    print(f"Recall: {recall_feature:.3f}")
    print(f"F1-Score: {f1_feature:.3f}")

# Calculate metrics for each individual feature in u_values_normalized
for feature_idx in range(u_values_normalized.shape[1]):
    predicted_labels_feature = (u_values_normalized[:, feature_idx] < threshold).astype(int)
    auc_feature = roc_auc_score(true_labels, u_values_normalized[:, feature_idx])
    accuracy_feature = accuracy_score(true_labels, predicted_labels_feature)
    precision_feature = precision_score(true_labels, predicted_labels_feature)
    recall_feature = recall_score(true_labels, predicted_labels_feature)
    f1_feature = f1_score(true_labels, predicted_labels_feature)

    print(f"\nFeature {feature_idx + 1} Metrics:")
    print(f"AUC: {auc_feature:.3f}")
    print(f"Accuracy: {accuracy_feature:.3f}")
    print(f"Precision: {precision_feature:.3f}")
    print(f"Recall: {recall_feature:.3f}")
    print(f"F1-Score: {f1_feature:.3f}")
