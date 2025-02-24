# novel-ae-features-two-stage-mu
This repository contains a Python script for analyzing acoustic emission data from an milling machine, specifically focusing on detecting tool wear. The script performs several key operations:

1. **Feature Extraction:** Calculates various time-domain features (e.g., Mean, Standard Deviation, IQR, etc.) from the input data, which is expected to be pre-processed time series data.
2. **Segmentation:** Divides the feature time series into overlapping segments.
3. **Statistical Testing:**  Applies the Mann-Whitney U test to compare each segment to a baseline reference segment. This helps in identifying segments where the wear significantly differ from the baseline. The Anderson-Darling and Kolmogorov-Smirnov tests are also included.
4. **Wear Detection:** Uses a thresholding technique on the combined anomaly score to identify segments with significant deviations, indicating possible anomalies.
5. **Performance Evaluation:** Calculates key performance metrics (AUC, Accuracy, Precision, Recall, F1-Score) for both the proposed method and individual features to evaluate the model's ability to detect anomalies.


**Dependencies:**

*   NumPy
*   SciPy
*   Matplotlib
*   Antropy
*   Scikit-learn


**Usage:**

The script requires pre-processed time-series data as input. Modify the relevant parts of the script to provide your data and adjust parameters such as segment length.  The output is a set of visualisations and performance metrics.


**Files:**

*   `features.py`: The script for feature extraction.
*   `proposed_test.py`: The script for proposed two-stage Mann-Whitney U test.
*   `single_test.py`: The script for standalone statistical tests.
*   `evaluate.py`: The script for accuracy performance evaluation.
