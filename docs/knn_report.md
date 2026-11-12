# k-Nearest Neighbors (k-NN) — Experimental Report

## 1. Objective

The objective of these experiments was to implement and evaluate a k-Nearest Neighbors (k-NN) classifier for two classification tasks: image classification and sentiment classification. The experiments investigated the effects of neighborhood size, distance metric, feature normalization, and distance-weighted voting on classification accuracy.

## 2. Datasets

Two datasets were used:

| Dataset | Total Samples | Features | Classes |
|---|---:|---:|---:|
| Image | 20,000 | 500 | 50 |
| Sentiment | 12,000 | 768 | 3 |

Each dataset contains an identifier column, a class label, and numerical embedding features.

Both datasets were divided into training and validation subsets using an 80/20 stratified split with `random_state=42`.

| Dataset | Training Samples | Validation Samples |
|---|---:|---:|
| Image | 16,000 | 4,000 |
| Sentiment | 9,600 | 2,400 |

The same split was maintained throughout the experiments to ensure comparability.

## 3. Implementation

The k-NN classifier was implemented from scratch using NumPy rather than relying on scikit-learn's k-NN implementation.

The classifier supports:

- Euclidean and Manhattan distance metrics.
- Uniform majority voting.
- Inverse-distance-weighted voting.
- Configurable neighborhood size (k).
- Batch prediction to manage memory consumption.
- Deterministic tie-breaking using the smallest class label.

Scikit-learn was used for stratified data splitting and accuracy evaluation.

## 4. Experimental Methodology

The following hyperparameters and preprocessing approaches were investigated:

**Neighborhood size:** Multiple values of k were tested. Image classification experiments considered k = 1, 3, 5, 10, and 20. Sentiment experiments initially considered k = 1, 3, 5, 10, 20, 30, and 50, followed by additional values up to 200 and a refined search near the best-performing configuration.

**Distance metrics:** Euclidean and Manhattan distances were compared for image classification. Euclidean distance was used for sentiment classification.

**Voting strategies:** Uniform voting and inverse-distance-weighted voting were evaluated.

**Preprocessing:** Original embeddings and L2-normalized embeddings were evaluated.

Classification accuracy on the validation subset was used as the primary performance metric.

A total of 69 experimental configurations were evaluated.

## 5. Image Classification Results

The best-performing image classification configuration was:

| Parameter | Selected Value |
|---|---|
| Neighborhood size | k = 1 |
| Distance metric | Euclidean |
| Normalization | None |
| Voting | Uniform |
| Validation accuracy | **53.08%** |

Image classification accuracy generally decreased as k increased. Distance-weighted voting improved performance for k greater than 1 but did not exceed the k=1 result.

Manhattan distance produced lower validation accuracy than Euclidean distance across the tested configurations.

L2 normalization did not produce a meaningful improvement over the original image embeddings.

These findings suggest that, among the configurations tested, image classification benefited most from retaining the nearest individual training example rather than aggregating information from a larger neighborhood.

## 6. Sentiment Classification Results

The best-performing sentiment classification configuration was:

| Parameter | Selected Value |
|---|---|
| Neighborhood size | k = 80 |
| Distance metric | Euclidean |
| Normalization | None |
| Voting | Inverse-distance weighted |
| Validation accuracy | **59.08%** |

### Neighborhood Size Analysis

Selected results for distance-weighted Euclidean k-NN using original embeddings:

| k | Validation Accuracy |
|---|---:|
| 30 | 57.79% |
| 50 | 58.08% |
| 60 | 58.75% |
| 65 | 58.83% |
| 70 | 58.75% |
| 75 | 58.96% |
| 80 | **59.08%** |
| 85 | 59.00% |
| 90 | 58.67% |
| 95 | 58.46% |
| 100 | 58.08% |
| 150 | 57.29% |
| 200 | 57.21% |

Accuracy generally improved as the neighborhood size increased from small values, reaching its highest observed value at k=80.

Beyond this point, accuracy declined, suggesting that incorporating increasingly distant neighbors introduced less useful information for classification.

The best sentiment configuration achieved an improvement of 1.00 percentage point over the initial k=50 weighted Euclidean configuration.

## 7. Comparative Analysis

| Parameter | Image Classification | Sentiment Classification |
|---|---|---|
| Best k | 1 | 80 |
| Distance metric | Euclidean | Euclidean |
| Normalization | None | None |
| Voting | Uniform | Distance-weighted |
| Validation accuracy | 53.08% | 59.08% |

The two datasets exhibited different responses to neighborhood size.

For image classification, smaller neighborhoods performed best, while sentiment classification benefited from larger neighborhoods.

This difference indicates that the optimal neighborhood size depends on the distribution and structure of the feature representations.

## 8. Reproducibility

The experiments were executed using Python, NumPy, pandas, and scikit-learn.

The implementation and evaluation are organized into:

- `knn.py`: Custom k-NN classifier.
- `knn_experiments.py`: Experiment configurations, dataset preparation, evaluation, and result persistence.
- `test_knn.py`: Basic implementation verification.
- `results/knn_results.csv`: Recorded experimental configurations and validation accuracies.

All experiments used a stratified 80/20 training-validation split with `random_state=42`.

Each completed experiment was saved immediately to a CSV file. Previously completed configurations were skipped when the experiment runner was restarted.

## 9. Limitations

The reported accuracies are validation results rather than independent test-set performance estimates.

Hyperparameters were selected through repeated evaluation on the same validation partition. Consequently, the selected configurations may exhibit some degree of validation-set selection bias.

The image classification experiments with feature-wise standardization were conducted only on a preliminary validation subset and are not included in the full-validation comparison.

Further evaluation on an independent test set would be required to establish generalization performance.

## 10. Conclusion

The experiments identified different optimal k-NN configurations for the two classification tasks.

Image classification achieved its highest observed validation accuracy of **53.08%** using unweighted Euclidean k-NN with k=1.

Sentiment classification achieved its highest observed validation accuracy of **59.08%** using distance-weighted Euclidean k-NN with k=80.

These configurations will serve as k-NN baselines for comparison with the other classification algorithms evaluated in the project.