# Neural Network Experiments

## Implementation and Experimental Setup

The Neural Network classifiers were implemented using PyTorch as fully connected feedforward networks. Each model consists of an input layer, one or more hidden layers with ReLU activation functions, and an output layer corresponding to the number of classification categories. Training was performed using the Adam optimizer and cross-entropy loss. Dropout and weight decay were investigated as regularization techniques to reduce overfitting and improve generalization.

Both datasets were divided into 80% training and 20% validation subsets using stratified sampling with a random state of 42. The image dataset contained 500 input features and 50 classes, while the sentiment dataset contained 768 input features and three classes. Each configuration was trained for 20 epochs with a batch size of 64. Model performance was evaluated using validation accuracy, with the best-performing epoch selected for each experiment.

A total of **39 experimental runs** were conducted across both datasets: 19 for image classification and 20 for sentiment classification. These experiments were organized into six categories per dataset: baseline evaluation, network architecture, learning rate, dropout, weight decay, and feature normalization. Some configurations were repeated across categories to maintain consistent experimental comparisons. Consequently, the 39 runs do not represent 39 unique hyperparameter combinations.

All experimental results were consolidated into `results/nn_results.csv`, with the original epoch-by-epoch training histories preserved separately for further analysis.

## Experiment 1: Baseline Feedforward Neural Network — Image Classification

**Objective:** Establish baseline performance for a feedforward neural network on the visual object recognition dataset.

**Dataset:** 20,000 labeled samples, 500 numerical features, and 50 output classes. The dataset was divided into 16,000 training samples and 4,000 validation samples using an 80/20 stratified split with random state 42.

**Architecture and hyperparameters:** Two hidden layers containing 256 and 64 neurons, respectively, with ReLU activation functions. The network was trained using the Adam optimizer, a learning rate of 0.001, a batch size of 64, cross-entropy loss, and 20 epochs. Features were standardized using parameters fitted exclusively on the training subset.

**Results:** The model achieved 100.00% training accuracy and 33.55% validation accuracy at epoch 20. The lowest validation loss, 2.6797, occurred at epoch 3, after which validation loss increased steadily to 4.9813.

**Analysis:** The substantial difference between training and validation accuracy indicates overfitting. Although the model learned to classify the training samples almost perfectly, its generalization to unseen samples remained limited.

## Experiment 2: Dropout Regularization — Image Classification

**Objective:** Investigate whether dropout regularization improves generalization and reduces overfitting.

**Methodology:** The baseline architecture and training configuration were preserved while dropout was introduced after each hidden-layer activation. Three dropout probabilities were evaluated: 0.2, 0.3, and 0.5. Each configuration was trained for 20 epochs using the same training and validation subsets.

| Dropout | Best Validation Accuracy | Best Epoch |
|---|---|---|
| 0.0 | 33.55% | 20 |
| 0.2 | 32.55% | 16 |
| 0.3 | 32.30% | 14 |
| 0.5 | 29.35% | 18 |

**Analysis:** Increasing dropout reduced training accuracy, demonstrating its regularizing effect. However, none of the tested configurations improved validation accuracy compared with the unregularized baseline. The highest dropout rate, 0.5, produced the lowest validation accuracy, suggesting that stronger regularization restricted the model's learning capacity under the tested conditions.

**Conclusion:** Dropout reduced the degree of overfitting but did not improve classification accuracy. The original baseline remains the strongest configuration among those evaluated. Subsequent experiments will investigate network architecture, optimization parameters, and alternative regularization strategies.

## Experiment 3: Network Architecture — Image Classification

**Objective:** Investigate the effect of hidden-layer width and network depth on classification performance.

**Methodology:** Five feedforward neural network architectures were evaluated using the same training and validation subsets. All configurations used ReLU activation, the Adam optimizer, a learning rate of 0.001, a batch size of 64, no dropout, and 20 training epochs. The architecture was the primary experimental variable.

| Architecture | Hidden Layers | Best Validation Accuracy | Best Epoch | Training Time (s) |
|---|---|---|---|---|
| Small | (64,) | 29.80% | 6 | 3.16 |
| Medium | (128, 64) | 31.28% | 5 | 3.32 |
| Baseline | (256, 64) | 33.55% | 20 | 3.74 |
| Large | (512, 256) | 37.98% | 15 | 6.24 |
| Deep | (256, 128, 64) | 31.35% | 20 | 4.25 |

**Analysis:** Increasing network width improved validation accuracy across the tested two-hidden-layer configurations. The large architecture achieved the highest validation accuracy of 37.975%, outperforming the baseline by 4.425 percentage points. In contrast, adding a third hidden layer reduced validation accuracy to 31.35%.

The large and deep architectures both reached 100% training accuracy, indicating that substantial overfitting remained despite their differing generalization performance. The deep architecture also exhibited a particularly high validation loss of 6.8261 at its best-accuracy epoch.

**Conclusion:** The (512, 256) architecture demonstrated the strongest validation performance among the tested configurations and will be selected for subsequent learning-rate experiments. However, its performance remains below the best k-NN classifier, indicating that additional optimization is necessary.

## Experiment 4: Learning Rate — Image Classification

**Objective:** Investigate the effect of learning rate on Neural Network convergence, classification accuracy, and generalization.

**Methodology:** Three learning rates (0.01, 0.001, and 0.0001) were evaluated using the (512, 256) architecture selected from the previous experiment. All models used ReLU activation, the Adam optimizer, cross-entropy loss, a batch size of 64, no dropout, and 20 training epochs. The same standardized training and validation subsets were used throughout.

| Learning Rate | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Training Time (s) |
|---|---|---|---|---|
| 0.01 | 11.60% | 6 | 13.39% | 7.34 |
| 0.001 | 37.98% | 15 | 100.00% | 6.99 |
| 0.0001 | 34.88% | 18 | 84.02% | 6.71 |

**Analysis:** The learning rate significantly influenced optimization performance. A learning rate of 0.01 resulted in poor training and validation accuracy, suggesting that excessively large parameter updates interfered with effective learning. A learning rate of 0.001 achieved the highest validation accuracy but exhibited substantial overfitting, with training accuracy reaching 100% and validation loss increasing during later epochs.

The smallest learning rate, 0.0001, produced slower convergence and lower validation accuracy within the fixed 20-epoch budget. However, it achieved a lower minimum validation loss than the other tested learning rates, suggesting less severe overfitting during training.

**Conclusion:** A learning rate of 0.001 provided the strongest validation accuracy among the tested configurations and remains the preferred setting for subsequent experiments. Further investigation of regularization and training duration is warranted to improve generalization.

## Experiment 5: Weight Decay Regularization — Image Classification

**Objective:** Investigate whether L2-style weight decay regularization improves Neural Network generalization and reduces overfitting.

**Methodology:** Four weight decay values (0, 0.0001, 0.001, and 0.01) were evaluated using the (512, 256) architecture. All models used the Adam optimizer, a learning rate of 0.001, ReLU activation, a batch size of 64, no dropout, and 20 training epochs. The same standardized training and validation subsets were used throughout.

| Weight Decay | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Validation Loss at Best Epoch |
|---|---|---|---|---|
| 0 | 37.98% | 15 | 100.00% | 4.1949 |
| 0.0001 | 38.13% | 17 | 100.00% | 3.6367 |
| 0.001 | 39.03% | 17 | 100.00% | 2.7753 |
| 0.01 | 31.78% | 20 | 59.93% | 2.5878 |

**Analysis:** Moderate weight decay improved validation performance compared with the unregularized baseline. The strongest configuration, using weight decay of 0.001, achieved 39.025% validation accuracy, an improvement of 1.05 percentage points over the model without regularization.

Weight decay also reduced validation loss at the best-accuracy epoch, suggesting improved generalization. However, the best-performing model still reached 100% training accuracy, demonstrating that overfitting remained present.

Increasing weight decay to 0.01 reduced training accuracy to approximately 59.93% and validation accuracy to 31.78%, suggesting excessive regularization under the tested conditions.

The model using weight decay of 0.001 also exhibited fluctuations in validation accuracy across epochs, indicating that further investigation of training stability may be beneficial.

**Conclusion:** A weight decay value of 0.001 provided the strongest validation accuracy among the tested configurations and will be retained as the preferred setting for subsequent experiments. The results demonstrate that moderate regularization can improve generalization, but stronger regularization does not necessarily produce better classification performance.

## Experiment 6: Feature Normalization — Image Classification

**Objective:** Investigate the effect of feature preprocessing on Neural Network convergence and classification performance.

**Methodology:** Three preprocessing methods were evaluated: no normalization, StandardScaler, and L2 normalization. All experiments used the (512, 256) architecture, Adam optimizer, learning rate of 0.001, weight decay of 0.001, batch size of 64, no dropout, and 20 training epochs. The same stratified training and validation split was maintained.

| Normalization | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Validation Loss at Best Epoch |
|---|---|---|---|---|
| None | 25.30% | 18 | 32.64% | 2.8130 |
| StandardScaler | 39.03% | 17 | 100.00% | 2.7753 |
| L2 | 25.55% | 18 | 32.86% | 2.8140 |

**Analysis:** StandardScaler achieved the highest validation accuracy of 39.025%, substantially outperforming both unnormalized features and L2 normalization. Standardization improved validation accuracy by 13.725 percentage points compared with no normalization.

The StandardScaler configuration also reached 100% training accuracy, indicating that overfitting remained present despite its superior classification performance.

The unnormalized and L2-normalized configurations exhibited slower convergence, with training accuracy remaining below 33% at their respective best validation epochs. Their validation accuracy was still improving near the end of the training period, suggesting that longer training or alternative hyperparameters might improve performance.

**Conclusion:** StandardScaler was the most effective preprocessing method under the tested conditions and will be retained for subsequent Neural Network experiments. These results demonstrate the importance of feature scaling in optimization and classification performance.

## Experiment 7: Baseline Neural Network — Sentiment Classification

**Objective:** Establish baseline Neural Network performance on the sentiment classification dataset and investigate initial training behaviour.

**Methodology:** A feedforward Neural Network with two hidden layers containing 256 and 64 neurons was trained using the sentiment dataset. The dataset contained 12,000 labeled samples with 768 numerical features and three sentiment classes. An 80/20 stratified training-validation split was used, producing 9,600 training samples and 2,400 validation samples.

The model used ReLU activation, the Adam optimizer, a learning rate of 0.001, a batch size of 64, cross-entropy loss, no dropout, no weight decay, and StandardScaler preprocessing. Training was conducted for 20 epochs with random seed 42.

**Results:**

| Metric | Value |
|---|---|
| Best validation accuracy | 65.87% |
| Best epoch | 1 |
| Training accuracy at best epoch | 63.98% |
| Validation loss at best epoch | 0.7590 |
| Final training accuracy | 98.18% |
| Final validation accuracy | 62.63% |

**Analysis:** The Neural Network achieved its highest validation accuracy of approximately 65.87% during the first training epoch. Although training accuracy subsequently increased to approximately 98–99%, validation accuracy did not improve.

Validation loss increased from 0.7590 at epoch 1 to 2.5723 at epoch 20, indicating substantial overfitting. This suggests that the model progressively learned patterns specific to the training data without improving generalization.

The Neural Network also outperformed the previously evaluated k-NN classifier, which achieved approximately 59.08% validation accuracy, by 6.79 percentage points.

**Conclusion:** The baseline Neural Network demonstrated stronger sentiment classification performance than k-NN but exhibited pronounced overfitting. Subsequent experiments will investigate regularization and network complexity to improve generalization.

## Experiment 8: Dropout Regularization — Sentiment Classification

**Objective:** Investigate whether dropout regularization reduces overfitting and improves Neural Network generalization on the sentiment classification dataset.

**Methodology:** Four dropout probabilities (0.0, 0.2, 0.3, and 0.5) were evaluated using a feedforward Neural Network with hidden layers of 256 and 64 neurons. All models used StandardScaler preprocessing, ReLU activation, the Adam optimizer, a learning rate of 0.001, a batch size of 64, no weight decay, and 20 training epochs. The same stratified training-validation split was maintained throughout.

| Dropout | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Validation Loss at Best Epoch |
|---|---|---|---|---|
| 0.0 | 65.88% | 1 | 63.98% | 0.7590 |
| 0.2 | 66.67% | 4 | 75.19% | 0.8020 |
| 0.3 | 66.46% | 2 | 69.01% | 0.7691 |
| 0.5 | 67.04% | 5 | 70.75% | 0.7590 |

**Analysis:** All three dropout configurations achieved higher best validation accuracy than the unregularized baseline. Dropout of 0.5 produced the strongest result, reaching 67.04% validation accuracy, an improvement of approximately 1.17 percentage points.

Dropout also reduced overfitting. Without dropout, training accuracy reached approximately 98.18% by epoch 20, while validation accuracy decreased to 62.63%. With dropout of 0.5, final training accuracy was 82.54% and validation accuracy was 66.08%.

The dropout 0.5 configuration also demonstrated a substantially smaller increase in validation loss over the training period. Nevertheless, validation accuracy peaked at epoch 5 and did not consistently improve afterward, indicating that some overfitting remained.

**Conclusion:** Dropout improved generalization on the sentiment dataset, with a dropout probability of 0.5 producing the highest validation accuracy among the tested configurations. This setting will be retained for subsequent architecture experiments.

## Experiment 9: Neural Network Architecture — Sentiment Classification

**Objective:** Evaluate how Neural Network width and depth influence sentiment classification accuracy and generalization.

**Methodology:** Five feedforward Neural Network architectures were evaluated: (64,), (128, 64), (256, 64), (512, 256), and (256, 128, 64). All models used StandardScaler preprocessing, ReLU activation, Adam optimization, a learning rate of 0.001, dropout of 0.5, no weight decay, a batch size of 64, and 20 training epochs. The same stratified training-validation split and random seed were maintained.

| Architecture | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Validation Loss at Best Epoch |
|---|---|---|---|---|
| Small (64,) | 67.38% | 12 | 75.93% | 0.7901 |
| Medium (128, 64) | 66.54% | 2 | 66.07% | 0.7608 |
| Baseline (256, 64) | 67.04% | 5 | 70.75% | 0.7590 |
| Large (512, 256) | 67.58% | 10 | 78.13% | 0.8150 |
| Deep (256, 128, 64) | 67.25% | 18 | 78.91% | 0.8428 |

**Analysis:** The large architecture (512, 256) achieved the highest validation accuracy of 67.58%, improving upon the previous sentiment Neural Network result of 67.04% by approximately 0.54 percentage points.

Increasing network depth did not consistently improve classification performance. The deeper architecture achieved 67.25%, while the smaller single-hidden-layer architecture achieved a competitive 67.38%.

These results suggest that model capacity influences performance, but increasing the number of neurons or hidden layers does not guarantee better generalization. The relatively small accuracy differences also indicate that repeated experiments with additional random seeds would be valuable.

**Conclusion:** The (512, 256) architecture was selected provisionally for subsequent learning-rate experiments because it achieved the highest validation accuracy. The (64,) architecture remains a competitive, lower-complexity alternative for final model selection.

## Experiment 10: Learning Rate — Sentiment Classification

**Objective:** Investigate the effect of learning rate on Neural Network convergence, optimization stability, and sentiment classification performance.

**Methodology:** Three learning rates (0.01, 0.001, and 0.0001) were evaluated using the (512, 256) Neural Network architecture. All models used StandardScaler preprocessing, ReLU activation, the Adam optimizer, dropout of 0.5, no weight decay, a batch size of 64, and 20 training epochs. The same stratified training-validation split and random seed were maintained.

| Learning Rate | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Validation Loss at Best Epoch |
|---|---|---|---|---|
| 0.01 | 56.21% | 1 | 51.53% | 0.9310 |
| 0.001 | 67.58% | 10 | 78.13% | 0.8150 |
| 0.0001 | 67.38% | 20 | 83.90% | 0.8728 |

**Analysis:** The learning rate of 0.001 achieved the highest validation accuracy of 67.58%. Increasing the learning rate to 0.01 resulted in unstable optimization and substantial deterioration in classification performance, with validation accuracy declining toward chance level by the final epoch.

Reducing the learning rate to 0.0001 produced more gradual convergence and achieved 67.38% validation accuracy at epoch 20. This was only approximately 0.21 percentage points below the best result obtained with 0.001. The smaller learning rate may benefit from additional training, although validation loss began increasing before the final epoch.

The 0.001 configuration continued to exhibit overfitting after reaching its highest validation accuracy at epoch 10.

**Conclusion:** A learning rate of 0.001 was retained for subsequent experiments because it achieved the highest validation accuracy under the tested training conditions. The results demonstrate that learning-rate selection significantly influences optimization stability and classification performance.

## Experiment 11: Weight Decay Regularization — Sentiment Classification

**Objective:** Investigate whether weight decay regularization improves Neural Network generalization and reduces overfitting on the sentiment classification dataset.

**Methodology:** Four weight decay values (0.0, 0.0001, 0.001, and 0.01) were evaluated using a feedforward Neural Network with hidden layers of 512 and 256 neurons. All models used StandardScaler preprocessing, ReLU activation, the Adam optimizer, a learning rate of 0.001, dropout of 0.5, a batch size of 64, and 20 training epochs. The same stratified training-validation split and random seed were maintained.

| Weight Decay | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Validation Loss at Best Epoch |
|---|---|---|---|---|
| 0.0 | 67.58% | 10 | 78.13% | 0.8150 |
| 0.0001 | 66.46% | 6 | 74.01% | 0.7822 |
| 0.001 | 67.08% | 6 | 72.51% | 0.7743 |
| 0.01 | 66.50% | 10 | 68.54% | 0.7634 |

**Analysis:** The unregularized configuration achieved the highest validation accuracy of 67.58%. Introducing weight decay reduced training accuracy at the respective best validation epochs, indicating that regularization constrained model fitting.

However, none of the weight decay configurations improved classification accuracy. Weight decay of 0.001 produced the closest result, achieving 67.08% validation accuracy.

Higher weight decay values were associated with lower validation loss at the epochs of maximum validation accuracy, suggesting a possible improvement in probabilistic predictions. Nevertheless, these measurements were obtained at different epochs, and validation accuracy remained the primary evaluation criterion.

**Conclusion:** Weight decay did not improve the highest validation accuracy under the tested conditions. A weight decay value of 0.0 was therefore retained for subsequent sentiment Neural Network experiments. Dropout alone provided the strongest regularization configuration identified so far.

## Experiment 12: Feature Normalization — Sentiment Classification

**Objective:** Evaluate the effect of feature normalization on Neural Network classification performance using 768-dimensional sentiment embeddings.

**Methodology:** Three preprocessing approaches were evaluated: no normalization, StandardScaler standardization, and L2 normalization. All models used a feedforward Neural Network with hidden layers of 512 and 256 neurons, ReLU activation, Adam optimization, a learning rate of 0.001, dropout of 0.5, no weight decay, a batch size of 64, and 20 training epochs. The same stratified training-validation split and random seed were maintained. StandardScaler parameters were fitted exclusively on the training subset.

| Normalization | Best Validation Accuracy | Best Epoch | Training Accuracy at Best Epoch | Validation Loss at Best Epoch |
|---|---|---|---|---|
| None | 66.79% | 17 | 75.08% | 0.7985 |
| StandardScaler | 67.58% | 10 | 78.13% | 0.8150 |
| L2 | 67.42% | 20 | 76.63% | 0.8032 |

**Analysis:** StandardScaler achieved the highest validation accuracy of 67.58%, outperforming the unnormalized configuration by approximately 0.79 percentage points.

L2 normalization produced a comparable result of 67.42%, only approximately 0.17 percentage points below StandardScaler. This suggests that preserving embedding direction while removing magnitude information can support effective sentiment classification.

The unnormalized model also achieved reasonable performance, indicating that the original embeddings contain useful classification information without additional preprocessing.

StandardScaler reached its best accuracy at epoch 10, whereas L2 normalization reached its best accuracy at epoch 20. However, validation loss increased during later training for both configurations, indicating that overfitting remained present.

**Conclusion:** StandardScaler was retained as the preferred preprocessing method because it achieved the highest validation accuracy. L2 normalization remained a competitive alternative. The small performance difference suggests that additional random-seed experiments would be useful before making definitive claims about preprocessing superiority.


## Overall Comparison

The Neural Network experiments demonstrated that classification performance depends substantially on model architecture, regularization, learning rate, and feature preprocessing.

For image classification, the best Neural Network achieved **39.03% validation accuracy**, using hidden layers of (512, 256), a learning rate of 0.001, weight decay of 0.001, no dropout, and StandardScaler preprocessing. However, the k-NN classifier achieved a higher validation accuracy of **53.08%**, outperforming the Neural Network by approximately 14.05 percentage points.

For sentiment classification, the best Neural Network achieved **67.58% validation accuracy**, using hidden layers of (512, 256), a learning rate of 0.001, dropout of 0.5, no weight decay, and StandardScaler preprocessing. This exceeded the best k-NN result of **59.08%** by approximately 8.50 percentage points.

Overall, k-NN currently provides stronger performance for image classification, whereas the Neural Network is more effective for sentiment classification. Final model selection will also consider the Decision Tree and Random Forest experiments.