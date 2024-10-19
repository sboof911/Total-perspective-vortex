# EEG Data Preprocessing and Classification

This project involves the preprocessing of EEG data and training multiple classifiers to predict different classes of EEG signals. The best-performing model is selected and used to predict on test data, achieving an accuracy of over 60%, as required.

## Dataset

You can download the dataset from the following link: [Physionet EEG Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
## Features

- **Preprocessing**: 
  - Noise filtering
  - Eye blink removal using Independent Component Analysis (ICA)

- **Classification**:
  - Six classifiers are trained:
    1. Logistic Regression using lbfgs solver
    2. Multilayer Perceptron (MLP)
    3. Support Vector Classifier (SVC)
    4. Random Forest Classifier
    5. Gradient Boosting Classifier
    6. Logistic Regression using liblinear solver
  - The best model is selected based on performance and used for predictions.

## Classifiers Explanation

1. **Logistic Regression**: 
   - A linear model used for binary or multiclass classification. It estimates the probability of a class using a logistic function.

2. **Multilayer Perceptron (MLP)**: 
   - A type of feedforward artificial neural network. MLP consists of multiple layers of neurons, capable of learning complex patterns in the data.

3. **Support Vector Classifier (SVC)**: 
   - A supervised learning algorithm used for classification by finding the optimal hyperplane that best separates the classes in the feature space.

4. **Random Forest Classifier**: 
   - An ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction. It helps in reducing overfitting and improving accuracy.

5. **Gradient Boosting Classifier**: 
   - An ensemble technique that builds a series of decision trees, where each tree corrects the errors of the previous one. It optimizes for better performance by focusing on misclassified instances.

6. **Logistic Regression with different parameters**: 
   - A variant of Logistic Regression where the model is fine-tuned with different hyperparameters to improve performance for the specific dataset.

## Requirements

Make sure to install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

The following are some of the key libraries used in this project:
- scikit-learn
- MNE

## Usage

1. Preprocessing: The data undergoes filtering to remove noise and eye blinks using ICA.

2. Training: Six classifiers are trained, and the one with the highest performance is saved for future predictions.

3. Prediction: The saved model is used to make predictions on the test set.

### Steps to run the program:

1. Clone this repository:
```bash
git clone https://github.com/sboof911/Total-perspective-vortex
```

2. Install the dependencies:
```bash
git clone https://github.com/sboof911/Total-perspective-vortex
```

3. Run the notebook or script that performs preprocessing and training.

## Results

The selected model achieves an accuracy of over 60% on the test dataset, which meets the required threshold.







