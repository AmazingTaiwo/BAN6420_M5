# BAN6420: Programming in R & Python
# Module 4 Assignment: Netflix Data Visualization

# Name: Babalola Taiwo
# Learner IS: 162894

This README provide what the script does, how to run it, and the output you can expect.

# Breast Cancer Dataset PCA and Logistic Regression

This project demonstrates the application of Principal Component Analysis (PCA) to reduce the dimensions of the breast cancer dataset and then uses Logistic Regression for classification. The following steps are included:

1. Load the Breast Cancer Dataset: The dataset from `sklearn.datasets` is loaded and analyzed.
2. Data Preprocessing: The data is standardized using `StandardScaler`.
3. Principal Component Analysis (PCA): PCA is used to reduce the dataset to two components for visualization.
4. Logistic Regression: A Logistic Regression model is trained using the reduced data (PCA-transformed data).
5. Model Evaluation: Various performance metrics are provided, including accuracy, confusion matrix, classification report, and ROC curve.

 Requirements

The following Python libraries are required to run the script:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

 Description of the Script

# 1. Dataset Loading and Preprocessing
   - The breast cancer dataset is loaded using `load_breast_cancer()` from `sklearn.datasets`.
   - The data is standardized using `StandardScaler` to make it ready for PCA.

# 2. Principal Component Analysis (PCA)
   - PCA is applied to reduce the dataset to two principal components.
   - A scatter plot is generated to visualize the distribution of the two components.
   - The explained variance ratio for each of the two components is printed to show how much variance is captured.

# 3. Logistic Regression Model
   - The dataset (after PCA transformation) is split into training and test sets using `train_test_split()`.
   - A logistic regression model is trained on the reduced data.
   - The model’s performance is evaluated by predicting the test set and computing various metrics:
     - Accuracy: The proportion of correctly classified instances.
     - Classification Report: Precision, recall, and F1-score for each class.
     - Confusion Matrix: A heatmap displaying the number of correct and incorrect predictions.
     - ROC Curve: A plot of the True Positive Rate vs. False Positive Rate with AUC score.

# 4. Visualizations
   - PCA Scatter Plot: Visualizes the distribution of the data in two dimensions (first and second PCA components) with a color map showing benign (0) vs malignant (1) cases.
   - Cumulative Explained Variance Plot: A graph that shows how much variance is captured by each additional principal component.
   - Confusion Matrix: A heatmap to understand the classification errors made by the logistic regression model.
   - ROC Curve: A plot that evaluates the true positive rate against the false positive rate, showing the trade-off between sensitivity and specificity.
   - Pair Plot: A seaborn pair plot of the first few features from the original dataset, visualized with color coding for malignant/benign classes.

 How to Run the Script

# 1. Clone the Repository

First, clone the repository to your local machine or directly download the script.

```bash
git clone <repository-url>
cd <repository-directory>
```

# 2. Run the Python Script

Once the dependencies are installed, run the Python script:

```bash
python <script-name>.py
```

The script will automatically:

1. Load and preprocess the data.
2. Apply PCA and visualize the results.
3. Train the logistic regression model and evaluate it using the various metrics.

# 3. Expected Output

- Basic Statistics: You will see the summary statistics of the original dataset.
- Explained Variance: The explained variance ratio and cumulative explained variance of the PCA components.
- PCA Scatter Plot: A scatter plot showing the data projected into two PCA components.
- Cumulative Explained Variance Plot: A plot that shows how the variance is captured as more components are added.
- Logistic Regression Accuracy: The accuracy of the logistic regression model.
- Classification Report: Precision, recall, and F1-scores for both malignant and benign classes.
- Confusion Matrix: A heatmap of true vs. predicted labels.
- ROC Curve: A plot showing the ROC curve and AUC score.
- Pair Plot: A visualization of pairwise relationships in the original dataset.

 Example of Output

```bash
Basic Statistics of the Original Dataset:
       mean        std  min  25%  50%  75%  max
...    ...         ...    ...   ...  ...   ...   ...
Explained variance ratio: [0.4470529 0.19207437]
Cumulative explained variance: [0.4470529 0.63912726]
```

# Confusion Matrix:
You will see a heatmap with actual vs. predicted values, e.g.:
```
              Predicted
             Malignant  Benign
Actual
Malignant        50       5
Benign           2        60
```

# ROC Curve:
The ROC curve will show how well the logistic regression model can distinguish between malignant and benign classes, with an AUC score (e.g., 0.95).
