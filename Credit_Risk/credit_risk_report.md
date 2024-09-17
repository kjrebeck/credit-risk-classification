# Credit Risk Analysis

## Overview 

* The purpose of this analysis was to develop and evaluate machine learning models using borrower's financial data to predict loan risk and creditworthiness. 
* The dataset includes borrower's loan status, loan size, interest rate, income, debt-to-income ratio, number of accounts, derogatory marks and total debt. 
* The value_counts() method was used to understand the distribution of the target variable (loan status), indicating that the dataset is imbalanced. Additional technqiues like class weighting and oversampling should be utilized
* Process: Pre-Process Data: Select and Scale Features, Split into Test & Train Datasets, Fit Model, Train Model, Generate Classification Reports (AUC, Accuracy, Precision, Recall & F1 scores) & Confusion Matrix (False Negatives, False Positives), Evaluate Performance
* Methods and Algorithms: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `SVC`, `KNeighborsClassifier`, `XGBClassifier`, `LGBMClassifier`

# Performance Results

---

### Combined Training and Test Performance
| **Model**                   | **Set**   | **AUC**   | **Accuracy** | **Precision (Class 1)** | **Recall (Class 1)** | **F1-Score (Class 1)** | **False Negatives** | **False Positives** |
|-----------------------------|-----------|-----------|--------------|-------------------------|----------------------|------------------------|---------------------|---------------------|
| Logistic Regression         | Training  | 0.9944    | 99%          | 0.85                    | 0.98                 | 0.91                   | 37                  | 319                 |
|                             | Test      | 0.9965    | 99%          | 0.87                    | 0.98                 | 0.92                   | 14                  | 90                  |
| Decision Tree               | Training  | 0.9999    | 100%         | 0.97                    | 0.95                 | 0.96                   | 102                 | 53                  |
|                             | Test      | 0.9399    | 99%          | 0.87                    | 0.81                 | 0.84                   | 120                 | 75                  |
| Random Forest               | Training  | 0.9998    | 100%         | 0.93                    | 0.99                 | 0.96                   | 27                  | 129                 |
|                             | Test      | 0.9959    | 99%          | 0.88                    | 0.87                 | 0.87                   | 79                  | 78                  |
| Support Vector Classifier   | Training  | 0.9957    | 99%          | 0.85                    | 0.99                 | 0.92                   | 12                  | 328                 |
|                             | Test      | 0.9954    | 100%         | 0.87                    | **1.00**             | **0.93**               | **3**               | 90                  |
| K-Nearest Neighbors         | Training  | 0.9974    | 99%          | 0.85                    | 0.99                 | 0.92                   | 15                  | 326                 |
|                             | Test      | 0.9960    | 100%         | 0.87                    | 0.99                 | 0.93                   | 5                   | 90                  |
| Extra Trees                 | Training  | 0.9999    | 100%         | 0.97                    | 0.95                 | 0.96                   | 102                 | 53                  |
|                             | Test      | 0.9614    | 99%          | 0.87                    | 0.82                 | 0.85                   | 110                 | 78                  |
| AdaBoost                    | Training  | 0.9971    | 99%          | 0.85                    | 0.99                 | 0.92                   | 15                  | 323                 |
|                             | Test      | 0.9964    | 100%         | 0.87                    | 0.99                 | 0.93                   | 5                   | 90                  |
| Gradient Boosting           | Training  | 0.9984    | 99%          | 0.86                    | 0.99                 | 0.92                   | 10                  | 312                 |
|                             | Test      | 0.9953    | 99%          | 0.87                    | 0.99                 | 0.93                   | 6                   | 91                  |
| XGBoost                     | Training  | 0.9978    | 99%          | 0.85                    | 0.99                 | 0.92                   | 13                  | 323                 |
|                             | Test      | 0.9957    | 100%         | 0.87                    | 0.99                 | 0.93                   | 4                   | 90                  |
| LightGBM                    | Training  | 0.9982    | 99%          | 0.85                    | 0.99                 | 0.92                   | 16                  | 318                 |
|                             | Test      | 0.9959    | 99%          | 0.87                    | 0.99                 | 0.93                   | 7                   | 90                  |
|                             |           |           |              |                         |                      |                        |                     |                     |

### **Best Performing Model**

- **Support Vector Classifier (SVC):**

  - **Test Set Performance:**
    - **AUC:** 0.9954
    - **Accuracy:** 100%
    - **Precision (Class 1):** 0.87
    - **Recall (Class 1):** **1.00**
    - **F1-Score (Class 1):** **0.93**
    - **False Negatives:** **3** (lowest among all models)
    - **False Positives:** 90

  **Rationale:**

  - **Highest Recall:** Achieved a perfect recall of **1.00** on the test set for high-risk loans, correctly identifying all high-risk cases except for 3 false negatives.
  - **Balanced Performance:** Maintained consistent precision and the highest F1-score, effectively balancing false positives and false negatives.
  - **Generalization:** Minimal difference between training and test performance indicates good generalization without overfitting.

---

## Further Insights

- **Models with High Recall (≥ 0.99) for High-Risk Loans:**
  - **SVC, KNN, AdaBoost, Gradient Boosting, XGBoost, LightGBM**
  - These models are highly effective in identifying high-risk loans, which is crucial for mitigating financial risk.

- **Models with Lower Test AUC Scores:**
  - **Decision Tree and Extra Trees** have noticeably lower AUC scores, indicating they are less effective at distinguishing between healthy and high-risk loans.

- **Precision Consistency:**
  - All models (except Random Forest with 0.88) have a precision of **0.87** for high-risk loans on the test set, suggesting a consistent rate of false positives across models.

- **False Positives (FP):**
  - The number of false positives is relatively similar across models, with most models having around **90** false positives.
  - **Decision Tree and Extra Trees** have slightly fewer false positives but at the cost of higher false negatives.

---

## Conclusion

- **Logistic Regression Evaluation:**  This model predicts healthy loans (`0`) almost perfectly with very high precision and recall.
For high-risk loans (`1`), it shows excellent recall (98%), meaning it correctly identifies most high-risk loans, though there is some trade-off in precision, indicating a few healthy loans are misclassified as high-risk. The high AUC values (close to 1.00) for both the train and test sets suggest the model is highly effective at distinguishing between healthy and high-risk loans. 

- **Recommended Model:**
  - **Support Vector Classifier (SVC)** is the best-performing model due to its perfect recall for high-risk loans, highest F1-score, and strong AUC score. It effectively minimizes false negatives, which is critical for loan risk prediction. Selecting a model, like SVC,  that excels in identifying high-risk loans ensures better risk management and reduces potential financial losses for the lending institution.

---


