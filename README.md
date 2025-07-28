
# **Salary Prediction App – Machine Learning Project Report**

## **1. Introduction**

The **Salary Prediction App** is a machine learning-based application designed to estimate the salary of individuals based on various features such as experience, education level, job role, and industry. This application is particularly useful for HR departments, job seekers, and compensation analysts.

## **2. Project Objectives**

* Predict salaries accurately using historical data.
* Create a user-friendly app interface.
* Ensure the model is robust, interpretable, and efficient.
* Provide a pipeline for model training, data preprocessing, and predictions.

---

## **3. Project Structure and Files**

The project includes the following key components:

| **File Name**          | **Description**                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `salary_data.csv`      | Raw dataset containing historical salary data with features such as education, experience, job role, etc.                          |
| `data_cleaning.ipynb`     | Python script for cleaning and preprocessing raw data (e.g., handling nulls, encoding categorical features, normalization).        |
| `new_cleaned_data.csv` | Cleaned and preprocessed data saved after running the `data_cleaning.py` script.                                                   |
| `model_selection.ipynb`   | Script used for comparing multiple models (e.g., Linear Regression, Random Forest, XGBoost) and selecting the best performing one. |
| `salary_model.pkl`     | Serialized model file using `pickle`, ready to be loaded into the app for predictions.                                             |
| `app.py` or UI script  | (Assumed) Web or CLI interface where users can input parameters and get salary predictions.                                        |

---

## **4. Data Cleaning and Preprocessing**

**Performed in:** `data_cleaning.py`

### Steps Taken:

* Removed null or irrelevant records.
* Handled outliers using IQR method.
* One-hot encoding for categorical features (e.g., Job Role, Industry).
* Label encoding for ordinal features (e.g., Education Level).
* Scaled numerical features (e.g., Experience Years) using StandardScaler.
* Saved the cleaned dataset as `new_cleaned_data.csv`.

---

## **5. Model Training and Selection**

**Performed in:** `model_selection.py`

### Models Evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting
* XGBoost Regressor

### Evaluation Metrics:

* **R² Score**
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

**Best Performing Model:** Random Forest Regressor (R² = 0.89)

The final model was saved using `pickle` as `salary_model.pkl`.

---

## **6. Deployment and Usage**

* The trained model (`salary_model.pkl`) is loaded into the app (web or CLI).
* User inputs relevant features (experience, education, job role, etc.).
* Model predicts the expected salary.
* App displays the predicted result in a user-friendly format.

---

## **7. Results and Insights**

* The model performs well with high accuracy on the test set.
* Feature importance indicates that **Experience**, **Education Level**, and **Job Role** are the most influential factors.
* The app allows real-time predictions for salary estimation, enhancing decision-making for users.

---

## **8. Future Improvements**

* Integrate more diverse datasets (e.g., from different countries).
* Use deep learning for more complex patterns.
* Build REST API using Flask or FastAPI for broader integration.
* Add interactive visualizations using Plotly or Streamlit.

## **.Output

<img width="1036" height="800" alt="image" src="https://github.com/user-attachments/assets/d2960a02-16b8-49ce-a433-e4bda6f8858a" />


---

## **9. Conclusion**

The Salary Prediction App successfully demonstrates how machine learning can be applied to estimate salaries with reasonable accuracy. It provides a reliable, scalable solution for both individuals and organizations needing salary insights.

