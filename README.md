# Titanic Survival Prediction

This project is a classic machine learning classification problem. The goal is to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data from the Titanic shipwreck.

---
## Dataset

The data is sourced from the Kaggle competition **"Titanic - Machine Learning from Disaster"**. It is split into two files: `train.csv` and `test.csv`.

-   **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
-   **Description**: The dataset contains passenger information, including:
    -   **`Survived`**: The target variable (0 = No, 1 = Yes)
    -   **`Pclass`**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
    -   **`Sex`**: Passenger's sex
    -   **`Age`**: Passenger's age in years
    -   **`SibSp`**: Number of siblings/spouses aboard
    -   **`Parch`**: Number of parents/children aboard
    -   **`Fare`**: Passenger fare
    -   **`Embarked`**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---
## Notebook Workflow: `Titanic_Dataset_assignment.ipynb`

The Jupyter Notebook details the entire process from data exploration to model training and evaluation.

### 1. Data Exploration & Visualization (EDA)
The first step involves a deep dive into the dataset to understand its structure and uncover initial insights.
-   Checked for data types, summary statistics, and missing values.
-   Visualized the survival rate based on key categorical features like **`Sex`**, **`Pclass`**, and **`Embarked`**.
-   Analyzed the distribution of numerical features like **`Age`** and **`Fare`**.
-   **Key Insights**: Survival was significantly higher for females and passengers in first class.

### 2. Data Cleaning & Preprocessing
To prepare the data for modeling, the following cleaning steps were performed:
-   **Missing Values**:
    -   `Age` was imputed using the median value.
    -   `Embarked` was filled with the mode (the most frequent port).
    -   The `Cabin` column was dropped due to a high number of missing entries.
-   **Irrelevant Features**: `PassengerId`, `Name`, and `Ticket` columns were dropped as they do not provide direct predictive power in their raw form.

### 3. Feature Engineering
New features were created from existing ones to potentially improve model performance:
-   `FamilySize`: A new feature created by combining `SibSp` and `Parch`.
-   `IsAlone`: A binary feature derived from `FamilySize` to indicate if a passenger was traveling alone.

### 4. Model Training & Evaluation
The preprocessed data was used to train and compare several classification models:
-   Logistic Regression
-   Random Forest Classifier

The models were evaluated based on their **accuracy score**, and a detailed **classification report** and **confusion matrix** were generated for the best-performing model.

---
## Results & Conclusion

The **Random Forest Classifier** was identified as the best-performing model for this dataset, achieving the highest accuracy in predicting passenger survival. The model's performance highlights the importance of features like sex, ticket class, and age in determining survival outcomes on the Titanic.

---