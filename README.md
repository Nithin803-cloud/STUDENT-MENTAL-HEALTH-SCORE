# Student Mental Health Score Prediction

A machine learning project that predicts a student's **Mental Health Score** from social-media usage, study habits, sleep, physical activity, stress level, and demographic information.

## What this project covers

- Exploratory Data Analysis (EDA)
- Duplicate removal and basic data cleaning
- Feature grouping by preprocessing type
- Log transformation for `Study_Hours`
- Standard scaling for numerical features
- Ordinal encoding for `Stress_Level`
- One-hot encoding for categorical features
- Linear Regression baseline
- Random Forest Regression
- Randomized hyperparameter search with 5-fold cross-validation
- Model evaluation using MAE and R²
- Saving the trained model with Joblib
- Serving predictions through a FastAPI endpoint

## Dataset

The project uses the Kaggle dataset **Student Social Media And Mental Health Impact**. The notebook downloads the dataset through `kagglehub`, so the CSV is intentionally excluded from this repository.

The target variable is:

```text
Mental_Health_Score
```

## Repository structure

```text
student-mental-health-ml/
├── mental_health_model.ipynb
├── main.py
├── MENTAL_HEALTH_MODEL.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd student-mental-health-ml
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the notebook

Open `mental_health_model.ipynb` in VS Code or Jupyter and run the cells from top to bottom.

The notebook downloads the dataset using `kagglehub`.

## Run the FastAPI model

Start the API with:

```bash
uvicorn main:app --reload
```

Then open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

Send a POST request to `/predict` with the required student features.

Example JSON:

```json
{
  "Age": 20,
  "Gender": "Male",
  "Country": "India",
  "Academic_Level": "Undergraduate",
  "Most_Used_Platform": "Instagram",
  "Purpose_Of_Use": "Education",
  "Avg_Daily_Usage_Hours": 4.5,
  "Daily_Unlocks": 50,
  "Study_Hours": 5.0,
  "Physical_Activity_Hours": 1.0,
  "Sleep_Hours_Per_Night": 7.0,
  "Stress_Level": "Medium"
}
```

## Machine learning workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ├── Log Transform + Scaling
   ├── Standard Scaling
   ├── Ordinal Encoding
   └── One-Hot Encoding
   ↓
Linear Regression Baseline
   ↓
Random Forest Regression
   ↓
RandomizedSearchCV
   ↓
Final Model
   ↓
Evaluation + Model Serialization
   ↓
FastAPI Prediction API
```

## Notes

- The trained model file is included because the FastAPI application needs it at runtime.
- The raw CSV is excluded through `.gitignore`.
- Model performance numbers should be added to this README after running the final notebook cells and recording the resulting MAE and R² values.
- This project is for machine-learning practice and should not be used to make clinical or medical decisions.
