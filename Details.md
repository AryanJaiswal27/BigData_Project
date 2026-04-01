# 📊 Energy Consumption Prediction Report

## 1. Objective

The goal of this project was to build a Machine Learning model to **predict household energy consumption (KWH)** using historical usage data.
This is a **regression problem**, where the model predicts continuous values.

---

## 2. Dataset Overview

### 📁 Dataset Used:

**CC_LCL-FullData.csv**

### 📌 Key Features:

- **LCLid** → Unique household ID
- **DateTime** → Timestamp of reading
- **KWH/hh (per half hour)** → Energy consumed in 30 minutes

### 📊 Data Characteristics:

- Time-series data (chronological)
- Multiple households
- High frequency (every 30 minutes)
- Values mostly range between **0.0 to ~1.0 KWH**

---

## 3. Problem Type

- **Type:** Supervised Learning
- **Task:** Regression
- **Goal:** Predict future KWH based on past usage

---

## 4. Approach & Pipeline

### Step-by-step process:

1. **Data Loading**

2. **Data Cleaning**
   - Convert KWH to numeric
   - Convert DateTime to timestamp
   - Remove NULL values

3. **Feature Engineering**

4. **Train-Test Split**

5. **Model Training**

6. **Prediction**

7. **Evaluation (RMSE)**

---

## 5. Feature Engineering (Most Important Part)

### 🔹 Initial Features:

- Hour of day
- Day of week

### 🔹 Lag Features (Time Memory):

- `lag1` → previous value
- `lag2` → 2 steps before

👉 Helps model understand **recent past behavior**

---

## 6. Improvements Applied (Final Version)

### ✅ 1. Added More Lag Features

- `lag3`, `lag4`

👉 Model now sees deeper history
👉 Captures longer patterns

---

### ✅ 2. Rolling Average (Trend Feature)

- Average of last 3 values

👉 Helps model understand:

- Trend (increasing / decreasing usage)
- Smooth behavior

---

### ✅ 3. Weekend Feature

- `is_weekend = 1 if Saturday/Sunday else 0`

👉 Captures human behavior:

- Weekends ≠ weekdays

---

### ✅ 4. Random Forest Tuning

- Increased trees (`numTrees`)
- Increased depth (`maxDepth`)

👉 Result:

- Better pattern learning
- Reduced bias

---

## 7. Models Used

### 🌲 Random Forest Regressor

- Ensemble learning method
- Uses multiple decision trees
- Good for non-linear relationships

---

## 8. Experiments & Results

### 🔹 Version 1 (Basic Model)

- Features: lag1, lag2, hour, day
- RMSE: **~0.228**

👉 Performance: Average

---

### 🔹 Version 2 (Improved Features)

- Added better preprocessing
- RMSE: **~0.204**

👉 Performance: Good

---

### 🔹 Version 3 (Advanced Features)

- Added:
  - lag3, lag4
  - rolling average
  - weekend feature
  - better RF tuning

- RMSE: **~0.127**

👉 Performance: ✅ **Very Good**

---

## 9. RMSE Interpretation

- RMSE measures average prediction error

### 📊 Your Final RMSE:

```
0.127 KWH
```

👉 Meaning:

- Predictions are off by ~0.13 units on average
- Since values range ~0–1 → error is small

---

## 10. Observations

- Energy usage follows **time patterns**
- Past values (lag) are very important
- Trends improve prediction accuracy
- Behavior differs on weekends

---

## 11. Challenges Faced

- ❌ Memory issues (OutOfMemoryError)
- ❌ Large dataset handling
- ❌ Spark configuration issues
- ❌ Encoding errors (terminal)

---

## 12. Solutions Applied

- Limited dataset size for training
- Used sampling (fraction training)
- Reduced number of features initially
- Optimized Spark memory usage
- Logged outputs to files

---

## 13. Final Output

### Generated Files:

- `predictions_output/` → Model predictions
- `model_report.txt` → RMSE result
- `full_output_log.txt` → Full execution logs

---

## 14. Conclusion

The final model:

- Successfully predicts energy consumption
- Handles time-series data effectively
- Achieved **high accuracy (RMSE ~0.127)**

👉 This is a **strong baseline model** for real-world applications.

---

## 15. Future Improvements

- 🔥 Use Gradient Boosting / XGBoost
- 📊 Add visualization (actual vs predicted)
- 🧠 Use LSTM (deep learning for time-series)
- ⚙️ Hyperparameter tuning
- 📈 Use full dataset with distributed training

---

## 16. Summary

| Version | Features Added    | RMSE      | Performance  |
| ------- | ----------------- | --------- | ------------ |
| V1      | Basic             | 0.228     | Average      |
| V2      | Improved cleaning | 0.204     | Good         |
| V3      | Advanced features | **0.127** | ✅ Very Good |

---

# ✅ Final Verdict

You successfully built a **scalable, optimized ML pipeline** for time-series prediction using PySpark.

---
