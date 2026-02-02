# House Price Prediction (Ames Housing)

End-to-end machine learning project to predict house prices using the Ames Housing dataset.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## Data Processing

- Handled missing values (domain-aware filling)
- Ordinal encoding for quality features
- One-hot encoding for categorical variables
- Feature engineering (HouseAge, GarageAge, TotalLivingArea)

## Model

- XGBoost Regressor
- 5-Fold Cross-Validation
- Metric: Mean Absolute Error (MAE)

**CV MAE:** ~15,000

## Results

- Train mean price: ~180,900
- Test prediction mean: ~179,100
- Predicted distribution aligns well with training data

## How to Run

```bash
pip install -r requirements.txt
python train.py
```
