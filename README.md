# Used Car Price Prediction

Predicting the selling price of used cars (CarDekho listings) with a regression workflow that builds up one controlled change at a time: data cleaning, EDA, a baseline model, then a feature-engineering experiment with a proper ablation and error analysis.

**Final model:** Linear Regression on a log-transformed target, with `brand` extracted from the car name.
**Held-out test result:** R² = 0.712, MAE ≈ ₹1.44 lakh, median absolute percentage error ≈ 25%.

---

## Problem and data

- **Task:** predict `selling_price` (₹) of a used car from its age, kilometres driven, fuel type, seller type, transmission, ownership history and brand.
- **Data:** CarDekho used-car listings (Kaggle). 4,340 raw rows, 3,577 after removing 763 duplicate rows. No missing values.
- **Features:** `name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`. `car_age` is derived as `2026 - year`, and `brand` is derived from the first word of `name`.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_Data_Claning.ipynb` | Inspects data quality, drops duplicates, creates `car_age`, saves `car_data_cleaned.csv` |
| 02 | `02_EDA.ipynb` | Exploratory analysis: price distribution, age vs mileage, seller type, fuel/transmission, correlations |
| 03 | `03_Baseline_model.ipynb` | Dummy baseline vs Linear Regression (log target), 5-fold CV |
| 04 | `04_Feature_Engineering_Brand.ipynb` | Adds `brand`, runs an ablation across folds, brand-wise error analysis, final one-time test evaluation |

Run them in order. Notebook 01 produces the cleaned CSV that notebooks 02 to 04 read.

## Method

- **Split:** 80/20 train/test (`random_state=42`): 2,861 train rows, 716 test rows. The test set is used exactly once, at the end of notebook 04, for the final model.
- **Preprocessing:** `ColumnTransformer` with one-hot encoding for `fuel`, `seller_type`, `transmission`, `owner`, `brand` (`handle_unknown='ignore'`) and standard scaling for `km_driven`, `car_age`. It lives inside the pipeline, so encoders and scalers are fitted on training folds only (no leakage).
- **Target:** `selling_price` is right-skewed, so the model is trained on `log1p(price)` via `TransformedTargetRegressor`. Predictions are converted back with `expm1`, so all metrics are in ₹.
- **Evaluation:** 5-fold `KFold(shuffle=True, random_state=42)` on the training set, with MAE, RMSE and R². The same folds are used for every model, so comparisons are paired and fair.

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy (median), CV | ₹2.73L | ₹5.06L | -0.063 |
| Linear Regression, no brand, CV | ₹1.71L | ₹3.51L | 0.491 |
| Linear Regression + brand, CV | ₹1.37L | ₹2.69L | 0.698 |
| **Linear Regression + brand, held-out test** | **₹1.44L** | **₹3.05L** | **0.712** |

Test median absolute percentage error (MdAPE) is 0.249, close to the CV value of 0.241.

## Key findings

**EDA**
- Price is heavily right-skewed. The log transform gives a roughly bell-shaped distribution.
- Car age is the strongest numeric driver of price (r = -0.42). Mileage is weaker (r = -0.19), and kilometres per year is essentially unrelated to price (r = -0.03).
- Median price by seller type: Individual ₹303,500 (79% of listings), Dealer ₹484,999, Trustmark Dealer ₹750,000 (only 33 cars).
- Automatic diesel cars are the most expensive group, most likely because they are concentrated in premium SUVs.

**Modelling**
- **Brand helps, consistently.** An ablation (same folds, with vs without `brand`) improved MAE in **5 of 5 folds**, by ₹34,563 ± ₹3,567 on average. The gain is much larger than the fold-to-fold variation.
- **The model is not overfitting.** Train and validation scores are close (R² gap ≈ 0.024). The remaining error is missing signal, not model complexity.
- **Test matches CV.** Test R² (0.712) is in line with CV R² (0.698), and no tuning was done after seeing the test score.
- **Typical error is about a quarter of the price.** Percentage error is similar across most brands, so large absolute errors on Audi, BMW, Volvo and Jaguar mostly reflect their higher prices.
- **Toyota is the clearest weak spot among high-volume brands** (132 cars, MdAPE ≈ 32%). Toyota spans cheap hatchbacks to expensive SUVs, so brand alone is too coarse for it.

## Limitations

- Only linear models were tested, so non-linear effects and interactions (for example age × brand) are not captured.
- `brand` is a coarse feature, and first-word extraction mislabels some brands (`Land Rover` becomes `Land`, `OpelCorsa` stays as one token). Several brands have very few cars, so their error estimates are unreliable.
- Some odometer values look implausible (for example a 20-year-old car with about 1,000 km), and the dataset is small (about 3.5k rows), so the 716-car test set gives only a rough estimate of real-world accuracy.
- `car_age` is computed against a fixed year (2026), so it needs to be recomputed if the data is reused later.

## Setup

Python 3.12. Main libraries: `pandas`, `numpy`, `scikit-learn` (1.4 or newer, for `root_mean_squared_error`), `matplotlib`, `seaborn`, `jupyter`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Place the raw CSV (`CAR DETAILS FROM CAR DEKHO.csv`) in the `Data/` folder, then run the notebooks in order (01 to 04).

## Project structure

```
.
├── data/
│   ├── CAR DETAILS FROM CAR DEKHO.csv   # raw data
│   └── car_data_cleaned.csv             # created by notebook 01
├── notebooks/
│   ├── 01_Data_Claning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Baseline_model.ipynb
│   └── 04_Feature_Engineering_Brand.ipynb
├── .gitignore
└── README.md
```
