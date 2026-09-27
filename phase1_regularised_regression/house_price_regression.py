"""
============================================================
Phase 1 — House Price Prediction with Regularised Regression
MA2221 – Mathematics for Machine Learning
Dataset : ankushpanday1 / India House Price (Kaggle)
Author  : Pearl Mendapara
============================================================
Theory connection (Deisenroth, Faisal & Ong – Ch. 7, 9):
  OLS   : minimise ||y - Xw||^2              →  MLE of w
  Ridge : minimise ||y - Xw||^2 + λ||w||^2   →  MAP with Gaussian prior
  Lasso : minimise ||y - Xw||^2 + λ||w||_1   →  MAP with Laplace prior
============================================================

Usage:
    python house_price_regression.py [path/to/india_house_price.csv]

Download the dataset from
https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction
and place it next to this script as `india_house_price.csv`.
Plots are written to ./figures/.
"""

# ──────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "india_house_price.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    """Save the current figure into ./figures and close it."""
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()
    print(f"[Plot saved] figures/{name}")


# Consistent plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

# ──────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────
if not DATA_PATH.exists():
    sys.exit(f"Dataset not found at {DATA_PATH}. Download it from Kaggle "
             "(see README) and place it next to this script.")

df_raw = pd.read_csv(DATA_PATH)
print("=" * 60)
print("SECTION 1 — RAW DATA OVERVIEW")
print("=" * 60)
print(f"Shape        : {df_raw.shape}  ({df_raw.shape[0]} rows, {df_raw.shape[1]} columns)")
print("\nColumn names :", df_raw.columns.tolist())
print("\nData types:\n", df_raw.dtypes)
print("\nFirst 5 rows:\n", df_raw.head())

# ──────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — EDA")
print("=" * 60)

# ── 2a. Missing values ──────────────────────────────────
print("\n--- Missing Values ---")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "% Missing": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])

# ── 2b. Target distribution ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_raw["Price"] / 1e6, bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("Raw Price Distribution (₹ millions)", fontsize=13)
axes[0].set_xlabel("Price (₹ millions)")
axes[0].set_ylabel("Frequency")

axes[1].hist(np.log1p(df_raw["Price"]), bins=40, color="#55A868", edgecolor="white")
axes[1].set_title("Log-Transformed Price Distribution", fontsize=13)
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
save_fig("plot1_price_distribution.png")

# ── 2c. Correlation heatmap ─────────────────────────────
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
corr = df_raw[num_cols].corr()

plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            annot_kws={"size": 9})
plt.title("Correlation Heatmap — Numeric Features", fontsize=14)
plt.tight_layout()
save_fig("plot2_correlation_heatmap.png")

# ── 2d. Outlier detection (boxplots) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["Price", "Area", "Age"]):
    data = df_raw[col].dropna()
    if col == "Price":
        data = data / 1e6
        label = "Price (₹M)"
    else:
        label = col
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax.set_title(f"Outliers – {label}", fontsize=12)
    ax.set_ylabel(label)
plt.tight_layout()
save_fig("plot3_outliers.png")

# ── 2e. Price by City (bar chart) ───────────────────────
city_avg = (df_raw.groupby("City")["Price"]
            .mean()
            .sort_values(ascending=False) / 1e6)
plt.figure(figsize=(10, 5))
city_avg.plot(kind="bar", color="#4C72B0", edgecolor="white")
plt.title("Average Price by City (₹ millions)", fontsize=13)
plt.xlabel("City")
plt.ylabel("Avg Price (₹M)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
save_fig("plot4_price_by_city.png")

# ──────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — FEATURE ENGINEERING")
print("=" * 60)

df = df_raw.copy()

# Step 1: Fill numeric missing values with median
for col in ["Bathroom", "Parking", "Age"]:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  Filled NaN in '{col}' with median = {median_val:.1f}")

# Step 2: Fill categorical missing values with mode
for col in ["Furnishing"]:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"  Filled NaN in '{col}' with mode   = {mode_val}")

# Step 3: Log-transform target (Price is right-skewed)
y_raw = df["Price"].values
y = np.log1p(y_raw)            # log(1 + Price) so predictions stay positive
print(f"\n  Target skewness (raw) : {pd.Series(y_raw).skew():.2f}")
print(f"  Target skewness (log) : {pd.Series(y).skew():.2f}")
print("  → Log transform applied to reduce right-skew.")

# Step 4: One-hot encode categorical columns
cat_cols = ["City", "Location", "Furnishing", "Status"]
df_encoded = pd.get_dummies(df.drop("Price", axis=1), columns=cat_cols, drop_first=True)
print(f"\n  Shape after one-hot encoding: {df_encoded.shape}")

# Step 5: Scale features
X_raw = df_encoded.values.astype(float)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print("  Features standardised : mean≈0, std≈1")
print(f"  Final feature matrix X : {X.shape}")
feature_names = df_encoded.columns.tolist()

# ──────────────────────────────────────────────────────────
# 4. MODEL IMPLEMENTATIONS (from scratch)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — MODEL IMPLEMENTATIONS")
print("=" * 60)


def ols_from_scratch(X, y):
    """
    Ordinary Least Squares via the Normal Equation.

    Solves w* = (X^T X)^{-1} X^T y — the MLE of the weights, minimising
    the sum of squared residuals with no regularisation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)  – scaled design matrix
    y : np.ndarray, shape (n_samples,)             – target vector

    Returns
    -------
    w : np.ndarray, shape (n_features + 1,)  – weights, bias at index 0
    """
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y


def predict_ols(X, w):
    """Predict targets from a non-augmented design matrix and weights (bias at index 0)."""
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_b @ w


def ridge_from_scratch(X, y, lam):
    """
    Ridge Regression via the closed-form solution.

    Solves w* = (X^T X + λI)^{-1} X^T y.
    MML theory: equivalent to MAP estimation with a Gaussian prior
    N(0, σ²/λ · I) on the weights (Ch. 9). Larger λ → stronger shrinkage.
    The bias term is not regularised.

    Parameters
    ----------
    X   : np.ndarray, shape (n_samples, n_features)
    y   : np.ndarray, shape (n_samples,)
    lam : float – regularisation strength (λ ≥ 0)

    Returns
    -------
    w : np.ndarray, shape (n_features + 1,)  (includes bias)
    """
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    I = np.eye(X_b.shape[1])
    I[0, 0] = 0
    return np.linalg.solve(X_b.T @ X_b + lam * I, X_b.T @ y)


def predict_ridge(X, w):
    """Predict with Ridge weights (same structure as OLS)."""
    return predict_ols(X, w)


# ──────────────────────────────────────────────────────────
# 5. 5-FOLD CROSS-VALIDATION
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — 5-FOLD CROSS-VALIDATION (RMSE on log-price)")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lambdas = np.logspace(-3, 3, 60)       # 60 values from 0.001 to 1000


def cross_validate_model(X, y, kf, model_fn, lam=None):
    """
    Run K-fold cross-validation and return the mean RMSE.

    model_fn is either f(X_train, y_train, lam) (Ridge) or
    f(X_train, y_train) (OLS, when lam is None).
    """
    fold_rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w = model_fn(X_tr, y_tr, lam) if lam is not None else model_fn(X_tr, y_tr)
        y_pred = predict_ols(X_val, w)
        fold_rmses.append(np.sqrt(np.mean((y_val - y_pred) ** 2)))
    return float(np.mean(fold_rmses))


# ── OLS (no λ) ──────────────────────────────────────────
ols_rmse = cross_validate_model(X, y, kf, ols_from_scratch)
print(f"\n  OLS   RMSE (5-fold) : {ols_rmse:.4f}")

# ── Ridge: sweep λ ──────────────────────────────────────
ridge_rmses = [cross_validate_model(X, y, kf, ridge_from_scratch, lam=lam) for lam in lambdas]
best_ridge_idx = int(np.argmin(ridge_rmses))
best_ridge_lam = lambdas[best_ridge_idx]
best_ridge_rmse = ridge_rmses[best_ridge_idx]
print(f"  Ridge RMSE (best)   : {best_ridge_rmse:.4f}   (λ = {best_ridge_lam:.4f})")

# ── Lasso: sweep α (sklearn coordinate descent) ─────────
lasso_rmses = []
for lam in lambdas:
    fold_rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = Lasso(alpha=lam, max_iter=10000).fit(X_tr, y_tr)
        fold_rmses.append(np.sqrt(np.mean((y_val - model.predict(X_val)) ** 2)))
    lasso_rmses.append(float(np.mean(fold_rmses)))

best_lasso_idx = int(np.argmin(lasso_rmses))
best_lasso_lam = lambdas[best_lasso_idx]
best_lasso_rmse = lasso_rmses[best_lasso_idx]
print(f"  Lasso RMSE (best)   : {best_lasso_rmse:.4f}   (α = {best_lasso_lam:.4f})")

# ── Summary table ────────────────────────────────────────
print("\n  ┌────────────┬─────────────┬──────────────┐")
print("  │ Model      │ Best λ / α  │ 5-fold RMSE  │")
print("  ├────────────┼─────────────┼──────────────┤")
print(f"  │ OLS        │      —      │    {ols_rmse:.4f}    │")
print(f"  │ Ridge      │ {best_ridge_lam:>10.4f}  │    {best_ridge_rmse:.4f}    │")
print(f"  │ Lasso      │ {best_lasso_lam:>10.4f}  │    {best_lasso_rmse:.4f}    │")
print("  └────────────┴─────────────┴──────────────┘")

# ──────────────────────────────────────────────────────────
# 6. PLOTS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — GENERATING PLOTS")
print("=" * 60)

# ── Plot 5: RMSE vs λ (Ridge and Lasso) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].semilogx(lambdas, ridge_rmses, color="#4C72B0", linewidth=2)
axes[0].axvline(best_ridge_lam, color="red", linestyle="--", label=f"Best λ={best_ridge_lam:.3f}")
axes[0].set_title("Ridge: RMSE vs λ (5-fold CV)", fontsize=13)
axes[0].set_xlabel("λ (log scale)")
axes[0].set_ylabel("Mean RMSE (log-price)")
axes[0].legend()

axes[1].semilogx(lambdas, lasso_rmses, color="#55A868", linewidth=2)
axes[1].axvline(best_lasso_lam, color="red", linestyle="--", label=f"Best α={best_lasso_lam:.3f}")
axes[1].set_title("Lasso: RMSE vs α (5-fold CV)", fontsize=13)
axes[1].set_xlabel("α (log scale)")
axes[1].set_ylabel("Mean RMSE (log-price)")
axes[1].legend()

plt.suptitle("Cross-Validation: RMSE vs Regularisation Strength", fontsize=14, y=1.02)
plt.tight_layout()
save_fig("plot5_rmse_vs_lambda.png")

# ── Plot 6: Regularisation path (Ridge) ─────────────────
w_paths = np.array([ridge_from_scratch(X, y, lam)[1:] for lam in lambdas])   # drop bias

# Top 8 features by absolute weight at best λ
best_w = ridge_from_scratch(X, y, best_ridge_lam)
top_idx = np.argsort(np.abs(best_w[1:]))[-8:]

plt.figure(figsize=(11, 6))
for i in top_idx:
    plt.semilogx(lambdas, w_paths[:, i], label=feature_names[i])
plt.axvline(best_ridge_lam, color="black", linestyle="--", alpha=0.5,
            label=f"Best λ={best_ridge_lam:.3f}")
plt.title("Ridge Regularisation Path — Top 8 Features", fontsize=13)
plt.xlabel("λ (log scale)")
plt.ylabel("Coefficient Weight")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
save_fig("plot6_regularisation_path.png")

# ── Plot 7: Predicted vs Actual (all three models, full data) ──
w_ols = ols_from_scratch(X, y)
w_ridge = ridge_from_scratch(X, y, best_ridge_lam)
lasso_model = Lasso(alpha=best_lasso_lam, max_iter=10000).fit(X, y)

# Back-transform from log-space for interpretability
y_actual_orig = np.expm1(y)
preds_orig = [np.expm1(predict_ols(X, w_ols)),
              np.expm1(predict_ridge(X, w_ridge)),
              np.expm1(lasso_model.predict(X))]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, y_pred, name, color in zip(axes, preds_orig,
                                   ["OLS", "Ridge", "Lasso"],
                                   ["#4C72B0", "#DD8452", "#55A868"]):
    lim_min = min(y_actual_orig.min(), y_pred.min()) / 1e6
    lim_max = max(y_actual_orig.max(), y_pred.max()) / 1e6
    ax.scatter(y_actual_orig / 1e6, y_pred / 1e6, alpha=0.4, color=color, s=20)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5)
    ax.set_title(f"{name}: Predicted vs Actual", fontsize=12)
    ax.set_xlabel("Actual Price (₹M)")
    ax.set_ylabel("Predicted Price (₹M)")

plt.suptitle("Predicted vs Actual Prices — OLS, Ridge, Lasso", fontsize=14, y=1.02)
plt.tight_layout()
save_fig("plot7_predicted_vs_actual.png")

# ── Plot 8: Feature importance (Lasso non-zero coefficients) ──
lasso_coef = pd.Series(lasso_model.coef_, index=feature_names)
nonzero = lasso_coef[lasso_coef != 0].sort_values(key=abs, ascending=False).head(15)

plt.figure(figsize=(10, 6))
colors = ["#4C72B0" if v > 0 else "#C44E52" for v in nonzero.values]
nonzero.plot(kind="barh", color=colors, edgecolor="white")
plt.title("Lasso: Top 15 Non-Zero Feature Coefficients", fontsize=13)
plt.xlabel("Coefficient Value")
plt.gca().invert_yaxis()
plt.tight_layout()
save_fig("plot8_lasso_feature_importance.png")

# ──────────────────────────────────────────────────────────
# 7. INTERPRETATION SUMMARY
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — INTERPRETATION SUMMARY")
print("=" * 60)

models = {"OLS": ols_rmse, "Ridge": best_ridge_rmse, "Lasso": best_lasso_rmse}
best_model = min(models, key=models.get)

print(f"""
  Best performing model : {best_model}  (RMSE = {models[best_model]:.4f})

  OLS   RMSE : {ols_rmse:.4f}
  Ridge RMSE : {best_ridge_rmse:.4f}   (λ = {best_ridge_lam:.4f})
  Lasso RMSE : {best_lasso_rmse:.4f}   (α = {best_lasso_lam:.4f})

  MML theory connection:
  • Ridge = MAP estimation with a Gaussian prior on weights (Ch. 9).
    λ||w||² penalises large weights, i.e. assumes w ~ N(0, σ²/λ).
  • Lasso = MAP estimation with a Laplace prior on weights (Ch. 9).
    λ||w||₁ yields sparse solutions — automatic feature selection.
  • OLS = Maximum Likelihood Estimation with no prior; can overfit
    on high-dimensional data.

  Top features by Lasso (non-zero coefficients):""")
for feat, coef in nonzero.head(10).items():
    direction = "↑ price" if coef > 0 else "↓ price"
    print(f"    {feat:<35} coef = {coef:+.4f}  ({direction})")

print("\n  Conclusion:")
print("  With 21 features and 500 samples all three models reach similar")
print("  RMSE; Lasso adds interpretability by zeroing irrelevant weights.")
print("\n[All done!] 8 plots saved to ./figures/")
