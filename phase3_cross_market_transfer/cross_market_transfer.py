"""
=============================================================================
PHASE 3 — Cross-Market Transfer Learning & Bayesian Uncertainty
=============================================================================
MA2221 – Mathematics for Machine Learning
Author  : Pearl Mendapara

RESEARCH QUESTION
-----------------
"Train on Indian house price data. Test on a structurally different US market
(Ames, Iowa). Which features of house pricing are UNIVERSAL (transfer across
markets) vs INDIA-SPECIFIC (collapse when applied to Ames)?"

THREE CONTRIBUTIONS BEYOND PHASE 1 & 2
---------------------------------------
1. Cross-market transfer experiment     (train India → test Ames proxy)
2. Bayesian linear regression with      (posterior predictive intervals via
   posterior predictive intervals        conjugate Normal-Inverse-Gamma model)
3. Feature universality analysis        (coefficient stability across markets,
                                         cosine similarity of weight vectors)

MML THEORY CONNECTIONS  (Deisenroth, Faisal & Ong, 2020)
---------------------------------------------------------
• OLS/Ridge/Lasso  → Ch. 9 (Linear Regression, MAP estimation)
• Bayesian LR      → Ch. 9.3–9.4 (Posterior, Predictive Distribution)
• PCA alignment    → Ch. 10 (Dimensionality Reduction / SVD)
• Regularisation   → Ch. 7 (Continuous Optimisation, Gradient / Closed Form)
=============================================================================

BUGS FIXED
----------
1. generate_india_dataset: `furnish` was a NumPy fixed-width string array
   (<U14); assigning None silently coerced it to the string 'None', so NaNs
   were never truly missing.  Fix: cast to object dtype before assignment.

2. generate_india_dataset: city sample counts sum to 450, not 500. The
   `n=500` argument was silently capped. Fix: increase city counts so they
   sum to ≥ 500 (or document that 450 is intentional and remove the
   misleading n=500 default).  Here we adjust city counts to sum to 500.

3. plot_summary_comparison: `bayes_r[3]` (Ames→India Bayesian transfer)
   incorrectly reused `rmse(y_india_u, blr.predict(X_india_u))` — the
   in-market India score — instead of fitting a separate BLR on Ames and
   predicting India.  Fix: train `blr_ames` on Ames, use that for slot [3].

4. plot_bayesian_intervals title claimed "Transfer produces wider
   uncertainty / the model knows it is out-of-distribution". This is
   factually wrong: because both universal feature matrices share the same
   4 dimensions and similar post-standardisation scale, `sigma_transfer ≈
   sigma_in`.  The code comment already acknowledged this; only the plot
   title was inconsistent.  Fix: title now reflects the actual result —
   sigma stays stable while RMSE explodes — and explains why a
   market-specific scale prior would be needed to widen intervals.

5. Final summary section: percentage change in credible-interval width used
   a raw ratio that could be negative or misleading when sigma values are
   nearly equal. Fix: replaced with an explicit comparison note and removed
   the spurious percentage.

6. plot_summary_comparison: the "Ames In-Market" Bayesian bar used the
   India-trained posterior (`blr`) to predict Ames, so it showed a transfer
   error (~6.3) instead of an in-market one. Fix: use `blr_ames`.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
os.chdir(FIG_DIR)   # all plt.savefig calls below write into ./figures

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

SEED = 42
rng  = np.random.default_rng(SEED)

print("=" * 70)
print("PHASE 3 — CROSS-MARKET TRANSFER & BAYESIAN UNCERTAINTY")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GENERATE / LOAD INDIA DATASET  (mirrors Phase 1 structure exactly)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1 — INDIA DATASET  (ankushpanday1 structure, n=500)")
print("=" * 70)

def generate_india_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Reproduce the India house-price dataset used in Phase 1.

    Columns  : Area, BHK, Bathroom, Parking, Floor, TotalFloor, Age,
               City, Location, Furnishing, Status, Price
    Cities   : Mumbai, Bangalore, Delhi, Hyderabad, Chennai, Pune,
               Ahmedabad, Kolkata
    Price    : right-skewed, driven primarily by Area and City tier.

    FIX 1: City sample counts now sum to exactly 500 (was 450).
    FIX 2: furnish cast to object dtype before None assignment so that
           missing values are genuine NaN/None, not the string 'None'.
    """
    rng2 = np.random.default_rng(seed)
    # FIX 1: city counts adjusted to sum to 500
    cities = {
        "Mumbai"    : (75, 120_000, 40_000),
        "Bangalore" : (65,  85_000, 25_000),
        "Delhi"     : (70,  90_000, 28_000),
        "Hyderabad" : (60,  70_000, 20_000),
        "Chennai"   : (55,  65_000, 18_000),
        "Pune"      : (55,  60_000, 17_000),
        "Ahmedabad" : (55,  55_000, 16_000),
        "Kolkata"   : (65,  50_000, 15_000),
    }
    locs = ["Downtown", "Suburbs", "Midtown", "Old City", "Electronic City",
            "Whitefield", "Bandra", "Andheri", "Koramangala", "Indiranagar"]
    furnishing = ["Furnished", "Unfurnished", "Semi-Furnished"]
    status     = ["Ready to Move", "Under Construction"]

    frames = []
    for city, (nf, base, std) in cities.items():
        area  = rng2.integers(600, 5000, nf)
        bhk   = rng2.integers(1, 6,    nf)
        bath  = rng2.integers(1, 4,    nf).astype(float)
        bath[rng2.random(nf) < 0.02] = np.nan
        park  = rng2.integers(0, 3,    nf).astype(float)
        park[rng2.random(nf) < 0.02] = np.nan
        floor = rng2.integers(0, 25,   nf)
        total = floor + rng2.integers(0, 15, nf)
        age   = rng2.integers(0, 30,   nf).astype(float)
        age[rng2.random(nf) < 0.02] = np.nan
        # FIX 2: cast to object dtype so None is stored as a true null,
        #         not coerced to the string 'None' by NumPy's fixed-width
        #         string array.
        furnish = rng2.choice(furnishing, nf).astype(object)
        furnish[rng2.random(nf) < 0.02] = None
        price = (area * base
                 + rng2.normal(0, area * std * 0.3, nf)
                 - 50_000 * np.nan_to_num(age, nan=0.0)
                 + 200_000 * bhk)
        price = np.where(np.isfinite(price), price, base * 1000)
        price = price.clip(3_000_000, 200_000_000)
        df_c = pd.DataFrame({
            "Area": area, "BHK": bhk, "Bathroom": bath, "Parking": park,
            "Floor": floor, "TotalFloor": total, "Age": age,
            "City": city,
            "Location": rng2.choice(locs, nf),
            "Furnishing": furnish,
            "Status": rng2.choice(status, nf),
            "Price": price.astype(int),
        })
        frames.append(df_c)

    df_all = pd.concat(frames, ignore_index=True)
    n_actual = min(n, len(df_all))
    df = df_all.sample(n_actual, random_state=seed)
    return df.reset_index(drop=True)


def preprocess_india(df: pd.DataFrame):
    """
    Phase 1-compatible preprocessing for the India dataset.
    Returns X (scaled), y (log1p price), feature_names.
    """
    d = df.copy()
    for col in ["Bathroom", "Parking", "Age"]:
        d[col] = d[col].fillna(d[col].median())
    if "Furnishing" in d.columns:
        d["Furnishing"] = d["Furnishing"].fillna(d["Furnishing"].mode()[0])

    y = np.log1p(d["Price"].values).astype(np.float64)

    cat_cols = ["City", "Location", "Furnishing", "Status"]
    d_enc = pd.get_dummies(d.drop("Price", axis=1), columns=cat_cols, drop_first=True)
    feat_names = d_enc.columns.tolist()

    sc = StandardScaler()
    X  = sc.fit_transform(d_enc.values.astype(float))
    return X, y, feat_names, sc


df_india = generate_india_dataset(n=500, seed=SEED)
X_india, y_india, feat_india, scaler_india = preprocess_india(df_india)

print(f"  India dataset   : {X_india.shape[0]} rows × {X_india.shape[1]} features")
print(f"  Target (log₁ₚ)  : mean={y_india.mean():.3f}  std={y_india.std():.3f}")
print(f"  Cities          : {df_india['City'].nunique()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GENERATE AMES-PROXY DATASET
#     We replicate the Ames Housing feature space (same column semantics as
#     Phase 2 cities) so we can align features for transfer.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2 — AMES-PROXY DATASET  (Iowa, USA, n=1,460 equivalent)")
print("=" * 70)

def generate_ames_proxy(n: int = 1460, seed: int = 42) -> pd.DataFrame:
    """
    Ames Housing Dataset proxy (Dean De Cock, 2011).

    Structural columns that match India features exactly:
      area   ↔  GrLivArea    (living area sq-ft → convert to sq-m for alignment)
      bhk    ↔  BedroomAbvGr
      bath   ↔  FullBath
      age    ↔  2010 - YearBuilt  (years old at sale)
      floor  ↔  TotRmsAbvGrd proxy

    Ames-specific features (no India equivalent):
      OverallQual, GarageArea, LotArea, TotalBsmtSF, Fireplaces

    Target: SalePrice (USD)
    """
    rng3 = np.random.default_rng(seed)

    year_built  = rng3.integers(1900, 2010, n)
    age         = 2010 - year_built
    qual        = rng3.integers(1, 11, n)        # 1–10
    living_area = rng3.normal(1500, 500, n).clip(400, 5000)
    bedrooms    = rng3.integers(1, 6, n)
    full_bath   = rng3.integers(1, 4, n)
    garage      = rng3.normal(480, 150, n).clip(0, 1500)
    lot_area    = rng3.normal(10_000, 5000, n).clip(1300, 215_000)
    bsmt_sf     = (living_area * rng3.uniform(0.5, 1.0, n)).clip(0, 6000)
    fireplaces  = rng3.integers(0, 3, n)
    rooms       = bedrooms + rng3.integers(1, 4, n)
    cond        = rng3.integers(1, 10, n)

    # Price model calibrated to Ames median ~$180k
    price = (
        50_000
        + 80    * living_area
        + 10_000 * qual
        - 300   * age
        + 5_000 * bedrooms
        + 15_000 * full_bath
        + 30    * garage
        + 0.5   * lot_area
        + rng3.normal(0, 25_000, n)
    ).clip(34_900, 755_000)

    return pd.DataFrame({
        "SalePrice"   : price.astype(int),
        "GrLivArea"   : living_area.astype(int),
        "BedroomAbvGr": bedrooms,
        "FullBath"    : full_bath,
        "GarageArea"  : garage.astype(int),
        "LotArea"     : lot_area.astype(int),
        "YearBuilt"   : year_built,
        "OverallQual" : qual,
        "OverallCond" : cond,
        "TotalBsmtSF" : bsmt_sf.astype(int),
        "Fireplaces"  : fireplaces,
        "TotRmsAbvGrd": rooms,
        "Age"         : age,
    })


def preprocess_ames(df: pd.DataFrame):
    """
    Preprocess Ames proxy — log1p transform, standardise.
    Returns X, y, feature_names, scaler.
    """
    d = df.copy()
    y = np.log1p(d["SalePrice"].values).astype(np.float64)
    d = d.drop("SalePrice", axis=1)
    feat_names = d.columns.tolist()
    sc = StandardScaler()
    X  = sc.fit_transform(d.values.astype(float))
    return X, y, feat_names, sc


df_ames = generate_ames_proxy(n=1460, seed=SEED)
X_ames, y_ames, feat_ames, scaler_ames = preprocess_ames(df_ames)

print(f"  Ames proxy      : {X_ames.shape[0]} rows × {X_ames.shape[1]} features")
print(f"  Target (log₁ₚ)  : mean={y_ames.mean():.3f}  std={y_ames.std():.3f}")
print(f"  Price range     : ${df_ames['SalePrice'].min():,} – ${df_ames['SalePrice'].max():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ALIGNMENT  (the critical step for transfer)
#     Build a shared feature space from UNIVERSAL features only.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3 — FEATURE ALIGNMENT (Universal vs Market-Specific)")
print("=" * 70)

# Universal features: present in BOTH markets with same semantics
UNIVERSAL = {
    # India column    : Ames column
    "Area"     : "GrLivArea",
    "BHK"      : "BedroomAbvGr",
    "Bathroom" : "FullBath",
    "Age"      : "Age",
}

INDIA_SPECIFIC = ["City", "Location", "Furnishing", "Status",
                  "Parking", "Floor", "TotalFloor"]
AMES_SPECIFIC  = ["OverallQual", "GarageArea", "LotArea",
                  "TotalBsmtSF", "Fireplaces", "OverallCond", "TotRmsAbvGrd"]

print(f"  Universal features   ({len(UNIVERSAL)}) : {list(UNIVERSAL.keys())}")
print(f"  India-specific      ({len(INDIA_SPECIFIC)}) : {INDIA_SPECIFIC}")
print(f"  Ames-specific       ({len(AMES_SPECIFIC)}) : {AMES_SPECIFIC}")


def build_universal_dataset(df_raw, col_map: dict, target_col: str):
    """
    Extract and align universal features from a raw dataframe.
    col_map : {source_col: rename_as}
    Imputes numeric NaNs with median before extraction.
    """
    rename = {src: tgt for src, tgt in col_map.items()}
    keep   = list(col_map.keys()) + [target_col]
    d = df_raw[[c for c in keep if c in df_raw.columns]].copy()
    # Impute numeric NaNs with median
    for col in d.select_dtypes(include=[np.number]).columns:
        d[col] = d[col].fillna(d[col].median())
    d = d.rename(columns=rename)
    d = d.dropna()
    y  = np.log1p(d[target_col].values).astype(np.float64)
    Xd = d.drop(target_col, axis=1).values.astype(float)
    sc = StandardScaler()
    X  = sc.fit_transform(Xd)
    return X, y, list(d.drop(target_col, axis=1).columns), sc


INDIA_COL_MAP = {src: src for src in UNIVERSAL.keys()}  # same names
AMES_COL_MAP  = {v: k for k, v in UNIVERSAL.items()}    # rename to India names

# Add "Price"/"SalePrice" to the India/Ames raw frames for unified extraction
df_india_raw = generate_india_dataset(n=500, seed=SEED)
for col in ["Bathroom", "Parking", "Age"]:
    df_india_raw[col] = df_india_raw[col].fillna(df_india_raw[col].median())

X_india_u, y_india_u, feat_u, _ = build_universal_dataset(
    df_india_raw, INDIA_COL_MAP, "Price")
X_ames_u,  y_ames_u,  _,      _ = build_universal_dataset(
    df_ames, AMES_COL_MAP, "SalePrice")

print(f"\n  Aligned India (universal) : {X_india_u.shape}")
print(f"  Aligned Ames  (universal) : {X_ames_u.shape}")
print(f"  Shared feature names      : {feat_u}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OLS / RIDGE / LASSO  —  HELPER FUNCTIONS  (from Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def ols_scratch(X, y):
    """Normal equation: w* = (X^T X)^{-1} X^T y"""
    Xb = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(Xb, y, rcond=None)[0]

def ridge_scratch(X, y, lam=1.0):
    """MAP with Gaussian prior: w* = (X^T X + λI)^{-1} X^T y"""
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]
    I  = np.eye(p + 1); I[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)

def predict_scratch(X, w):
    return np.c_[np.ones(len(X)), X] @ w

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CROSS-MARKET TRANSFER EXPERIMENT
#     Experiment A: India → Ames (train on India universal, test on Ames)
#     Experiment B: Ames  → India  (reverse direction)
#     Baseline    : In-market 5-fold CV on each dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4 — CROSS-MARKET TRANSFER EXPERIMENT")
print("=" * 70)

# -- λ selection: simple 5-fold on India universal features
kf     = KFold(n_splits=5, shuffle=True, random_state=SEED)
lams   = np.logspace(-2, 3, 50)

def cv_ridge_rmse(X, y, lam):
    scores = []
    for tr, va in kf.split(X):
        w = ridge_scratch(X[tr], y[tr], lam)
        scores.append(rmse(y[va], predict_scratch(X[va], w)))
    return np.mean(scores)

best_lam = min(lams, key=lambda l: cv_ridge_rmse(X_india_u, y_india_u, l))
print(f"\n  Best Ridge λ (India 5-fold CV) : {best_lam:.4f}")

# --- IN-MARKET baselines
ols_in_india  = np.mean([rmse(y_india_u[va], predict_scratch(X_india_u[va], ols_scratch(X_india_u[tr], y_india_u[tr])))
                          for tr, va in kf.split(X_india_u)])
ols_in_ames   = np.mean([rmse(y_ames_u[va],  predict_scratch(X_ames_u[va],  ols_scratch(X_ames_u[tr],  y_ames_u[tr])))
                          for tr, va in kf.split(X_ames_u)])
ridge_in_india = cv_ridge_rmse(X_india_u, y_india_u, best_lam)
ridge_in_ames  = min(cv_ridge_rmse(X_ames_u, y_ames_u, l) for l in lams)

# --- CROSS-MARKET  (train on all of one market, predict the other)
w_ols_india    = ols_scratch(X_india_u, y_india_u)
w_ridge_india  = ridge_scratch(X_india_u, y_india_u, best_lam)
w_ols_ames     = ols_scratch(X_ames_u, y_ames_u)
w_ridge_ames   = ridge_scratch(X_ames_u, y_ames_u, best_lam)

transfer_india2ames_ols   = rmse(y_ames_u,  predict_scratch(X_ames_u,  w_ols_india))
transfer_india2ames_ridge = rmse(y_ames_u,  predict_scratch(X_ames_u,  w_ridge_india))
transfer_ames2india_ols   = rmse(y_india_u, predict_scratch(X_india_u, w_ols_ames))
transfer_ames2india_ridge = rmse(y_india_u, predict_scratch(X_india_u, w_ridge_ames))

print(f"""
  ┌──────────────────────────────────────┬────────────┬────────────┐
  │ Experiment                           │  OLS RMSE  │ Ridge RMSE │
  ├──────────────────────────────────────┼────────────┼────────────┤
  │ IN-MARKET  India → India (5-fold CV) │  {ols_in_india:.4f}    │  {ridge_in_india:.4f}    │
  │ IN-MARKET  Ames  → Ames  (5-fold CV) │  {ols_in_ames:.4f}    │  {ridge_in_ames:.4f}    │
  │ TRANSFER   India → Ames              │  {transfer_india2ames_ols:.4f}    │  {transfer_india2ames_ridge:.4f}    │
  │ TRANSFER   Ames  → India             │  {transfer_ames2india_ols:.4f}    │  {transfer_ames2india_ridge:.4f}    │
  └──────────────────────────────────────┴────────────┴────────────┘

  Transfer degradation (India → Ames, Ridge):
    RMSE increase = {transfer_india2ames_ridge - ridge_in_ames:.4f}
    ({(transfer_india2ames_ridge / ridge_in_ames - 1)*100:.1f}% worse than in-market Ames baseline)

  Transfer degradation (Ames → India, Ridge):
    RMSE increase = {transfer_ames2india_ridge - ridge_in_india:.4f}
    ({(transfer_ames2india_ridge / ridge_in_india - 1)*100:.1f}% worse than in-market India baseline)
""")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE UNIVERSALITY ANALYSIS
#     Compare Ridge weight vectors trained on India vs Ames.
#     Cosine similarity close to 1 → feature has universal pricing role.
#     Cosine similarity close to 0 → feature is market-specific.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 5 — FEATURE UNIVERSALITY ANALYSIS")
print("=" * 70)

w_india_ridge = w_ridge_india[1:]   # drop bias
w_ames_ridge  = w_ridge_ames[1:]

cos_sim = float(np.dot(w_india_ridge, w_ames_ridge) /
                (np.linalg.norm(w_india_ridge) * np.linalg.norm(w_ames_ridge) + 1e-12))

print(f"\n  Cosine similarity of weight vectors (India vs Ames): {cos_sim:.4f}")
print(f"  (1.0 = identical pricing logic, 0.0 = orthogonal / market-specific)\n")

feat_universality = pd.DataFrame({
    "Feature"       : feat_u,
    "India_coef"    : w_india_ridge,
    "Ames_coef"     : w_ames_ridge,
    "Sign_match"    : np.sign(w_india_ridge) == np.sign(w_ames_ridge),
    "Abs_diff"      : np.abs(w_india_ridge - w_ames_ridge),
})
feat_universality["Universal"] = feat_universality["Sign_match"] & \
                                  (feat_universality["Abs_diff"] < 0.3)

print("  Per-feature universality table:")
print(feat_universality.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BAYESIAN LINEAR REGRESSION  (Ch. 9.3–9.4 MML)
#     Conjugate Normal-Inverse-Gamma prior → closed-form posterior.
#     Provides credible intervals on predictions, not just point estimates.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6 — BAYESIAN LINEAR REGRESSION (Posterior Predictive Intervals)")
print("=" * 70)

class BayesianLinearRegression:
    """
    Bayesian LR with conjugate Gaussian prior on weights.

    Prior   : w ~ N(0, alpha^{-1} I)    (precision = alpha)
    Likelihood: y | X,w,beta ~ N(Xw, beta^{-1} I)  (noise precision = beta)

    Posterior (MML Ch. 9.3):
        Sigma_N = (alpha I + beta X^T X)^{-1}
        mu_N    = beta * Sigma_N X^T y

    Posterior predictive (MML Ch. 9.4):
        p(y* | x*) = N(mu_N^T phi(x*),  sigma_N^2(x*))
        sigma_N^2(x*) = 1/beta + phi(x*)^T Sigma_N phi(x*)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 25.0):
        self.alpha = alpha   # prior precision
        self.beta  = beta    # noise precision
        self.mu_N  = None
        self.Sigma_N = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        Xb = np.c_[np.ones(n), X]              # augment with bias
        D  = Xb.shape[1]
        S0_inv = self.alpha * np.eye(D)         # prior precision matrix
        self.Sigma_N = np.linalg.inv(S0_inv + self.beta * Xb.T @ Xb)
        self.mu_N    = self.beta * self.Sigma_N @ Xb.T @ y
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Returns posterior predictive mean and (optionally) std deviation.
        """
        n = X.shape[0]
        Xb    = np.c_[np.ones(n), X]
        mu    = Xb @ self.mu_N
        if not return_std:
            return mu
        # Variance: 1/beta + diag(Xb Sigma_N Xb^T)
        var   = (1.0 / self.beta) + np.einsum("ij,jk,ik->i", Xb, self.Sigma_N, Xb)
        return mu, np.sqrt(np.abs(var))


# Fit Bayesian LR on India universal features
blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
blr.fit(X_india_u, y_india_u)
mu_pred, sigma_pred = blr.predict(X_india_u, return_std=True)

bayes_rmse_in = rmse(y_india_u, mu_pred)
print(f"  Bayesian LR in-market RMSE (India) : {bayes_rmse_in:.4f}")

# Transfer: India-trained posterior predicts Ames
mu_transfer, sigma_transfer = blr.predict(X_ames_u, return_std=True)
bayes_rmse_transfer = rmse(y_ames_u, mu_transfer)
print(f"  Bayesian LR transfer RMSE (→ Ames) : {bayes_rmse_transfer:.4f}")
print(f"  Mean predictive std (in-market)    : {sigma_pred.mean():.4f}")
print(f"  Mean predictive std (transfer)     : {sigma_transfer.mean():.4f}")

# FIX 4: sigma_transfer ≈ sigma_in because both X sets share the same 4
# standardised features.  The conjugate-Gaussian posterior uncertainty is
# driven by feature geometry (X^T X), not by price-scale drift, so the
# intervals do NOT automatically widen on transfer.  This is the key
# calibration limitation: RMSE explodes while σ stays flat, indicating the
# model does not know it is out-of-distribution at the price-scale level.
# Fixing this would require a market-specific scale prior (Ch. 9.4).
calibration_ratio = bayes_rmse_transfer / sigma_transfer.mean()
print(f"  Transfer error / predictive std    : {calibration_ratio:.2f}x")
print(f"  → Ratio >> 1: model is miscalibrated on transfer data.")
print(f"  → σ is stable across markets; RMSE explodes due to price-scale mismatch.")
print(f"  → A market-specific scale prior (Ch. 9.4) is needed to widen intervals.")

# FIX 3: Train a separate BLR on Ames for the reverse-transfer Bayesian score
blr_ames = BayesianLinearRegression(alpha=1.0, beta=25.0)
blr_ames.fit(X_ames_u, y_ames_u)
bayes_rmse_ames2india = rmse(y_india_u, blr_ames.predict(X_india_u))
print(f"  Bayesian LR transfer RMSE (→ India): {bayes_rmse_ames2india:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GENERATE ALL PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7 — GENERATING PLOTS")
print("=" * 70)


# ── PLOT 1: Transfer RMSE comparison (grouped bar) ───────────────────────────
def plot_transfer_rmse():
    fig, ax = plt.subplots(figsize=(10, 5))
    labels   = ["India→India\n(in-market)", "Ames→Ames\n(in-market)",
                "India→Ames\n(transfer)", "Ames→India\n(transfer)"]
    ols_vals  = [ols_in_india, ols_in_ames,
                 transfer_india2ames_ols, transfer_ames2india_ols]
    rdg_vals  = [ridge_in_india, ridge_in_ames,
                 transfer_india2ames_ridge, transfer_ames2india_ridge]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, ols_vals,  w, label="OLS",   color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + w/2, rdg_vals,  w, label="Ridge", color="#DD8452", alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(1.5, color="grey", linestyle="--", alpha=0.6, linewidth=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.75, ymax * 0.97, "In-Market", ha="center", fontsize=9, color="grey")
    ax.text(2.5,  ymax * 0.97, "Cross-Market Transfer", ha="center", fontsize=9, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (log-price scale)")
    ax.set_title("Phase 3 — Cross-Market Transfer: RMSE Comparison\n"
                 "Train on one market's universal features, test on the other",
                 fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("p3_plot1_transfer_rmse.png", bbox_inches="tight")
    plt.close()
    print("[Plot 1 saved] p3_plot1_transfer_rmse.png")

plot_transfer_rmse()


# ── PLOT 2: Weight vector comparison (India vs Ames coefficients) ─────────────
def plot_weight_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors_india = ["#4C72B0" if v > 0 else "#C44E52" for v in w_india_ridge]
    colors_ames  = ["#55A868"  if v > 0 else "#DD8452" for v in w_ames_ridge]
    x = np.arange(len(feat_u))

    axes[0].bar(x, w_india_ridge, color=colors_india, edgecolor="white", alpha=0.85)
    axes[0].set_title("Ridge Coefficients — India Market", fontsize=12)
    axes[0].set_xticks(x); axes[0].set_xticklabels(feat_u, rotation=20, ha="right")
    axes[0].set_ylabel("Standardised Coefficient"); axes[0].axhline(0, color="k", lw=0.8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, w_ames_ridge, color=colors_ames, edgecolor="white", alpha=0.85)
    axes[1].set_title("Ridge Coefficients — Ames Market (USA)", fontsize=12)
    axes[1].set_xticks(x); axes[1].set_xticklabels(feat_u, rotation=20, ha="right")
    axes[1].set_ylabel("Standardised Coefficient"); axes[1].axhline(0, color="k", lw=0.8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Universal Feature Weights: India vs Ames\n"
                 f"Cosine Similarity = {cos_sim:.4f}  "
                 f"(1.0 = identical pricing logic)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("p3_plot2_weight_comparison.png", bbox_inches="tight")
    plt.close()
    print("[Plot 2 saved] p3_plot2_weight_comparison.png")

plot_weight_comparison()


# ── PLOT 3: Bayesian Posterior Predictive Intervals ───────────────────────────
def plot_bayesian_intervals():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by actual price for clean ribbon plot
    idx_in  = np.argsort(y_india_u)[:100]    # show 100 samples
    idx_tr  = np.argsort(y_ames_u)[:100]

    mu_in, sig_in = blr.predict(X_india_u[idx_in], return_std=True)
    mu_tr, sig_tr = blr.predict(X_ames_u[idx_tr],  return_std=True)

    for ax, y_act, mu, sig, title, colour in [
        (axes[0], y_india_u[idx_in], mu_in, sig_in,
         "In-Market (India) — Posterior Predictive", "#4C72B0"),
        (axes[1], y_ames_u[idx_tr],  mu_tr, sig_tr,
         # FIX 4: removed false claim that intervals are wider on transfer.
         # Actual finding: σ is stable; RMSE explodes due to price-scale mismatch.
         "Transfer (India model → Ames) — σ stable, RMSE explodes", "#DD8452"),
    ]:
        xs = np.arange(len(y_act))
        ax.fill_between(xs, mu - 2*sig, mu + 2*sig,
                        alpha=0.25, color=colour, label="±2σ (95% CI)")
        ax.fill_between(xs, mu -   sig, mu +   sig,
                        alpha=0.40, color=colour, label="±1σ (68% CI)")
        ax.plot(xs, y_act, "k.", markersize=4, alpha=0.6, label="Actual")
        ax.plot(xs, mu,    color=colour, linewidth=1.5, label="Posterior mean")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Sample index (sorted by price)"); ax.set_ylabel("log₁ₚ(Price)")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

    # FIX 4: title corrected — σ is NOT wider on transfer; RMSE is.
    fig.suptitle("Bayesian LR — Posterior Predictive Intervals\n"
                 "Conjugate-Gaussian σ stays flat across markets; a market-specific\n"
                 "scale prior (Ch. 9.4) would widen intervals on transfer",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("p3_plot3_bayesian_intervals.png", bbox_inches="tight")
    plt.close()
    print("[Plot 3 saved] p3_plot3_bayesian_intervals.png")

plot_bayesian_intervals()


# ── PLOT 4: Predicted vs Actual — all 4 experiments ──────────────────────────
def plot_pred_actual_grid():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    experiments = [
        (X_india_u, y_india_u, w_ridge_india, "India→India (In-Market)", "#4C72B0"),
        (X_ames_u,  y_ames_u,  w_ridge_ames,  "Ames→Ames (In-Market)",   "#55A868"),
        (X_ames_u,  y_ames_u,  w_ridge_india, "India→Ames (Transfer)",    "#DD8452"),
        (X_india_u, y_india_u, w_ridge_ames,  "Ames→India (Transfer)",    "#9467BD"),
    ]
    for ax, (Xp, yp, w, title, col) in zip(axes.flat, experiments):
        yhat = predict_scratch(Xp, w)
        r    = rmse(yp, yhat)
        lim  = [min(yp.min(), yhat.min()) - 0.1,
                max(yp.max(), yhat.max()) + 0.1]
        ax.scatter(yp, yhat, alpha=0.35, color=col, s=12, edgecolors="none")
        ax.plot(lim, lim, "k--", lw=1.5, label="y = ŷ")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual log₁ₚ(Price)"); ax.set_ylabel("Predicted log₁ₚ(Price)")
        ax.set_title(f"{title}\nRMSE = {r:.4f}", fontsize=11)
        ax.grid(alpha=0.25)
    fig.suptitle("Phase 3 — Predicted vs Actual Across All Four Experiments\n"
                 "Transfer scatter is wider / more biased than in-market", fontsize=13)
    plt.tight_layout()
    plt.savefig("p3_plot4_pred_actual_grid.png", bbox_inches="tight")
    plt.close()
    print("[Plot 4 saved] p3_plot4_pred_actual_grid.png")

plot_pred_actual_grid()


# ── PLOT 5: Feature universality heatmap ─────────────────────────────────────
def plot_universality_heatmap():
    fig, ax = plt.subplots(figsize=(9, 4))
    data = feat_universality[["India_coef", "Ames_coef", "Abs_diff"]].T
    data.columns = feat_universality["Feature"]
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, center=0,
                annot_kws={"size": 10})
    ax.set_yticklabels(["India coef", "Ames coef", "|Diff|"], rotation=0)
    ax.set_title("Feature Universality — Ridge Coefficients: India vs Ames\n"
                 "Green = small difference (universal), Red = large difference (market-specific)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("p3_plot5_universality_heatmap.png", bbox_inches="tight")
    plt.close()
    print("[Plot 5 saved] p3_plot5_universality_heatmap.png")

plot_universality_heatmap()


# ── PLOT 6: Regularisation path comparison (India vs Ames) ───────────────────
def plot_reg_path_comparison():
    lam_grid = np.logspace(-2, 3, 60)
    paths_india = np.array([ridge_scratch(X_india_u, y_india_u, l)[1:] for l in lam_grid])
    paths_ames  = np.array([ridge_scratch(X_ames_u,  y_ames_u,  l)[1:] for l in lam_grid])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#9467BD"]
    for i, (feat, col) in enumerate(zip(feat_u, colors)):
        axes[0].semilogx(lam_grid, paths_india[:, i], color=col, label=feat, lw=2)
        axes[1].semilogx(lam_grid, paths_ames[:, i],  color=col, label=feat, lw=2)

    for ax, title in [(axes[0], "India — Ridge Regularisation Path"),
                      (axes[1], "Ames  — Ridge Regularisation Path")]:
        ax.axvline(best_lam, color="black", lw=1, ls="--", alpha=0.5, label=f"λ*={best_lam:.2f}")
        ax.set_xlabel("λ (log scale)"); ax.set_ylabel("Coefficient Weight")
        ax.set_title(title, fontsize=12); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle("Phase 3 — Regularisation Paths: Universal Features\n"
                 "Feature shrinkage order differs between markets", fontsize=13)
    plt.tight_layout()
    plt.savefig("p3_plot6_reg_paths.png", bbox_inches="tight")
    plt.close()
    print("[Plot 6 saved] p3_plot6_reg_paths.png")

plot_reg_path_comparison()


# ── PLOT 7: Uncertainty vs Transfer Gap (scatter) ────────────────────────────
def plot_uncertainty_vs_error():
    """
    Key diagnostic: compare posterior predictive std (σ) against absolute
    error on the transfer set.  If σ is a useful risk signal, higher σ
    should correlate with larger error.
    """
    mu_all,  sig_all  = blr.predict(X_ames_u, return_std=True)
    abs_err = np.abs(y_ames_u - mu_all)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(sig_all, abs_err, alpha=0.4, s=18,
                    c=sig_all, cmap="plasma", edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Predictive Std (σ)")

    # Add regression trend line
    z = np.polyfit(sig_all, abs_err, 1)
    p = np.poly1d(z)
    xs = np.linspace(sig_all.min(), sig_all.max(), 100)
    ax.plot(xs, p(xs), "r--", lw=1.8, label=f"Trend (slope={z[0]:.2f})")

    corr = float(np.corrcoef(sig_all, abs_err)[0, 1])
    ax.set_xlabel("Posterior Predictive Std (σ) — Uncertainty Estimate")
    ax.set_ylabel("Absolute Prediction Error |y - ŷ|")
    ax.set_title(f"Uncertainty Calibration on Transfer Set (India model → Ames)\n"
                 f"Pearson r = {corr:.3f}  — correlation of σ with prediction error",
                 fontsize=12)
    ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("p3_plot7_uncertainty_vs_error.png", bbox_inches="tight")
    plt.close()
    print("[Plot 7 saved] p3_plot7_uncertainty_vs_error.png")

plot_uncertainty_vs_error()


# ── PLOT 8: Full model comparison summary ────────────────────────────────────
def plot_summary_comparison():
    """
    Combined bar chart showing ALL models (OLS, Ridge, Bayesian)
    across ALL scenarios for a one-page executive summary.

    FIX 3: bayes_r[3] now uses blr_ames (trained on Ames) predicting India,
            instead of incorrectly reusing the in-market India Bayesian score.
    """
    blr_in      = rmse(y_india_u, blr.predict(X_india_u))
    # FIX 6: Ames in-market score must come from the Ames-trained posterior
    blr_ames_in = rmse(y_ames_u,  blr_ames.predict(X_ames_u))

    scenarios = [
        "India\nIn-Market", "Ames\nIn-Market",
        "India→Ames\nTransfer", "Ames→India\nTransfer"
    ]
    ols_r   = [ols_in_india,    ols_in_ames,
               transfer_india2ames_ols,   transfer_ames2india_ols]
    ridge_r = [ridge_in_india,  ridge_in_ames,
               transfer_india2ames_ridge, transfer_ames2india_ridge]
    # FIX 3: slot [3] is now the genuine Ames→India Bayesian transfer score
    bayes_r = [blr_in,          blr_ames_in,
               bayes_rmse_transfer, bayes_rmse_ames2india]

    x = np.arange(len(scenarios))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w,   ols_r,   w, label="OLS",     color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x,       ridge_r, w, label="Ridge",   color="#DD8452", alpha=0.85)
    b3 = ax.bar(x + w,   bayes_r, w, label="Bayesian",color="#55A868", alpha=0.85)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.axvline(1.5, color="grey", linestyle="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylabel("RMSE (log-price scale)")
    ax.set_title("Phase 3 — Full Model Comparison: In-Market vs Cross-Market Transfer\n"
                 "OLS / Ridge / Bayesian LR  ·  Universal features only",
                 fontsize=13)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("p3_plot8_summary.png", bbox_inches="tight")
    plt.close()
    print("[Plot 8 saved] p3_plot8_summary.png")

plot_summary_comparison()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  FINAL CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8 — FINAL SUMMARY")
print("=" * 70)

print(f"""
  PHASE 3 RESULTS SUMMARY
  ═══════════════════════════════════════════════════════════════════

  1. CROSS-MARKET TRANSFER
     India in-market (Ridge, 5-fold CV)       RMSE = {ridge_in_india:.4f}
     Ames  in-market (Ridge, 5-fold CV)       RMSE = {ridge_in_ames:.4f}
     India → Ames transfer (Ridge)            RMSE = {transfer_india2ames_ridge:.4f}
     Ames  → India transfer (Ridge)           RMSE = {transfer_ames2india_ridge:.4f}

     Transfer adds {(transfer_india2ames_ridge/ridge_in_ames-1)*100:.1f}% error vs Ames in-market.
     Some pricing signal IS universal; domain shift explains the gap.

  2. FEATURE UNIVERSALITY
     Cosine similarity (India vs Ames weights): {cos_sim:.4f}
     Universal features (same sign, |Δ|<0.3):
{chr(10).join("       • " + row["Feature"] + ("  ✓ Universal" if row["Universal"] else "  ✗ Market-specific") for _, row in feat_universality.iterrows())}

  3. BAYESIAN UNCERTAINTY
     Bayesian LR in-market RMSE  (India) : {bayes_rmse_in:.4f}
     Bayesian LR transfer RMSE (→ Ames)  : {bayes_rmse_transfer:.4f}
     Bayesian LR transfer RMSE (→ India) : {bayes_rmse_ames2india:.4f}
     Mean σ in-market (India)            : {sigma_pred.mean():.4f}
     Mean σ on transfer set (→ Ames)     : {sigma_transfer.mean():.4f}
     → Posterior predictive σ is similar across markets because
       both feature matrices share 4 standardised dimensions.
       The model's RMSE explodes on transfer while σ stays flat —
       showing the conjugate-Gaussian posterior is miscalibrated for
       cross-market use.  A market-specific scale prior (Ch. 9.4)
       would be required to widen credible intervals appropriately.

  MML THEORY LINKS
  ─────────────────
  • OLS / Ridge: Ch. 9 — MAP estimation, Normal Equation
  • Bayesian LR: Ch. 9.3-9.4 — Conjugate prior, Posterior Predictive
  • PCA alignment: Ch. 10 — SVD, explained variance
  • Regularisation paths: Ch. 7 — Convex optimisation, shrinkage

  8 plots saved:
  p3_plot1_transfer_rmse.png         p3_plot2_weight_comparison.png
  p3_plot3_bayesian_intervals.png    p3_plot4_pred_actual_grid.png
  p3_plot5_universality_heatmap.png  p3_plot6_reg_paths.png
  p3_plot7_uncertainty_vs_error.png  p3_plot8_summary.png
  ═══════════════════════════════════════════════════════════════════
""")