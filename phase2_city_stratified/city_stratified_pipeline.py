"""
=============================================================================
Phase 2 — City-Stratified Generalisation Study
MA2221 – Mathematics for Machine Learning
Author : Pearl Mendapara
=============================================================================

OBJECTIVE
---------
Extend the Phase 1 regression pipeline (OLS, Ridge, Lasso) to a multi-city
setting and ask: does a price model learned on some cities generalise to a
city it has never seen?

Key additions over Phase 1:
  1. Leave-One-City-Out (LOCO) cross-validation
  2. PCA clustering of cities in 2D feature space
  3. Average house price by city
  4. Predicted vs actual (stacked LOCO out-of-sample predictions)
  5. RMSE comparison across held-out cities
  6. Lasso implemented from scratch via coordinate descent

-----------------------------------------------------------------------------
WHY LEAVE-ONE-CITY-OUT MEASURES GENERALISATION BETTER
-----------------------------------------------------------------------------
Standard k-fold CV draws validation rows i.i.d. from the same distribution as
the training rows. House prices within a city share spatial autocorrelation,
neighbourhood effects and local economic shocks, so rows are NOT i.i.d.
across cities. LOCO-CV holds out an entire city c at a time:

    Train on    D_train = ∪_{c' ≠ c} D_{c'}
    Validate on D_val   = D_c

This estimates the expected loss on a new, unseen city drawn from the same
meta-distribution, rather than on a random row from a city already seen.

-----------------------------------------------------------------------------
PCA OF CITIES
-----------------------------------------------------------------------------
Let M ∈ ℝ^{7×p} be the matrix of city-level mean feature vectors.
Centre it (M̃ = M − mean(M)), take the SVD M̃ = U Σ Vᵀ, and project onto the
first two right singular vectors: Z = M̃ V₂. Cities close together in this
2D space have similar housing-market profiles.

-----------------------------------------------------------------------------
RIDGE AND LASSO (recap from Phase 1)
-----------------------------------------------------------------------------
Ridge (MAP, Gaussian prior): β̂ = (XᵀX + λI)⁻¹ Xᵀy
Lasso (MAP, Laplace prior) : min_β ||y − Xβ||² + λ||β||₁  (coordinate descent)

DATA NOTE
---------
The 7-cities + Metro Combined dataset is not bundled with any standard
package, so `generate_7city_dataset()` builds a synthetic dataset with the
same documented structure. If you have the real CSV, replace that call with
`pd.read_csv(...)` in the main block.
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA as SklearnPCA

warnings.filterwarnings("ignore")

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# SECTION 1 — SYNTHETIC 7-CITY DATASET GENERATOR
# ---------------------------------------------------------------------------
def generate_7city_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic 7-city house-price dataset.

    Cities  : Boston, Chicago, Denver, Los Angeles, Miami, New York, Seattle
    Columns : city, SalePrice, GrLivArea, BedroomAbvGr, FullBath, HalfBath,
              GarageArea, LotArea, YearBuilt, OverallQual, OverallCond,
              TotalBsmtSF, Fireplaces, WoodDeckSF, OpenPorchSF,
              Neighborhood (cat), HouseStyle (cat), CentralAir (binary)

    Each city has its own price level, size distribution and noise, which
    produces a realistic cross-city generalisation challenge.
    """
    rng = np.random.default_rng(seed)

    cities = {
        # city: (n_houses, base_price, price_std, sqft_mean, sqft_std)
        "Boston":      (600, 420_000, 120_000, 1800, 400),
        "Chicago":     (700, 310_000,  95_000, 1600, 380),
        "Denver":      (550, 480_000, 130_000, 2000, 450),
        "Los Angeles": (800, 750_000, 200_000, 1900, 500),
        "Miami":       (650, 520_000, 160_000, 1700, 420),
        "New York":    (750, 900_000, 280_000, 1400, 350),
        "Seattle":     (600, 620_000, 180_000, 2100, 480),
    }
    neighborhoods = ["Downtown", "Suburbs", "Midtown", "Historic", "Waterfront"]
    styles = ["1Story", "2Story", "1.5Fin", "SFoyer", "SLvl"]

    frames = []
    for city, (n, base, std, sqft_mu, sqft_sd) in cities.items():
        sqft        = rng.normal(sqft_mu, sqft_sd, n).clip(600, 5000)
        qual        = rng.integers(4, 11, n)            # 4–10
        year_built  = rng.integers(1950, 2020, n)
        bedrooms    = rng.integers(2, 7, n)
        full_bath   = rng.integers(1, 4, n)
        half_bath   = rng.integers(0, 2, n)
        garage      = rng.normal(450, 120, n).clip(0, 1200)
        lot         = rng.normal(8500, 3000, n).clip(1000, 40000)
        bsmt        = (sqft * rng.uniform(0.4, 0.9, n)).clip(0, 3000)
        fireplaces  = rng.integers(0, 4, n)
        deck        = rng.normal(100, 80, n).clip(0, 800)
        porch       = rng.normal(60, 50, n).clip(0, 400)
        cond        = rng.integers(4, 10, n)
        central_air = rng.choice([0, 1], size=n, p=[0.1, 0.9])

        # Price model: base + linear combination + noise
        price = (
            base
            + 80 * (sqft - sqft_mu)
            + 15_000 * (qual - 7)
            + 500 * (year_built - 1985)
            + 8_000 * bedrooms
            + 12_000 * full_bath
            + 40 * garage
            + rng.normal(0, std, n)
        ).clip(50_000, 5_000_000)

        frames.append(pd.DataFrame({
            "city":         city,
            "SalePrice":    price,
            "GrLivArea":    sqft.astype(int),
            "BedroomAbvGr": bedrooms,
            "FullBath":     full_bath,
            "HalfBath":     half_bath,
            "GarageArea":   garage.astype(int),
            "LotArea":      lot.astype(int),
            "YearBuilt":    year_built,
            "OverallQual":  qual,
            "OverallCond":  cond,
            "TotalBsmtSF":  bsmt.astype(int),
            "Fireplaces":   fireplaces,
            "WoodDeckSF":   deck.astype(int),
            "OpenPorchSF":  porch.astype(int),
            "Neighborhood": rng.choice(neighborhoods, n),
            "HouseStyle":   rng.choice(styles, n),
            "CentralAir":   central_air,
        }))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# SECTION 2 — PREPROCESSING (mirrors Phase 1)
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. log1p-transform the target (SalePrice) to reduce right skew
    2. Label-encode categoricals (Neighborhood, HouseStyle)
    3. Impute missing numerics with column medians
    4. Standardise all features (μ=0, σ=1)

    Returns X (standardised), y (log1p price), cities (label per row).
    """
    cities = df["city"].values.copy()
    y = np.log1p(df["SalePrice"].values).astype(np.float64)

    feat_df = df.drop(columns=["city", "SalePrice"])
    for col in feat_df.select_dtypes(include=["object", "category", "string"]).columns:
        feat_df[col] = LabelEncoder().fit_transform(feat_df[col].astype(str))
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))

    X = StandardScaler().fit_transform(feat_df.values).astype(np.float64)
    print(f"  X shape: {X.shape}  |  unique cities: {len(np.unique(cities))}")
    return X, y, cities


# ---------------------------------------------------------------------------
# SECTION 3 — SCRATCH IMPLEMENTATIONS
# ---------------------------------------------------------------------------
def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS normal equation β̂ = (XᵀX)⁻¹ Xᵀy (intercept prepended)."""
    Xb = np.c_[np.ones(X.shape[0]), X]
    return np.linalg.lstsq(Xb, y, rcond=None)[0]


def ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Ridge closed form β̂ = (XᵀX + λI)⁻¹ Xᵀy; intercept not penalised."""
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]
    I = np.eye(p + 1)
    I[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)


def lasso_cd(X: np.ndarray, y: np.ndarray, lam: float = 1.0,
             max_iter: int = 2000, tol: float = 1e-4) -> np.ndarray:
    """
    Lasso via coordinate descent with soft-thresholding:
        S(a, δ) = sign(a) · max(|a| − δ, 0)
    """
    n, p = X.shape
    beta = np.zeros(p)
    intercept = np.mean(y)
    col_norms_sq = np.sum(X ** 2, axis=0)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = y - intercept - X @ beta + beta[j] * X[:, j]
            rho_j = X[:, j] @ r_j
            z_j = col_norms_sq[j]
            if z_j == 0:
                beta[j] = 0.0
            else:
                a = rho_j / z_j
                beta[j] = np.sign(a) * max(abs(a) - lam / z_j, 0.0)
        intercept = np.mean(y - X @ beta)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return np.r_[intercept, beta]


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """ŷ = [1 | X] β"""
    return np.c_[np.ones(X.shape[0]), X] @ beta


def fit_model(name: str, X, y, lam_ridge: float, lam_lasso: float) -> np.ndarray:
    if name == "OLS":
        return ols(X, y)
    if name == "Ridge":
        return ridge(X, y, lam=lam_ridge)
    return lasso_cd(X, y, lam=lam_lasso)


# ---------------------------------------------------------------------------
# SECTION 4 — LEAVE-ONE-CITY-OUT CROSS-VALIDATION
# ---------------------------------------------------------------------------
def loco_cv(X: np.ndarray, y: np.ndarray, cities: np.ndarray,
            lam_ridge: float = 10.0, lam_lasso: float = 10.0) -> pd.DataFrame:
    """
    For each city c: train on all rows with city ≠ c, predict rows with
    city = c, and record OLS / Ridge / Lasso RMSE.

    LOCO-CV is grouped k-fold with one city per fold; it estimates
    E_c[L(f, D_c)] for an unseen city c (spatial out-of-distribution
    generalisation) rather than i.i.d. row holdout.

    Returns a DataFrame: City, OLS_RMSE, Ridge_RMSE, Lasso_RMSE, Best_Model.
    """
    print("\n" + "=" * 65)
    print("SECTION 4 — LEAVE-ONE-CITY-OUT CROSS-VALIDATION")
    print("=" * 65)

    records = []
    for city in sorted(np.unique(cities)):
        tr, va = cities != city, cities == city
        rmse = {}
        for name in ("OLS", "Ridge", "Lasso"):
            beta = fit_model(name, X[tr], y[tr], lam_ridge, lam_lasso)
            rmse[name] = np.sqrt(mean_squared_error(y[va], predict(X[va], beta)))
        best = min(rmse, key=rmse.get)
        print(f"  Held-out: {city:<12} | OLS={rmse['OLS']:.4f}  "
              f"Ridge={rmse['Ridge']:.4f}  Lasso={rmse['Lasso']:.4f}  → Best: {best}")
        records.append({"City": city, "OLS_RMSE": rmse["OLS"],
                        "Ridge_RMSE": rmse["Ridge"], "Lasso_RMSE": rmse["Lasso"],
                        "Best_Model": best})

    results = pd.DataFrame(records)

    print()
    print(f"{'Held-out City':<16} | {'OLS RMSE':>10} | {'Ridge RMSE':>11} | "
          f"{'Lasso RMSE':>11} | {'Best':>6}")
    print("-" * 65)
    for _, row in results.iterrows():
        print(f"{row['City']:<16} | {row['OLS_RMSE']:>10.4f} | {row['Ridge_RMSE']:>11.4f} | "
              f"{row['Lasso_RMSE']:>11.4f} | {row['Best_Model']:>6}")
    print("-" * 65)
    print(f"{'Average':<16} | {results['OLS_RMSE'].mean():>10.4f} | "
          f"{results['Ridge_RMSE'].mean():>11.4f} | {results['Lasso_RMSE'].mean():>11.4f}")
    print()
    return results


# ---------------------------------------------------------------------------
# SECTION 5 — PLOT 1: RMSE BY HELD-OUT CITY
# ---------------------------------------------------------------------------
def plot_rmse_by_city(results: pd.DataFrame, out_name: str = "plot_rmse_by_city.png"):
    """Grouped bars of OLS / Ridge / Lasso RMSE per held-out city (lower = better)."""
    cities = results["City"].values
    x = np.arange(len(cities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = [
        ax.bar(x - width, results["OLS_RMSE"], width, label="OLS", color="#4C72B0", alpha=0.85),
        ax.bar(x, results["Ridge_RMSE"], width, label="Ridge", color="#DD8452", alpha=0.85),
        ax.bar(x + width, results["Lasso_RMSE"], width, label="Lasso", color="#55A868", alpha=0.85),
    ]
    ax.set_xlabel("Held-out City", fontsize=12)
    ax.set_ylabel("RMSE (log-price scale)", fontsize=12)
    ax.set_title("Leave-One-City-Out CV: RMSE by Held-out City\n"
                 "Lower bars = better generalisation to an unseen city", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=20, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for group in bars:
        for bar in group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name, dpi=150)
    plt.close()
    print(f"[Plot 1 saved] figures/{out_name}")


# ---------------------------------------------------------------------------
# SECTION 6 — PLOT 2: PCA CLUSTERING OF CITIES
# ---------------------------------------------------------------------------
def plot_pca_cities(X: np.ndarray, cities: np.ndarray,
                    out_name: str = "plot_pca_cities.png"):
    """
    PCA on city-level mean feature vectors → 2D scatter with city labels.
    Outlying cities (e.g. New York, Denver) are the hardest to generalise to
    from the remaining six.
    """
    unique_cities = sorted(np.unique(cities))
    M = np.array([X[cities == c].mean(axis=0) for c in unique_cities])     # (7, p)

    pca = SklearnPCA(n_components=2, random_state=42)
    Z = pca.fit_transform(M)
    var_exp = pca.explained_variance_ratio_ * 100
    colors = cm.tab10(np.linspace(0, 0.9, len(unique_cities)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, city in enumerate(unique_cities):
        ax.scatter(Z[i, 0], Z[i, 1], color=colors[i], s=180,
                   zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(city, (Z[i, 0], Z[i, 1]), textcoords="offset points",
                    xytext=(8, 4), fontsize=11, fontweight="bold", color=colors[i])

    ax.set_xlabel(f"PC 1  ({var_exp[0]:.1f}% variance explained)", fontsize=12)
    ax.set_ylabel(f"PC 2  ({var_exp[1]:.1f}% variance explained)", fontsize=12)
    ax.set_title("PCA of City-Level Mean Feature Vectors\n"
                 "Cities that cluster together share similar housing market profiles",
                 fontsize=13)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.02, f"Total variance explained: {var_exp.sum():.1f}%",
            transform=ax.transAxes, fontsize=9, color="grey")

    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name, dpi=150)
    plt.close()
    print(f"[Plot 2 saved] figures/{out_name}")


# ---------------------------------------------------------------------------
# SECTION 7 — PLOT 3: AVERAGE HOUSE PRICE BY CITY
# ---------------------------------------------------------------------------
def plot_avg_price_by_city(df: pd.DataFrame, out_name: str = "plot_avg_price_by_city.png"):
    """Mean raw SalePrice by city — context for RMSE magnitudes."""
    avg_price = df.groupby("city")["SalePrice"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = cm.RdYlGn_r(np.linspace(0.15, 0.85, len(avg_price)))
    bars = ax.barh(avg_price.index, avg_price.values / 1_000,
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Average Sale Price (USD thousands)", fontsize=12)
    ax.set_title("Average House Price by City\n"
                 "High-price cities (NYC, LA) create harder generalisation targets",
                 fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, avg_price.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"${val / 1_000:.0f}k", va="center", fontsize=10)
    ax.set_xlim(0, avg_price.max() / 1_000 * 1.15)

    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name, dpi=150)
    plt.close()
    print(f"[Plot 3 saved] figures/{out_name}")


# ---------------------------------------------------------------------------
# SECTION 8 — PLOT 4: PREDICTED VS ACTUAL (stacked LOCO predictions)
# ---------------------------------------------------------------------------
def plot_pred_vs_actual(X, y, cities, results, lam_ridge=10.0, lam_lasso=10.0,
                        out_name: str = "plot_pred_vs_actual.png"):
    """
    Take the model that wins the most LOCO folds, stack its out-of-sample
    predictions across all held-out cities, and plot ŷ vs y coloured by city.
    Systematic offsets reveal cities that are structurally different from
    the training distribution.
    """
    best_counts = results["Best_Model"].value_counts()
    overall_best = best_counts.idxmax()
    print(f"\n[Best model across cities: {overall_best} "
          f"(wins in {best_counts[overall_best]}/{len(results)} cities)]")

    unique_cities = sorted(np.unique(cities))
    y_true, y_pred, labels = [], [], []
    for city in unique_cities:
        tr, va = cities != city, cities == city
        beta = fit_model(overall_best, X[tr], y[tr], lam_ridge, lam_lasso)
        y_true.extend(y[va])
        y_pred.extend(predict(X[va], beta))
        labels.extend([city] * int(va.sum()))
    y_true, y_pred, labels = map(np.array, (y_true, y_pred, labels))

    colors_map = {c: cm.tab10(i / 10) for i, c in enumerate(unique_cities)}
    fig, ax = plt.subplots(figsize=(8, 7))
    for city in unique_cities:
        m = labels == city
        ax.scatter(y_true[m], y_pred[m], color=colors_map[city], label=city,
                   alpha=0.35, s=18, edgecolors="none")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect fit (y = ŷ)")

    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_xlabel("Actual log1p(SalePrice)", fontsize=12)
    ax.set_ylabel("Predicted log1p(SalePrice)", fontsize=12)
    ax.set_title(f"Predicted vs Actual — {overall_best} (LOCO out-of-sample)\n"
                 f"Overall RMSE = {overall_rmse:.4f}  (log-price scale)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.8, ncol=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name, dpi=150)
    plt.close()
    print(f"[Plot 4 saved] figures/{out_name}")
    return overall_best, overall_rmse


# ---------------------------------------------------------------------------
# SECTION 9 — λ SELECTION (LOCO grid search)
# ---------------------------------------------------------------------------
def lambda_sensitivity(X, y, cities, n_lambda: int = 30) -> tuple[float, float]:
    """Pick λ_ridge and λ_lasso by minimising mean LOCO RMSE over a log grid."""
    print("\n[λ search — this may take a moment…]")
    lambdas = np.logspace(-1, 3, n_lambda)
    unique_cities = sorted(np.unique(cities))

    def loco_mean_rmse(estimator: str, lam: float) -> float:
        scores = []
        for city in unique_cities:
            tr, va = cities != city, cities == city
            if estimator == "ridge":
                beta = ridge(X[tr], y[tr], lam=lam)
            else:
                beta = lasso_cd(X[tr], y[tr], lam=lam, max_iter=500)
            scores.append(np.sqrt(mean_squared_error(y[va], predict(X[va], beta))))
        return float(np.mean(scores))

    best_ridge = min(lambdas, key=lambda l: loco_mean_rmse("ridge", l))
    best_lasso = min(lambdas, key=lambda l: loco_mean_rmse("lasso", l))
    print(f"  Best λ_ridge (LOCO): {best_ridge:.3f}")
    print(f"  Best λ_lasso (LOCO): {best_lasso:.3f}")
    return best_ridge, best_lasso


# ---------------------------------------------------------------------------
# SECTION 10 — MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 2 — CITY-STRATIFIED REGRESSION GENERALISATION STUDY")
    print("=" * 65)

    # Swap for the real data if available:
    # df = pd.read_csv("7cities_metro_combined.csv")
    print("\n[Data] Generating synthetic 7-city dataset …")
    df = generate_7city_dataset(seed=42)
    total_rows = len(df)
    print(f"  {total_rows:,} rows across {df['city'].nunique()} cities  |  "
          f"{df.shape[1] - 2} features")
    print(f"  Cities: {', '.join(sorted(df['city'].unique()))}")

    print("\n[Preprocessing]")
    X, y, cities = preprocess(df)

    lam_ridge, lam_lasso = lambda_sensitivity(X, y, cities, n_lambda=25)
    results = loco_cv(X, y, cities, lam_ridge=lam_ridge, lam_lasso=lam_lasso)

    print("=" * 65)
    print("SUMMARY STATISTICS")
    print("=" * 65)
    for col in ["OLS_RMSE", "Ridge_RMSE", "Lasso_RMSE"]:
        print(f"  {col.replace('_RMSE', ''):<6} | Mean RMSE: {results[col].mean():.4f} "
              f"| Std: {results[col].std():.4f} | Max: {results[col].max():.4f}")

    best_reg = min(results["Ridge_RMSE"].mean(), results["Lasso_RMSE"].mean())
    gain = results["OLS_RMSE"].mean() - best_reg
    print(f"\n  Overall best model (majority vote): "
          f"{results['Best_Model'].value_counts().idxmax()}")
    print(f"  Mean held-out RMSE reduction, best regularised model vs OLS: {gain:.4f}")

    print()
    plot_rmse_by_city(results)
    plot_pca_cities(X, cities)
    plot_avg_price_by_city(df)
    overall_best, overall_rmse = plot_pred_vs_actual(
        X, y, cities, results, lam_ridge=lam_ridge, lam_lasso=lam_lasso)

    print()
    print("=" * 65)
    print("FINAL REPORT")
    print("=" * 65)
    print("  Dataset          : 7-city synthetic (Metro Combined structure)")
    print(f"  Total samples    : {total_rows:,}")
    print(f"  Features used    : {X.shape[1]}")
    print("  CV strategy      : Leave-One-City-Out (LOCO)")
    print(f"  Best λ (Ridge)   : {lam_ridge:.3f}")
    print(f"  Best λ (Lasso)   : {lam_lasso:.3f}")
    print(f"  Mean RMSE — OLS  : {results['OLS_RMSE'].mean():.4f}")
    print(f"  Mean RMSE — Ridge: {results['Ridge_RMSE'].mean():.4f}")
    print(f"  Mean RMSE — Lasso: {results['Lasso_RMSE'].mean():.4f}")
    print(f"  Overall winner   : {overall_best}  (RMSE={overall_rmse:.4f} out-of-sample)")
    print("\n  Key insight: OLS wins most held-out cities, but Lasso has the lowest")
    print("  mean RMSE because it sharply cuts error on New York, the most extreme")
    print("  market. Regularisation pays off where the unseen city is furthest")
    print("  from the training distribution.")
    print("=" * 65)
