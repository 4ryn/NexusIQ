"""
ETL Pipeline — Data ingestion, cleaning, feature engineering.

Real datasets (download once, place in data/raw/):
  online_retail.xlsx   → https://archive.ics.uci.edu/dataset/352/online+retail
  customer_churn.csv   → https://kaggle.com/datasets/blastchar/telco-customer-churn
  superstore.csv       → https://kaggle.com/datasets/vivek468/superstore-dataset-final

Run:  python -m src.pipeline.etl
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from loguru import logger
from src.utils.config import cfg


# ── Loaders ───────────────────────────────────────────────────────────

def load_online_retail() -> pd.DataFrame:
    path = cfg.raw_dir / "online_retail.xlsx"
    if path.exists():
        logger.info("Loading online_retail.xlsx …")
        df = pd.read_excel(path, engine="openpyxl")
        df = df.dropna(subset=["CustomerID", "InvoiceDate"])
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["Revenue"]     = (df["Quantity"] * df["UnitPrice"]).round(2)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Month"]       = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
        df["CustomerID"]  = df["CustomerID"].astype(int).astype(str)
        logger.success(f"Retail loaded: {len(df):,} rows")
        return df
    logger.warning("online_retail.xlsx not found — using synthetic data")
    return _synthetic_retail()


def load_churn() -> pd.DataFrame:
    path = cfg.raw_dir / "customer_churn.csv"
    if path.exists():
        logger.info("Loading customer_churn.csv …")
        df = pd.read_csv(path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        df["Churn"]        = (df["Churn"] == "Yes").astype(int)
        logger.success(f"Churn loaded: {len(df):,} rows | churn rate {df['Churn'].mean():.1%}")
        return df
    logger.warning("customer_churn.csv not found — using synthetic data")
    return _synthetic_churn()


# ── Synthetic fallbacks ───────────────────────────────────────────────

def _synthetic_retail(n_months: int = 36) -> pd.DataFrame:
    np.random.seed(42)
    dates    = pd.date_range("2021-01-01", periods=n_months, freq="MS")
    products = ["Widget-A", "Widget-B", "Widget-C", "Widget-D"]
    countries = ["United Kingdom", "Germany", "France", "Spain", "Netherlands"]
    rows = []
    for d in dates:
        for p in products:
            for c in countries:
                base     = {"Widget-A":80000,"Widget-B":55000,"Widget-C":35000,"Widget-D":20000}[p]
                seasonal = 1 + 0.25 * np.sin(2 * np.pi * (d.month - 3) / 12)
                trend    = 1 + 0.018 * ((d.year-2021)*12 + d.month - 1)
                cmult    = {"United Kingdom":1.4,"Germany":1.1,"France":0.9,"Spain":0.7,"Netherlands":0.8}[c]
                rev      = base * seasonal * trend * cmult * np.random.normal(1, 0.07)
                rows.append({"InvoiceDate": d, "Month": d, "Description": p,
                             "Country": c, "Revenue": max(0, rev),
                             "Quantity": max(1, int(rev / np.random.uniform(3, 12))),
                             "CustomerID": f"C{np.random.randint(1000,9999)}"})
    df = pd.DataFrame(rows)
    logger.info(f"[Synthetic] {len(df):,} retail rows")
    return df


def _synthetic_churn(n: int = 7043) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        "customerID":       [f"CUST-{i}" for i in range(n)],
        "SeniorCitizen":    np.random.binomial(1, 0.16, n),
        "tenure":           np.random.randint(0, 72, n),
        "PhoneService":     np.random.choice(["Yes","No"], n, p=[0.9,0.1]),
        "InternetService":  np.random.choice(["DSL","Fiber optic","No"], n, p=[0.34,0.44,0.22]),
        "OnlineSecurity":   np.random.choice(["Yes","No","No internet service"], n),
        "TechSupport":      np.random.choice(["Yes","No","No internet service"], n),
        "Contract":         np.random.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.21,0.24]),
        "PaperlessBilling": np.random.choice(["Yes","No"], n, p=[0.59,0.41]),
        "PaymentMethod":    np.random.choice(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], n),
        "MonthlyCharges":   np.round(np.random.uniform(18, 120, n), 2),
        "TotalCharges":     np.round(np.random.uniform(0, 8500, n), 2),
    })
    prob = (0.25*(df["Contract"]=="Month-to-month") +
            0.20*(df["tenure"]<6) +
            0.15*(df["InternetService"]=="Fiber optic") +
            0.10*(df["TechSupport"]=="No") +
            0.05*(df["PaperlessBilling"]=="Yes") +
            np.random.uniform(0,0.10,n))
    df["Churn"] = (prob > 0.35).astype(int)
    logger.info(f"[Synthetic] {n:,} churn rows | churn rate {df['Churn'].mean():.1%}")
    return df


# ── Feature engineering ───────────────────────────────────────────────

def engineer_sales(df: pd.DataFrame) -> pd.DataFrame:
    gc = ["Description", "Country"]
    monthly = (df.groupby(gc + ["Month"])
                 .agg(Revenue=("Revenue","sum"), Orders=("InvoiceDate","count"))
                 .reset_index()
                 .sort_values(gc + ["Month"]))

    for lag in [1, 2, 3, 6, 12]:
        monthly[f"lag_{lag}"] = monthly.groupby(gc)["Revenue"].shift(lag)
    for w in [3, 6, 12]:
        monthly[f"roll_mean_{w}"] = monthly.groupby(gc)["Revenue"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        monthly[f"roll_std_{w}"]  = monthly.groupby(gc)["Revenue"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))

    monthly["mom_growth"] = monthly.groupby(gc)["Revenue"].pct_change(1).clip(-1, 5)
    monthly["yoy_growth"] = monthly.groupby(gc)["Revenue"].pct_change(12).clip(-1, 5)

    monthly["month"] = monthly["Month"].dt.month
    monthly["month_sin"] = np.sin(2*np.pi*monthly["month"]/12)
    monthly["month_cos"] = np.cos(2*np.pi*monthly["month"]/12)
    monthly["quarter"]   = monthly["Month"].dt.quarter
    monthly["qtr_sin"]   = np.sin(2*np.pi*monthly["quarter"]/4)
    monthly["qtr_cos"]   = np.cos(2*np.pi*monthly["quarter"]/4)
    monthly["year"]      = monthly["Month"].dt.year

    monthly = monthly.dropna(subset=["lag_3"]).reset_index(drop=True)
    logger.info(f"Sales features: {monthly.shape[0]:,} rows × {monthly.shape[1]} cols")
    return monthly


def engineer_churn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Binary encode
    for c in ["PhoneService","PaperlessBilling","OnlineSecurity","TechSupport"]:
        if c in df.columns:
            df[c] = (df[c] == "Yes").astype(int)
    # Ordinal
    if "Contract" in df.columns:
        df["contract_ord"] = df["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2}).fillna(0)
    if "InternetService" in df.columns:
        df["isp_ord"] = df["InternetService"].map({"No":0,"DSL":1,"Fiber optic":2}).fillna(0)
    # Derived
    if "MonthlyCharges" in df.columns:
        df["avg_monthly"] = df["TotalCharges"] / df["tenure"].clip(1)
    # One-hot payment method
    if "PaymentMethod" in df.columns:
        df = pd.get_dummies(df, columns=["PaymentMethod"], drop_first=True, dtype=int)
    # Drop non-numeric
    drop = ["customerID","Contract","InternetService","PhoneService",
            "OnlineSecurity","TechSupport","PaperlessBilling"]
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)
    logger.info(f"Churn features: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    if "CustomerID" not in df.columns or "InvoiceDate" not in df.columns:
        return pd.DataFrame()
    snap = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snap-x.max()).days),
        Frequency=("InvoiceDate","nunique"),
        Monetary=("Revenue","sum"),
    ).reset_index()
    rfm["R"] = pd.qcut(rfm["Recency"],  5, labels=[5,4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"),  5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM"] = rfm["R"] + rfm["F"] + rfm["M"]
    def seg(r):
        if r["RFM"]>=13: return "Champions"
        elif r["R"]>=4:  return "Loyal"
        elif r["R"]>=3:  return "Potential"
        elif r["R"]==2:  return "At Risk"
        else:            return "Lost"
    rfm["Segment"] = rfm.apply(seg, axis=1)
    logger.info(f"RFM: {len(rfm):,} customers\n{rfm['Segment'].value_counts().to_string()}")
    return rfm


# ── Main ──────────────────────────────────────────────────────────────

def run_pipeline():
    logger.info("=" * 50)
    logger.info("ETL PIPELINE STARTING")
    logger.info("=" * 50)
    out = cfg.processed_dir

    retail = load_online_retail()
    sales  = engineer_sales(retail)
    sales.to_parquet(out / "sales.parquet", index=False)
    logger.success(f"Saved sales.parquet")

    rfm = compute_rfm(retail)
    if not rfm.empty:
        rfm.to_parquet(out / "rfm.parquet", index=False)
        logger.success("Saved rfm.parquet")

    churn_raw = load_churn()
    churn     = engineer_churn(churn_raw)
    churn.to_parquet(out / "churn.parquet", index=False)
    logger.success("Saved churn.parquet")

    logger.success("=" * 50)
    logger.success("ETL COMPLETE")
    logger.success("=" * 50)
    return sales, churn, rfm


if __name__ == "__main__":
    run_pipeline()
