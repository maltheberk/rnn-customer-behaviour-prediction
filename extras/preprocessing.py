import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def load_and_prepare_data(
    data_path, holiday_path, customer_ids_path, data_end_date="2024-12-31"
):
    """Load and merge datasets"""
    # Load main data
    df = pd.read_csv(data_path, parse_dates=["DATE"])
    df.columns = df.columns.str.lower()
    df = df.sort_values(by="date", ascending=True)
    df = df[df["date"] <= data_end_date]

    # Load holiday calendar
    holiday_calendar = pd.read_csv(holiday_path, parse_dates=["DATE"])
    holiday_calendar.columns = holiday_calendar.columns.str.lower()
    df = df.merge(holiday_calendar, on="date", how="left")
    df["is_holiday_eu"] = df["is_holiday_eu"].astype(int)
    df["days_since_last_is_holiday_eu"] = df["days_since_last_is_holiday_eu"].astype(
        int
    )

    # Filter customers
    customer_ids_df = pd.read_csv(customer_ids_path)
    customer_ids = customer_ids_df["DIM_CUSTOMER_ID"].to_list()
    df = df[df["customer_id"].isin(customer_ids)].copy()

    return df


def clean_data_types(df):
    """Fix data types"""
    df["has_item_been_refunded"] = df["has_item_been_refunded"].apply(
        lambda x: 1 if x == "Y" else 0
    )

    categorical_cols = [
        "delivery_country",
        "has_item_been_refunded",
        "is_holiday_eu",
        "days_since_last_is_holiday_eu",
        "is_campaign",
    ]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df


def clean_missing_and_invalid(df):
    """Remove missing values and invalid records"""
    # Remove null delivery countries
    df = df[df["delivery_country"].notna()]

    # Remove zero/negative sales (cancelled orders)
    df = df[df["net_sales"] > 0]

    return df


def identify_outliers(df, variable_name, iqr_multiplier=3.5):
    """Identify outliers using IQR method"""
    q1 = np.percentile(df[variable_name], 25)
    q3 = np.percentile(df[variable_name], 75)
    iqr = q3 - q1

    outlier_indices = df.index[
        (df[variable_name] < q1 - iqr_multiplier * iqr)
        | (df[variable_name] > q3 + iqr_multiplier * iqr)
    ]

    return outlier_indices


def remove_outliers(df):
    """Remove outliers from numerical columns"""
    numerical_cols = ["net_sales", "costs", "orders", "items_sold"]
    outliers_to_remove = set()

    for col in numerical_cols:
        outliers = identify_outliers(df, col)
        outliers_to_remove.update(outliers)

    # Remove customers with outlier transactions
    customers_to_remove = list(set(df.loc[list(outliers_to_remove), "customer_id"]))
    df = df[~df["customer_id"].isin(customers_to_remove)]

    # Remove extreme customers
    df_customers = (
        df.groupby("customer_id")
        .agg({"orders": "sum", "net_sales": "sum"})
        .reset_index()
    )

    extreme_customers = df_customers[
        (df_customers["net_sales"] > 3600) | (df_customers["orders"] > 60)
    ]["customer_id"]
    df = df[~df["customer_id"].isin(extreme_customers)]

    return df


def engineer_features(df, training_end):
    """Create new features"""
    # Scale numerical features
    scaler = MinMaxScaler()
    cols_to_scale = ["net_sales", "costs", "discount", "items_sold"]

    for col in cols_to_scale:
        df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])

    # Encode categorical features
    df["gender_mode"] = df["gender_mode"].apply(
        lambda x: "Female" if x == "female" else x
    )
    df["gender_mode"] = df["gender_mode"].apply(
        lambda x: "1" if x == "Male" else ("0" if x == "Female" else "3")
    )
    df["gender_mode"] = df["gender_mode"].astype(int).astype("category")
    df["delivery_country"] = df["delivery_country"].astype("category").cat.codes

    # Rename columns
    df = df.rename(columns={"gender_mode": "gender", "delivery_country": "country"})

    # Create RFM features
    df_calibration = df[df["date"] <= training_end]
    df_calibration["date"] = pd.to_datetime(df_calibration["date"])
    snapshot_date = pd.to_datetime(training_end)

    rfm = df_calibration.groupby("customer_id").agg(
        Recency=("date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("date", "count"),
        Monetary=("net_sales", "sum"),
    )

    rfm["r_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = rfm["Frequency"].apply(lambda x: x if x < 5 else 5)
    rfm["m_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_segment"] = (
        rfm["r_score"].map(str) + rfm["f_score"].map(str) + rfm["m_score"].map(str)
    ).astype(int)
    rfm["rfm_score"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)

    rfm.columns = rfm.columns.str.lower()

    # Merge RFM scores
    df = df.merge(
        rfm[["r_score", "f_score", "m_score", "rfm_segment", "rfm_score"]],
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    # Scale RFM features
    rfm_cols = ["r_score", "f_score", "m_score", "rfm_score"]
    for col in rfm_cols:
        df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])

    return df


def prepare_cohort_data(df, training_start, training_end):
    """Filter to cohort customers who started in first 6 months"""
    six_months_after_start = pd.to_datetime(training_start) + pd.DateOffset(months=6)

    cohort_customers = (
        df.groupby("customer_id")["date"]
        .min()
        .query("date <= @six_months_after_start")
        .index.tolist()
    )

    df_cohort = df.query("customer_id in @cohort_customers")
    df_cohort = df_cohort.sort_values(by="customer_id").reset_index(drop=True)

    return df_cohort


def get_random_sample(df, n_customers, seed=42):
    """Get random sample of customers"""
    np.random.seed(seed)
    unique_customers = df["customer_id"].unique()
    sampled_customers = np.random.choice(
        unique_customers, size=n_customers, replace=False
    )
    return df[df["customer_id"].isin(sampled_customers)]


def preprocess_data(
    data_path,
    holiday_path,
    customer_ids_path,
    training_start,
    training_end,
    output_path=None,
    sample_size=None,
    seed=42,
):
    """Main preprocessing pipeline"""

    # Load data
    df = load_and_prepare_data(data_path, holiday_path, customer_ids_path)

    # Clean data
    df = clean_data_types(df)
    df = clean_missing_and_invalid(df)
    df = remove_outliers(df)

    # Engineer features
    df = engineer_features(df, training_end)

    # Prepare cohort
    df = prepare_cohort_data(df, training_start, training_end)

    # Sample if requested
    if sample_size:
        df = get_random_sample(df, sample_size, seed)

    # Save if path provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df


# Example usage:
# df = preprocess_data(
#     data_path="data/raw/daily_orders.csv",
#     holiday_path="data/raw/holiday_calendar.csv",
#     customer_ids_path="data/raw/customer_ids.csv",
#     training_start="2021-01-01",
#     training_end="2021-12-31",
#     output_path="data/processed/preprocessed_data.csv",
#     sample_size=50000
# )
