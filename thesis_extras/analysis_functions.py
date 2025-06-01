import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from config import training_start, training_end, holdout_start, holdout_end


def create_benchmark_df(aggregate_counts, df_cohort):
    """Create mean transaction and status quo benchmark"""
    training_start_dt = pd.to_datetime(training_start)
    training_end_dt = pd.to_datetime(training_end)
    holdout_start_dt = pd.to_datetime(holdout_start)
    holdout_end_dt = pd.to_datetime(holdout_end)

    agg = aggregate_counts.copy()
    agg["transactions"] = agg["customer_id"].astype(int)
    agg = agg[["date", "transactions"]]
    agg["date"] = pd.to_datetime(agg["date"])

    ind = df_cohort[["date", "customer_id"]].copy()
    ind = ind.sort_values(by="date")
    ind["date"] = pd.to_datetime(ind["date"])

    agg_train = agg[(agg.date >= training_start_dt) & (agg.date <= training_end_dt)]
    agg_hold = agg[(agg.date >= holdout_start_dt) & (agg.date <= holdout_end_dt)]

    n_cal_weeks = len(agg_train)
    n_hold_weeks = len(agg_hold)

    mean_weekly = agg_train["transactions"].mean()
    mean_benchmark = mean_weekly * n_hold_weeks
    mean_benchmark_per_customer = mean_benchmark / len(
        df_cohort["customer_id"].unique()
    )

    ind_train = ind[(ind.date >= training_start_dt) & (ind.date <= training_end_dt)]
    counts_benchmark = ind_train.groupby("customer_id").size().to_frame("n_cal")

    counts_benchmark["status_quo"] = (
        counts_benchmark["n_cal"] * (n_hold_weeks / n_cal_weeks)
    ).round(4)
    counts_benchmark["mean_tx"] = mean_benchmark_per_customer.round(4)

    return counts_benchmark.reset_index()


def create_cohort_summary(df_cohort: pd.DataFrame):
    """Summarize cohort statistics"""
    training_start_dt = pd.to_datetime(training_start)
    training_end_dt = pd.to_datetime(training_end)
    holdout_start_dt = pd.to_datetime(holdout_start)
    holdout_end_dt = pd.to_datetime(holdout_end)

    cohort_size = df_cohort[df_cohort["date"] <= training_end_dt][
        "customer_id"
    ].nunique()

    cal_weeks = (training_end_dt - training_start_dt).days // 7
    df_cal = df_cohort[
        (df_cohort["date"] >= training_start_dt)
        & (df_cohort["date"] <= training_end_dt)
    ]

    cal_counts = df_cal.groupby("customer_id").size()
    cal_mean_events = cal_counts.mean()
    cal_first_time_buyers = (cal_counts == 1).sum()
    cal_first_time_buyers_pct = cal_first_time_buyers / len(cal_counts)

    hold_weeks = (holdout_end_dt - holdout_start_dt).days // 7
    df_hold = df_cohort[
        (df_cohort["date"] >= holdout_start_dt) & (df_cohort["date"] <= holdout_end_dt)
    ]

    hold_counts = (
        df_hold.groupby("customer_id").size().reindex(cal_counts.index).fillna(0)
    )

    hold_mean_events = hold_counts.mean()
    inactive_mask = hold_counts == 0
    hold_inactive_customers = inactive_mask.sum()
    hold_inactive_customers_pct = hold_inactive_customers / len(hold_counts)

    return {
        "cohort_size": cohort_size,
        "cal_weeks": cal_weeks,
        "cal_mean_events": round(cal_mean_events, 2),
        "cal_first_time_buyers": cal_first_time_buyers,
        "cal_first_time_buyers_pct": round(cal_first_time_buyers_pct, 4),
        "hold_weeks": hold_weeks,
        "hold_mean_events": round(hold_mean_events, 2),
        "hold_inactive_customers": hold_inactive_customers,
        "hold_inactive_customers_pct": round(hold_inactive_customers_pct, 4),
    }


def get_holdout_actuals(df):
    """Aggregate actuals by customer_id and count transactions in holdout period"""
    df = df[df["date"] >= holdout_start]
    df = df.groupby("customer_id")["date"].count().reset_index(name="holdout_actuals")
    return df


def merge_pred_with_actuals(pred_df: pd.DataFrame, actual_df: pd.DataFrame):
    """Combine actuals and predictions"""
    actuals = get_holdout_actuals(actual_df)
    predictions = pred_df

    df_combined = pd.merge(predictions, actuals, on="customer_id", how="left")
    df_combined.fillna({"holdout_actuals": 0}, inplace=True)
    df_combined["absolute_error"] = abs(
        df_combined["holdout_actuals"] - df_combined["holdout_predicted"]
    )

    return df_combined


def calculate_metrics(
    df, actual_col="holdout_actuals", pred_col="holdout_predicted", threshold=0.5
):
    """Calculate performance metrics for predictions"""
    mae = mean_absolute_error(df[actual_col], df[pred_col])
    rmse = np.sqrt(mean_squared_error(df[actual_col], df[pred_col]))

    accuracy = np.mean(
        ((df[actual_col] >= threshold) & (df[pred_col] >= threshold))
        | ((df[actual_col] < threshold) & (df[pred_col] < threshold))
    )

    bias = 100 * (sum(df[pred_col]) - sum(df[actual_col])) / sum(df[actual_col])

    majority_class = int(df[actual_col].mean() >= threshold)
    majority_preds = (
        np.ones(len(df)) * majority_class if majority_class == 1 else np.zeros(len(df))
    )
    majority_acc = np.mean(
        ((df[actual_col] >= threshold) & (majority_class == 1))
        | ((df[actual_col] < threshold) & (majority_class == 0))
    )

    majority_mae = mean_absolute_error(df[actual_col], majority_preds)
    majority_rmse = np.sqrt(mean_squared_error(df[actual_col], majority_preds))
    majority_bias = (
        100 * (sum(majority_preds) - sum(df[actual_col])) / sum(df[actual_col])
    )

    return {
        "actuals": round(sum(df[actual_col])),
        "predicted": round(sum(df[pred_col])),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "accuracy": round(accuracy, 2),
        "bias (%)": round(bias, 2),
        "majority_baseline_acc": round(majority_acc, 2),
        "ZeroR_class": majority_class,
        "ZeroR_mae": round(majority_mae, 2),
        "ZeroR_rmse": round(majority_rmse, 2),
        "ZeroR_bias (%)": round(majority_bias, 2),
    }


def evaluate_models(
    dataframes, model_ids, actual_col="holdout_actuals", pred_col="holdout_predicted"
):
    """Evaluate multiple models"""
    results = []

    for df, model_id in zip(dataframes, model_ids):
        metrics = calculate_metrics(df, actual_col, pred_col)
        results.append({"model_id": model_id, **metrics})

    return pd.DataFrame(results)


def customers_more_active_in_holdout(
    df: pd.DataFrame, df_holdout_pred: pd.DataFrame = None, min_hold_tx: int = 0.0
) -> list[int]:
    """
    Find customers whose weekly purchase rate in holdout > calibration
    AND who made at least min_hold_tx transactions in holdout window
    """
    ts, te = pd.to_datetime(training_start), pd.to_datetime(training_end)
    hs, he = pd.to_datetime(holdout_start), pd.to_datetime(holdout_end)

    cal_weeks = (te - ts).days / 7.0
    hold_weeks = (he - hs).days / 7.0

    df_cal = df.query("@ts <= date <= @te")
    df_hold = df.query("@hs <= date <= @he")

    cal_counts = df_cal.groupby("customer_id").size()

    if df_holdout_pred is None:
        hold_counts = df_hold.groupby("customer_id").size()
    else:
        df_holdout_pred = df_holdout_pred.set_index("customer_id")
        hold_counts = pd.Series(
            [
                (
                    df_holdout_pred.loc[cid, "holdout_predicted"]
                    if cid in df_holdout_pred.index
                    else 0
                )
                for cid in cal_counts.index
            ],
            index=cal_counts.index,
        )

    rates = pd.concat(
        {
            "cal_rate": cal_counts / cal_weeks,
            "hold_rate": hold_counts / hold_weeks,
            "hold_tx": hold_counts,
        },
        axis=1,
    ).fillna(0)

    mask = rates["hold_rate"] > rates["cal_rate"]
    return rates.index[mask].tolist()


def opportunity_tx_share(
    df: pd.DataFrame, opportunity_ids: list[int], min_hold_tx: int = 0.0
) -> tuple[float, int, int]:
    """
    Share of holdout transactions from qualified opportunity customers
    """
    hs, he = pd.to_datetime(holdout_start), pd.to_datetime(holdout_end)
    df_hold = df.query("@hs <= date <= @he")

    hold_tx_counts = df_hold.groupby("customer_id").size()
    qualified_ids = [
        cid for cid in opportunity_ids if hold_tx_counts.get(cid, 0) >= min_hold_tx
    ]

    opp_tx = df_hold["customer_id"].isin(qualified_ids).sum()
    total_tx = len(df_hold)

    share = opp_tx / total_tx if total_tx else 0.0
    return share, opp_tx, total_tx


def get_actual_nitt_df(df_cohort):
    """Calculate next inter-transaction time (NITT) from actual data"""
    df_nitt = df_cohort[["customer_id", "date"]].copy()

    df_nitt_cal = df_nitt.loc[df_nitt["date"] <= training_end]
    df_nitt_hold = df_nitt.loc[df_nitt["date"] >= holdout_start]

    first_holdout = (
        df_nitt_hold.groupby("customer_id")["date"]
        .min()
        .rename("first_holdout_date")
        .reset_index()
    )

    df_nitt_actual = first_holdout.merge(
        df_nitt_cal[["customer_id"]].drop_duplicates(), on="customer_id", how="right"
    )

    df_nitt_actual["actual_nitt_weeks"] = (
        df_nitt_actual["first_holdout_date"] - pd.to_datetime(training_end)
    ).dt.days.floordiv(7)

    df_nitt_actual = df_nitt_actual[df_nitt_actual["actual_nitt_weeks"].notna()]
    return df_nitt_actual


def next_buy_time(path):
    """Find index of first non-zero value in path"""
    for i, value in enumerate(path):
        if value != 0:
            return i
    return len(path)


def calculate_nitt_multiple_simulations(scenario_df, df_nitt_actual):
    """Calculate NITT from multiple model simulations"""
    holdout_weeks = int(
        (pd.to_datetime(holdout_end) - pd.to_datetime(holdout_start)).days / 7
    )

    all_nitts = {}
    scenario = scenario_df.copy()

    if "Unnamed: 0" in scenario.columns:
        scenario = scenario.rename(columns={"Unnamed: 0": "customer_id"})

    has_scenario_col = "scenario" in scenario.columns
    if has_scenario_col:
        scenario = scenario.drop("scenario", axis=1)

    transaction_cols = scenario.columns.difference(["customer_id"])
    holdout_cols = transaction_cols[-holdout_weeks:]

    for customer_id, customer_df in scenario.groupby("customer_id"):
        customer_transactions = customer_df[holdout_cols]

        has_transactions = customer_transactions.sum(axis=1) > 0
        valid_scenarios = customer_transactions[has_transactions]

        if len(valid_scenarios) > 0:
            customer_nitts = []
            for _, scenario_data in valid_scenarios.iterrows():
                customer_nitts.append(next_buy_time(scenario_data))

            all_nitts[customer_id] = sum(customer_nitts) / len(customer_nitts)

    pred_nitt_df = pd.DataFrame(
        list(all_nitts.items()), columns=["customer_id", "predicted_nitt_weeks"]
    )

    combined_df = pred_nitt_df.merge(
        df_nitt_actual[["customer_id", "actual_nitt_weeks"]],
        on="customer_id",
        how="inner",
    )

    predicted_avg = combined_df["predicted_nitt_weeks"].mean()
    actual_avg = combined_df["actual_nitt_weeks"].mean()
    bias = 100 * (predicted_avg - actual_avg) / actual_avg
    rmse = np.sqrt(
        mean_squared_error(
            combined_df["actual_nitt_weeks"], combined_df["predicted_nitt_weeks"]
        )
    )

    return predicted_avg, bias, rmse


def evaluate_opportunity_customers(holdout_predictions, df_cohort, min_freq=1):
    """Evaluate opportunity customer identification across models"""
    true_opportunity_cust = customers_more_active_in_holdout(
        df=df_cohort, df_holdout_pred=None
    )

    results = []
    for model_name, pred_df in holdout_predictions:
        opp_ids_predicted = customers_more_active_in_holdout(
            df=df_cohort, df_holdout_pred=pred_df, min_hold_tx=min_freq
        )

        share, opp_tx, total_tx = opportunity_tx_share(df_cohort, opp_ids_predicted)

        pred_df["opp_customer_actual"] = (
            pred_df["customer_id"].isin(true_opportunity_cust).astype(int)
        )
        pred_df["opp_customer_predicted"] = (
            pred_df["customer_id"].isin(opp_ids_predicted).astype(int)
        )

        zero_predictions = np.zeros(len(pred_df["opp_customer_actual"]))

        conf_matrix = confusion_matrix(
            pred_df["opp_customer_actual"], pred_df["opp_customer_predicted"]
        )
        conf_matrix_benchmark = confusion_matrix(
            pred_df["opp_customer_actual"], zero_predictions
        )

        precision = precision_score(
            pred_df["opp_customer_actual"], pred_df["opp_customer_predicted"]
        )
        recall = recall_score(
            pred_df["opp_customer_actual"], pred_df["opp_customer_predicted"]
        )
        f1 = f1_score(pred_df["opp_customer_actual"], pred_df["opp_customer_predicted"])
        accuracy = accuracy_score(
            pred_df["opp_customer_actual"], pred_df["opp_customer_predicted"]
        )
        accuracy_benchmark = accuracy_score(
            pred_df["opp_customer_actual"], zero_predictions
        )

        results.append(
            {
                "model": model_name,
                "opp_customers_predicted": len(opp_ids_predicted),
                "opp_tx": opp_tx,
                "total_tx": total_tx,
                "tx_share": share,
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "f1_score": round(f1, 2),
                "accuracy": round(accuracy, 2),
                "accuracy_benchmark": round(accuracy_benchmark, 2),
            }
        )

    return pd.DataFrame(results)


def calculate_seasonal_bias(
    aggregate_weekly_predictions, aggregate_counts_seasonality, periods
):
    """Calculate bias for seasonal peaks"""
    holdout_weeks = 52
    slices = []

    for model_name, df in aggregate_weekly_predictions:
        df_holdout = df.tail(holdout_weeks).copy()

        if "btyd" in model_name:
            df_holdout = df_holdout.reset_index()
            df_holdout["week"] = df_holdout.index + 1

        for period_name, weeks in periods.items():
            sub = df_holdout[df_holdout["week"].isin(weeks)].copy()
            sub["model"] = model_name
            sub["period"] = period_name
            slices.append(sub)

    df_periodic = pd.concat(slices, ignore_index=True)
    df_periodic = df_periodic[["model", "period", "week", "transactions"]]

    actuals = aggregate_counts_seasonality.copy()
    actuals["period"] = None
    for period_name, weeks in periods.items():
        actuals.loc[actuals["week"].isin(weeks), "period"] = period_name
    actuals = actuals.dropna(subset=["period"])

    records = []
    for (model, period), grp in df_periodic.groupby(["model", "period"]):
        pred = grp.set_index("week")["transactions"]

        act = aggregate_counts_seasonality.loc[
            aggregate_counts_seasonality["week"].isin(pred.index)
        ].set_index("week")["transactions"]

        common_weeks = pred.index.intersection(act.index)
        p = pred.loc[common_weeks]
        a = act.loc[common_weeks]

        total_pred = p.sum()
        total_act = a.sum()
        bias_pct = 100 * (total_pred - total_act) / total_act
        rmse = np.sqrt(mean_squared_error(a, p))

        records.append(
            {
                "model": model,
                "period": period,
                "weeks": common_weeks.to_list(),
                "actual": round(total_act, 0),
                "predicted": round(total_pred, 0),
                "bias_pct": round(bias_pct, 2),
                "bias_abs": round(abs(bias_pct), 2),
            }
        )

    return pd.DataFrame(records)
