library(BTYD)
library(BTYDplus)

# Data preprocessing -----------------------------------------------------------

# Load transaction data with customer_id and date columns
df <- read.csv("data/transactions.csv", header = TRUE, stringsAsFactors = FALSE)[, c("customer_id", "date")]

# Rename customer_id to cust for BTYD compatibility
colnames(df)[colnames(df) == "customer_id"] <- "cust"

# Convert date column to Date type
df$date <- as.Date(df$date)

# Define calibration and holdout periods
train_start <- as.Date("2021-01-01")
train_end <- as.Date("2023-12-31")
hold_start <- as.Date("2024-01-01")
hold_end <- as.Date("2024-12-31")

# Calculate recency/frequency statistics per customer
cbs <- elog2cbs(df, T.cal = train_end, T.tot = hold_end)

# MBG/NBD model estimation and prediction -------------------------------------

# Estimate MBG/NBD parameters via MLE
mbgnbd_params <- mbgcnbd.EstimateParameters(
  cal.cbs = cbs,
  k = 1,
  par.start = c(1, 3, 1, 3),
  max.param.value = 1e4,
  trace = 0
)

# Generate cumulative expected transactions for 52 weeks
mbgnbd_cum_pred <- sapply(
  1:52,
  function(t) {
    mbgcnbd.ConditionalExpectedTransactions(
      params = mbgnbd_params,
      T.star = t,
      x = cbs$x,
      t.x = cbs$t.x,
      T.cal = cbs$T.cal
    )
  }
)

# Calculate metrics for MBG/NBD
mbgnbd_actual <- cbs$x.star
mbgnbd_pred52 <- mbgnbd_cum_pred[, 52]
mbgnbd_errors <- mbgnbd_actual - mbgnbd_pred52
mbgnbd_MAE <- mean(abs(mbgnbd_errors))
mbgnbd_RMSE <- sqrt(mean(mbgnbd_errors^2))
mbgnbd_bias_pct <- 100 * (sum(mbgnbd_pred52) - sum(mbgnbd_actual)) / sum(mbgnbd_actual)

# BG/NBD model estimation and prediction --------------------------------------

# Estimate BG/NBD parameters via MLE
bgnbd_params <- BTYD::bgnbd.EstimateParameters(
  cal.cbs = cbs[, c("x", "t.x", "T.cal")],
  par.start = c(1, 1, 1, 1),
  max.param.value = 10000,
  method = "L-BFGS-B",
  hardie = TRUE
)

# Generate cumulative expected transactions for 52 weeks
bgnbd_cum_pred <- sapply(
  1:52,
  function(t) {
    BTYD::bgnbd.ConditionalExpectedTransactions(
      params = bgnbd_params,
      T.star = t,
      x = cbs$x,
      t.x = cbs$t.x,
      T.cal = cbs$T.cal,
      hardie = TRUE
    )
  }
)

# Calculate metrics for BG/NBD
bgnbd_actual <- cbs$x.star
bgnbd_pred52 <- bgnbd_cum_pred[, 52]
bgnbd_errors <- bgnbd_actual - bgnbd_pred52
bgnbd_MAE <- mean(abs(bgnbd_errors))
bgnbd_RMSE <- sqrt(mean(bgnbd_errors^2))
bgnbd_bias_pct <- 100 * (sum(bgnbd_pred52) - sum(bgnbd_actual)) / sum(bgnbd_actual)

# NITT (Next Inter-Transaction Time) prediction -------------------------------

# Load actual NITT data for validation
actual_nitt_df <- read.csv("actual_nitt.csv")
cust_ids <- cbs$cust
actual_nitt <- actual_nitt_df$actual_nitt_weeks[
  match(cust_ids, actual_nitt_df$customer_id)
]

# Function to simulate NITT predictions
simulate_nitt <- function(cum_pred, actual_nitt, cust_ids, model_id, M = 1000, seed = 123) {
  set.seed(seed)
  N <- nrow(cum_pred)
  H <- ncol(cum_pred)
  
  # Calculate incremental expectations
  indiv_pred <- matrix(NA_real_, nrow = N, ncol = H)
  indiv_pred[, 1] <- cum_pred[, 1]
  indiv_pred[, 2:H] <- cum_pred[, 2:H] - cum_pred[, 1:(H-1)]
  
  # Monte Carlo simulation of first purchase week
  sims <- matrix(NA_integer_, nrow = N, ncol = M)
  for (i in seq_len(N)) {
    probs <- indiv_pred[i, ]
    cum_probs <- cumsum(probs)
    total <- sum(probs)
    for (m in seq_len(M)) {
      u <- runif(1)
      if (u <= total) sims[i, m] <- which(cum_probs >= u)[1L]
    }
  }
  
  # Point prediction as average simulated week
  predicted <- rowMeans(sims, na.rm = TRUE)
  
  # Calculate metrics
  valid <- !is.na(actual_nitt) & !is.na(predicted)
  rmse <- sqrt(mean((predicted[valid] - actual_nitt[valid])^2))
  bias <- mean(predicted[valid] - actual_nitt[valid])
  pct_bias <- 100 * sum(predicted[valid] - actual_nitt[valid]) / sum(actual_nitt[valid])
  
  return(list(
    predicted = predicted,
    rmse = rmse,
    bias = bias,
    pct_bias = pct_bias
  ))
}

# Generate NITT predictions for both models
results_mbgnbd_nitt <- simulate_nitt(
  cum_pred = mbgnbd_cum_pred,
  actual_nitt = actual_nitt,
  cust_ids = cust_ids,
  model_id = "mbgnbd"
)

results_bgnbd_nitt <- simulate_nitt(
  cum_pred = bgnbd_cum_pred,
  actual_nitt = actual_nitt,
  cust_ids = cust_ids,
  model_id = "bgnbd"
)

# Output results ---------------------------------------------------------------

# Model performance comparison
cat("Model Performance Summary:\n")
cat(sprintf("MBG/NBD - MAE: %.4f, RMSE: %.4f, Bias: %.2f%%\n", 
            mbgnbd_MAE, mbgnbd_RMSE, mbgnbd_bias_pct))
cat(sprintf("BG/NBD - MAE: %.4f, RMSE: %.4f, Bias: %.2f%%\n", 
            bgnbd_MAE, bgnbd_RMSE, bgnbd_bias_pct))

cat("\nNITT Prediction Performance:\n")
cat(sprintf("MBG/NBD NITT - RMSE: %.4f, Bias: %.2f%%\n", 
            results_mbgnbd_nitt$rmse, results_mbgnbd_nitt$pct_bias))
cat(sprintf("BG/NBD NITT - RMSE: %.4f, Bias: %.2f%%\n", 
            results_bgnbd_nitt$rmse, results_bgnbd_nitt$pct_bias)) 