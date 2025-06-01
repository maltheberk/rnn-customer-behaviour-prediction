# Thesis code listings: RNN Customer Behavior Prediction

Code listings for my master thesis project on predicting customer behaviour using a RNN-based Framework. This code builds on the framework proposed by Valendin et. al (2022), by including GRU and SimpleRNN variants and hyperparameter tuning.

A TensorFlow-based implementation for predicting customer transaction behavior using Recurrent Neural Networks (RNN). This project supports multiple RNN architectures and includes comprehensive hyperparameter tuning capabilities.

## Features

- **Multiple RNN Architectures**: LSTM, GRU, and Simple RNN models
- **Hyperparameter Tuning**: Automated optimization using Optuna
- **Two-Phase Training**: Initial training followed by fine-tuning on full calibration data
- **Scenario Generation**: Monte Carlo simulation for uncertainty quantification
- **Comprehensive Evaluation**: MAE, RMSE, and bias metrics
- **Time Series Preprocessing**: Automatic conversion of daily transactions to weekly aggregates

## Installation

```bash
pip install tensorflow tensorflow-probability pandas numpy optuna tqdm
```

## Quick Start

```python
import pandas as pd
from rnn_customer_behaviour_prediction_clean import RNNCustomerPredictor, ModelType

# Load your transaction data
df = pd.read_csv("customer_transactions.csv", parse_dates=["date"])

# Initialize predictor
predictor = RNNCustomerPredictor(
    model_type=ModelType.LSTM,
    max_epochs=50,
    finetune_epochs=5,
    n_scenarios=10
)

# Prepare data (training: 2021-2023, holdout: 2024)
train_samples, train_targets, valid_samples, valid_targets = predictor.prepare_data(
    df, "2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "%Y-%m-%d"
)

# Train model
trained_model, val_loss = predictor.train_model(
    train_samples, train_targets, valid_samples, valid_targets
)

# Generate predictions
prediction_model = predictor.setup_prediction_model(trained_model)
seed = np.array([df.values for df in predictor.calibration], dtype=np.float32)
scenarios = predictor.create_predictions(prediction_model, seed)

# Evaluate results
df_holdout = df[df["date"] >= "2024-01-01"]
results = predictor.evaluate_predictions(scenarios, df_holdout, seed.shape[0])

print(f"MAE: {results['mae']:.2f}")
print(f"RMSE: {results['rmse']:.2f}")
print(f"Bias: {results['bias']:.2f}%")
```

## Data Format

Your input DataFrame should contain:
- `customer_id`: Unique identifier for each customer
- `date`: Transaction date (datetime format)
- Additional columns are ignored

Example:
```csv
customer_id,date,amount
12345,2021-01-15,49.99
12345,2021-01-22,29.99
67890,2021-01-10,99.99
```

## Model Architectures

### LSTM (Default)
Optimized for capturing long-term dependencies in customer behavior patterns.

### GRU
Computationally efficient alternative to LSTM with competitive performance.

### Simple RNN
Lightweight option for simpler patterns or resource-constrained environments.

## Configuration

### Key Parameters

```python
predictor = RNNCustomerPredictor(
    model_type=ModelType.LSTM,        # LSTM, GRU, or SIMPLE_RNN
    max_epochs=50,                    # Initial training epochs
    finetune_epochs=5,                # Fine-tuning epochs
    n_scenarios=10,                   # Number of prediction scenarios
    validation_split=0.25,            # Validation set size
    batch_size_train=32               # Training batch size
)
```

### Pre-tuned Hyperparameters

The model includes optimized hyperparameters for each architecture:

- **LSTM**: 2 memory layers, 128 units, 256 dense units
- **GRU**: 2 memory layers, 64 units, 256 dense units  
- **Simple RNN**: 1 memory layer, 32 units, 128 dense units

## Hyperparameter Tuning

```python
# Tune hyperparameters automatically
best_params = predictor.tune_hyperparameters(
    train_samples, train_targets, 
    valid_samples, valid_targets, 
    n_trials=100
)

# Train with optimized parameters
trained_model, val_loss = predictor.train_model(
    train_samples, train_targets, 
    valid_samples, valid_targets, 
    model_params=best_params
)
```

## Prediction Process

1. **Data Preparation**: Converts daily transactions to weekly aggregates
2. **Training**: Two-phase approach with early stopping and EMA weights
3. **Fine-tuning**: Additional training on full calibration dataset
4. **Prediction Setup**: Converts model to stateful mode for autoregressive forecasting
5. **Scenario Generation**: Monte Carlo simulation with probabilistic sampling
6. **Evaluation**: Comprehensive metrics calculation

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
- **Bias**: Percentage difference between predicted and actual totals

## Advanced Usage

### Custom Model Parameters

```python
custom_params = {
    "n_memory_layers": 3,
    "n_dense_layers": 2,
    "memory_units": 256,
    "dense_units": 512,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "recurrent_dropout": 0.1,
    "dense_dropout": 0.3
}

trained_model, val_loss = predictor.train_model(
    train_samples, train_targets,
    valid_samples, valid_targets,
    model_params=custom_params
)
```

### Ensemble Predictions

```python
# Train multiple models with different seeds
models = []
for seed in [1, 2, 3, 4, 5]:
    model, _ = predictor.train_model(
        train_samples, train_targets,
        valid_samples, valid_targets,
        seed=seed
    )
    models.append(model)

# Average predictions from multiple models for better accuracy
```

## Time Complexity

- **Training**: O(T × N × H) where T=time steps, N=customers, H=hidden units
- **Prediction**: O(T × N × S) where S=number of scenarios
- **Memory**: O(N × T × F) where F=number of features

## Limitations

- Requires at least 2 years of transaction history for reliable predictions
- Weekly aggregation may lose important daily patterns
- Assumes stationary customer behavior patterns
- Limited to univariate transaction count prediction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@software{rnn_customer_prediction,
  title={RNN Customer Behavior Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rnn-customer-prediction}
}
```