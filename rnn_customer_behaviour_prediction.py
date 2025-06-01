import logging
import tensorflow as tf
import numpy as np
import random
import os
import datetime
import tensorflow_probability as tfp
import pandas as pd
import tensorflow.keras.backend as K
from tqdm.auto import tqdm
from enum import Enum
import optuna
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Concatenate,
    Input,
    Lambda,
    SimpleRNN,
    GRU,
    Dropout,
)
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    SwapEMAWeights,
    CSVLogger,
)
import concurrent.futures
import itertools


class ModelType(Enum):
    SIMPLE_RNN = "simple_rnn"
    GRU = "gru"
    LSTM = "lstm"


class RNNCustomerPredictor:
    def __init__(
        self,
        model_type=ModelType.LSTM,
        max_epochs=10,
        finetune_epochs=1,
        n_scenarios=2,
        validation_split=0.25,
        batch_size_train=32,
    ):
        self.model_type = model_type
        self.max_epochs = max_epochs
        self.finetune_epochs = finetune_epochs
        self.n_scenarios = n_scenarios
        self.validation_split = validation_split
        self.batch_size_train = batch_size_train
        self.learning_rate = 1e-3
        self.finetune_lr = 1e-4

        # Model parameters for each architecture
        self.model_params = {
            ModelType.SIMPLE_RNN: {
                "n_memory_layers": 1,
                "n_dense_layers": 1,
                "memory_units": 32,
                "dense_units": 128,
                "learning_rate": 0.004067554302665728,
                "dropout": 0.3,
                "recurrent_dropout": 0.0,
                "dense_dropout": 0.1,
            },
            ModelType.GRU: {
                "n_memory_layers": 2,
                "n_dense_layers": 1,
                "memory_units": 64,
                "dense_units": 256,
                "learning_rate": 0.0006574024292501898,
                "dropout": 0.0,
                "recurrent_dropout": 0.0,
                "dense_dropout": 0.3,
            },
            ModelType.LSTM: {
                "n_memory_layers": 2,
                "n_dense_layers": 1,
                "memory_units": 128,
                "dense_units": 256,
                "learning_rate": 0.002067285247188307,
                "dropout": 0.0,
                "recurrent_dropout": 0.0,
                "dense_dropout": 0.0,
            },
        }

    def prepare_data(
        self, df, training_start, training_end, holdout_start, holdout_end, date_format
    ):
        """Prepare training data from customer transaction dataframe"""

        def date_range(start, end):
            start = datetime.datetime.strptime(start, date_format)
            end = datetime.datetime.strptime(end, date_format)
            r = (end + datetime.timedelta(days=1) - start).days
            return [start + datetime.timedelta(days=i) for i in range(r)]

        ed = pd.DataFrame(date_range(training_start, holdout_end), columns=["date"])
        ed["year"] = ed["date"].dt.year
        ed["week"] = (ed["date"].dt.dayofyear // 7).clip(upper=51)

        self.holdout_calendar = (
            ed[ed["date"] >= holdout_start]
            .drop(columns=["date"])
            .drop_duplicates()
            .drop(columns=["year"])
        )

        samples, targets, calibration, holdout = [], [], [], []
        max_trans_per_week = 0

        ids = df["customer_id"].unique()
        random.shuffle(ids)

        for customer in tqdm(ids, desc="Preparing dataset"):
            subset = (
                df.query("customer_id == @customer")
                .groupby(["date"])
                .agg({"customer_id": "count"})
                .reset_index()
            )
            user = subset.copy().rename(columns={"customer_id": "transactions"})

            frame = ed.copy()
            frame["customer_id"] = customer
            frame = frame.merge(user, on=["date"], how="left")

            frame = (
                frame.groupby(["year", "week"])
                .agg({"transactions": "sum", "date": "min"})
                .sort_values(["date"])
                .reset_index()
            )

            max_trans_per_week = max(max_trans_per_week, max(frame["transactions"]))

            training = (
                frame[frame["date"] < holdout_start]
                .drop(columns=["date", "year"])
                .astype(int)
            )
            calibration.append(training)

            sample = training[:-1].values
            samples.append(sample)
            target = training.loc[1:, "transactions"].values
            targets.append(target)

            hold = frame[frame["date"] >= holdout_start].drop(columns="date")
            holdout.append(hold)

        self.max_trans_per_week = int(max_trans_per_week)
        self.seq_len = samples[0].shape[0]

        validation_size = round(len(samples) * self.validation_split)
        valid_samples, valid_targets = (
            samples[-validation_size:],
            targets[-validation_size:],
        )
        train_samples, train_targets = (
            samples[:-validation_size],
            targets[:-validation_size],
        )

        self.calibration = calibration
        self.holdout = holdout
        return train_samples, train_targets, valid_samples, valid_targets

    def decode_sample(self, sample, target):
        """Transform raw data into Keras compatible format"""
        tensor = (
            {
                "week": tf.cast(tf.expand_dims(sample[:, 0], axis=-1), "int32"),
                "transactions": tf.cast(tf.expand_dims(sample[:, 1], axis=-1), "int32"),
            },
            tf.cast(tf.expand_dims(target, axis=-1), "int32"),
        )
        return tensor

    def emb_size(self, feature_max):
        """Calculate embedding size"""
        return int(feature_max**0.5) + 1

    def build_lstm_model(self, params):
        """Build LSTM model"""
        max_week = 52
        max_trans = self.max_trans_per_week + 1

        base_feats = [("week", max_week), ("transactions", max_trans)]
        inputs, embeds = {}, []

        for name, vocab in base_feats:
            inp = Input(shape=(self.seq_len,), name=name)
            emb = Embedding(vocab, self.emb_size(vocab), name=f"emb_{name}")(inp)
            embeds.append(emb)
            inputs[name] = inp

        x = Concatenate(name="feature_concat")(embeds)

        for i in range(params["n_memory_layers"]):
            x = LSTM(
                params["memory_units"],
                return_sequences=True,
                dropout=params.get("dropout", 0.0),
                recurrent_dropout=params.get("recurrent_dropout", 0.0),
                name=f"lstm_{i}",
            )(x)

        for i in range(params["n_dense_layers"]):
            x = Dense(params["dense_units"], activation="relu", name=f"dense_{i}")(x)
            if params.get("dense_dropout", 0.0) > 0:
                x = Dropout(params["dense_dropout"], name=f"dropout_dense_{i}")(x)

        outputs = Dense(max_trans, activation="softmax", name="softmax")(x)
        model = Model(inputs=list(inputs.values()), outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"], use_ema=True, ema_momentum=0.99
        )
        model.compile(loss=sparse_categorical_crossentropy, optimizer=optimizer)
        return model

    def build_gru_model(self, params):
        """Build GRU model"""
        max_week = 52
        max_trans = self.max_trans_per_week + 1

        base_feats = [("week", max_week), ("transactions", max_trans)]
        inputs, embeds = {}, []

        for name, vocab in base_feats:
            inp = Input(shape=(self.seq_len,), name=name)
            emb = Embedding(vocab, self.emb_size(vocab), name=f"emb_{name}")(inp)
            embeds.append(emb)
            inputs[name] = inp

        x = Concatenate(name="feature_concat")(embeds)

        for i in range(params["n_memory_layers"]):
            x = GRU(
                params["memory_units"],
                return_sequences=True,
                dropout=params.get("dropout", 0.0),
                recurrent_dropout=params.get("recurrent_dropout", 0.0),
                name=f"gru_{i}",
            )(x)

        for i in range(params["n_dense_layers"]):
            x = Dense(params["dense_units"], activation="relu", name=f"dense_{i}")(x)
            if params.get("dense_dropout", 0.0) > 0:
                x = Dropout(params["dense_dropout"], name=f"dropout_dense_{i}")(x)

        outputs = Dense(max_trans, activation="softmax", name="softmax")(x)
        model = Model(inputs=list(inputs.values()), outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"], use_ema=True, ema_momentum=0.99
        )
        model.compile(loss=sparse_categorical_crossentropy, optimizer=optimizer)
        return model

    def build_simple_rnn_model(self, params):
        """Build Simple RNN model"""
        max_week = 52
        max_trans = self.max_trans_per_week + 1

        base_feats = [("week", max_week), ("transactions", max_trans)]
        inputs, embeds = {}, []

        for name, vocab in base_feats:
            inp = Input(shape=(self.seq_len,), name=name)
            emb = Embedding(vocab, self.emb_size(vocab), name=f"emb_{name}")(inp)
            embeds.append(emb)
            inputs[name] = inp

        x = Concatenate(name="feature_concat")(embeds)

        parallel_outputs = []
        for i in range(params["n_memory_layers"]):
            rnn_out = SimpleRNN(
                params["memory_units"],
                return_sequences=True,
                dropout=params.get("dropout", 0.0),
                recurrent_dropout=params.get("recurrent_dropout", 0.0),
                name=f"simple_rnn_{i}",
            )(x)
            parallel_outputs.append(rnn_out)

        x = (
            parallel_outputs[0]
            if len(parallel_outputs) == 1
            else Concatenate(name="parallel_concat")(parallel_outputs)
        )

        for i in range(params["n_dense_layers"]):
            x = Dense(params["dense_units"], activation="relu", name=f"dense_{i}")(x)
            if params.get("dense_dropout", 0.0) > 0:
                x = Dropout(params["dense_dropout"], name=f"dropout_dense_{i}")(x)

        outputs = Dense(max_trans, activation="softmax", name="softmax")(x)
        model = Model(inputs=list(inputs.values()), outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"], use_ema=True, ema_momentum=0.99
        )
        model.compile(loss=sparse_categorical_crossentropy, optimizer=optimizer)
        return model

    def build_model(self, params=None):
        """Build model based on type"""
        if params is None:
            params = self.model_params[self.model_type]

        if self.model_type == ModelType.LSTM:
            return self.build_lstm_model(params)
        elif self.model_type == ModelType.GRU:
            return self.build_gru_model(params)
        elif self.model_type == ModelType.SIMPLE_RNN:
            return self.build_simple_rnn_model(params)

    def tune_hyperparameters(
        self,
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        n_trials=100,
        seed=42,
    ):
        """Tune hyperparameters using Optuna"""
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        def objective(trial):
            params = {
                "n_memory_layers": trial.suggest_int("n_memory_layers", 1, 3),
                "n_dense_layers": trial.suggest_int("n_dense_layers", 1, 3),
                "dense_units": trial.suggest_categorical(
                    "dense_units", [32, 64, 128, 256]
                ),
                "memory_units": trial.suggest_categorical(
                    "memory_units", [32, 64, 128, 256]
                ),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
                "recurrent_dropout": trial.suggest_float(
                    "recurrent_dropout", 0.0, 0.5, step=0.1
                ),
                "dense_dropout": trial.suggest_float(
                    "dense_dropout", 0.0, 0.5, step=0.1
                ),
            }

            model = self.build_model(params)

            train_dataset = (
                tf.data.Dataset.from_tensor_slices((train_samples, train_targets))
                .map(self.decode_sample)
                .batch(self.batch_size_train)
                .repeat()
            )
            valid_dataset = (
                tf.data.Dataset.from_tensor_slices((valid_samples, valid_targets))
                .map(self.decode_sample)
                .batch(len(valid_samples))
                .repeat()
            )

            callbacks = [
                SwapEMAWeights(swap_on_epoch=False),
                EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                ),
            ]

            hist = model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=50,
                callbacks=callbacks,
                steps_per_epoch=len(train_samples) // self.batch_size_train,
                validation_steps=1,
                verbose=0,
            )
            return min(hist.history["val_loss"])

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.RandomSampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def train_model(
        self,
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        model_params=None,
        seed=1,
    ):
        """Train and fine-tune model"""
        K.clear_session()
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if model_params is None:
            model_params = self.model_params[self.model_type]

        # Prepare datasets
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_samples, train_targets))
            .repeat()
            .shuffle(buffer_size=len(train_samples), seed=seed)
            .map(self.decode_sample)
            .batch(self.batch_size_train)
            .prefetch(tf.data.AUTOTUNE)
        )

        valid_dataset = (
            tf.data.Dataset.from_tensor_slices((valid_samples, valid_targets))
            .map(self.decode_sample)
            .batch(len(valid_samples))
            .repeat()
        )

        # Build and train model
        model = self.build_model(model_params)

        callbacks = [
            SwapEMAWeights(swap_on_epoch=False),
            EarlyStopping("val_loss", patience=5, restore_best_weights=True, verbose=1),
        ]

        hist = model.fit(
            train_dataset,
            epochs=self.max_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_dataset,
            validation_steps=1,
            steps_per_epoch=len(train_samples) // self.batch_size_train,
        )

        # Fine-tune on full calibration data
        all_samples = train_samples + valid_samples
        all_targets = train_targets + valid_targets
        calib_ds = (
            tf.data.Dataset.from_tensor_slices((all_samples, all_targets))
            .map(self.decode_sample)
            .batch(self.batch_size_train * 4)
            .repeat()
        )

        optimizer = Adam(
            learning_rate=self.finetune_lr, use_ema=True, ema_momentum=0.99
        )
        model.compile(loss=sparse_categorical_crossentropy, optimizer=optimizer)

        model.fit(
            calib_ds,
            epochs=self.finetune_epochs,
            steps_per_epoch=len(all_samples) // (self.batch_size_train * 4),
            verbose=1,
        )

        return model, min(hist.history["val_loss"])

    def setup_prediction_model(self, trained_model, batch_size_pred=1024):
        """Setup model for prediction with sampling"""

        @tf.keras.utils.register_keras_serializable()
        def sample_multinomial(probs):
            samp = tfp.distributions.Categorical(probs=probs).sample()
            return tf.cast(
                tf.expand_dims(samp, axis=-1), dtype=tf.keras.backend.floatx()
            )

        sample_layer = Lambda(sample_multinomial, name="sample_transactions")
        feature_names = ["week", "transactions"]

        pred_inputs = [
            Input(batch_shape=(batch_size_pred, None), name=feat, dtype="int32")
            for feat in feature_names
        ]

        def clone_with_stateful(layer):
            if isinstance(
                layer,
                (tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.SimpleRNN),
            ):
                cfg = layer.get_config()
                cfg["stateful"] = True
                return layer.__class__.from_config(cfg)
            return layer.__class__.from_config(layer.get_config())

        base_pred = clone_model(
            trained_model, input_tensors=pred_inputs, clone_function=clone_with_stateful
        )
        pred_probs = base_pred.outputs[0]
        pred_sample = sample_layer(pred_probs)

        return Model(inputs=pred_inputs, outputs=pred_sample, name="prediction_model")

    def create_predictions(self, prediction_model, seed, batch_size_pred=1024):
        """Generate predictions using the trained model"""
        no_samples = seed.shape[0]
        no_timesteps = seed.shape[1]
        no_features = seed.shape[2]
        no_batches = int(np.ceil(no_samples / batch_size_pred))
        holdout_length = self.holdout[0].shape[0]

        # Pad if needed
        if seed.shape[0] < (batch_size_pred * no_batches):
            padding = np.zeros(
                ((batch_size_pred * no_batches) - no_samples, no_timesteps, no_features)
            )
            seed = np.concatenate((seed, padding), axis=0)

        scenarios = []
        for _ in tqdm(range(self.n_scenarios), desc="Simulating scenarios"):
            batches_predicted = []
            for j in range(no_batches):
                pred = []

                # Reset LSTM states
                for layer in prediction_model.layers:
                    if hasattr(layer, "reset_states"):
                        layer.reset_states()

                batch_start = j * batch_size_pred
                batch_end = (j + 1) * batch_size_pred

                batch = {
                    "week": seed[batch_start:batch_end, :, 0:1],
                    "transactions": seed[batch_start:batch_end, :, 1:2],
                }

                prediction = prediction_model.predict(
                    batch, batch_size=batch_size_pred, verbose=0
                )
                pred.append(prediction[:, :, :])

                # Autoregressive prediction
                for i in range(holdout_length - 1):
                    batch = {}
                    feature = np.repeat(
                        self.holdout_calendar.iloc[i]["week"], batch_size_pred
                    )
                    batch["week"] = feature[:, np.newaxis, np.newaxis]
                    batch["transactions"] = pred[-1][:, -1:, :]

                    prediction = prediction_model.predict(
                        batch, batch_size=batch_size_pred, verbose=0
                    )
                    pred.append(prediction[:, :, :])

                batches_predicted.append(pred)
            scenarios.append(batches_predicted)

        return scenarios

    def evaluate_predictions(self, scenarios, df_holdout, no_samples):
        """Evaluate prediction accuracy"""
        # Process scenarios into final predictions
        z = []
        for scenario in scenarios:
            y = []
            for batch in scenario:
                x = []
                for time_step in batch:
                    if type(time_step) == np.ndarray:
                        x.append(time_step)
                    else:
                        complete_time_step = np.concatenate(time_step, axis=-1)
                        x.append(complete_time_step)
                y.append(np.concatenate(x, axis=1))
            z.append(np.concatenate(y, axis=0)[:no_samples, :, :])

        predictions = np.asarray(z)

        # Calculate individual predictions
        individual_predictions = pd.DataFrame(
            np.squeeze(np.mean(predictions, axis=0))
        ).set_index(pd.Series(df_holdout["customer_id"].unique()))

        holdout_length = self.holdout[0].shape[0]
        individual_predictions_holdout = individual_predictions.iloc[
            :, -holdout_length:
        ]
        individual_predictions_holdout = pd.DataFrame(
            np.squeeze(np.sum(individual_predictions_holdout, axis=1)),
            columns=["total_holdout_transactions"],
        )
        individual_predictions_holdout.index.name = "customer_id"

        # Get actual transactions
        df_holdout_transactions = (
            df_holdout.groupby("customer_id")["date"].count().reset_index()
        )
        individual_predictions_holdout = pd.merge(
            individual_predictions_holdout,
            df_holdout_transactions,
            on="customer_id",
            how="left",
        )
        individual_predictions_holdout = individual_predictions_holdout.rename(
            columns={"date": "actual_holdout_transactions"}
        )
        individual_predictions_holdout["actual_holdout_transactions"] = (
            individual_predictions_holdout["actual_holdout_transactions"].fillna(0)
        )

        # Calculate metrics
        individual_predictions_holdout["absolute_error"] = np.abs(
            individual_predictions_holdout["total_holdout_transactions"]
            - individual_predictions_holdout["actual_holdout_transactions"]
        )

        mae = np.mean(individual_predictions_holdout["absolute_error"])
        rmse = np.sqrt(np.mean(individual_predictions_holdout["absolute_error"] ** 2))

        # Calculate bias on aggregate level
        predicted_total = individual_predictions_holdout[
            "total_holdout_transactions"
        ].sum()
        actual_total = individual_predictions_holdout[
            "actual_holdout_transactions"
        ].sum()
        bias = (
            100 * (predicted_total - actual_total) / actual_total
            if actual_total > 0
            else 0
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "individual_predictions": individual_predictions_holdout,
            "aggregate_predictions": np.squeeze(
                np.sum(np.mean(predictions, axis=0), axis=0)
            ),
        }


# Usage example
def main():
    # Load your data
    df = pd.read_csv("your_data.csv", parse_dates=["date"])

    # Initialize predictor
    predictor = RNNCustomerPredictor(
        model_type=ModelType.LSTM, max_epochs=10, finetune_epochs=1
    )

    # Prepare data
    train_samples, train_targets, valid_samples, valid_targets = predictor.prepare_data(
        df, "2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "%Y-%m-%d"
    )

    # Optional: Tune hyperparameters
    # best_params = predictor.tune_hyperparameters(train_samples, train_targets, valid_samples, valid_targets, n_trials=50)

    # Train model
    trained_model, val_loss = predictor.train_model(
        train_samples, train_targets, valid_samples, valid_targets
    )

    # Setup for prediction
    prediction_model = predictor.setup_prediction_model(trained_model)

    # Create prediction seed
    seed = np.array([df.values for df in predictor.calibration], dtype=np.float32)

    # Generate predictions
    scenarios = predictor.create_predictions(prediction_model, seed)

    # Evaluate
    df_holdout = df[df["date"] >= "2024-01-01"]
    results = predictor.evaluate_predictions(scenarios, df_holdout, seed.shape[0])

    print(f"MAE: {results['mae']:.2f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"Bias: {results['bias']:.2f}%")


if __name__ == "__main__":
    main()
