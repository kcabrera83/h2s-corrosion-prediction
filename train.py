"""Training script for H2S corrosion prediction models."""

import os
import sys
import time

import numpy as np
import pandas as pd

from h2s_corrosion.data_generator import CorrosionDataGenerator
from h2s_corrosion.models.corrosion_predictor import CorrosionPredictor
from h2s_corrosion.models.life_estimator import LifeEstimator
from h2s_corrosion.utils.preprocessor import CorrosionPreprocessor

MODEL_DIR = os.path.join("outputs", "models")


def main():
    print("=" * 60)
    print("  H2S Corrosion Prediction - Model Training")
    print("=" * 60)

    # Generate data
    print("\n[1/5] Generating synthetic corrosion data...")
    gen = CorrosionDataGenerator(seed=42)
    df = gen.generate(n_samples=3000)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Preprocess
    print("\n[2/5] Preprocessing data...")
    preprocessor = CorrosionPreprocessor()
    X_train, X_test, y_train_c, y_test_c, pipeline = preprocessor.prepare(
        df, target="corrosion_rate_mpy"
    )
    X_train_l, X_test_l, y_train_l, y_test_l, _ = preprocessor.prepare(
        df, target="remaining_useful_life_years"
    )
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")

    # Train corrosion predictor
    print("\n[3/5] Training GradientBoosting (corrosion rate)...")
    t0 = time.time()
    predictor = CorrosionPredictor()
    predictor.fit(X_train, y_train_c)
    _, metrics_c = predictor.evaluate(X_test, y_test_c)
    print(f"  Time: {time.time() - t0:.2f}s")
    for k, v in metrics_c.items():
        print(f"  {k.upper()}: {v}")

    # Train life estimator
    print("\n[4/5] Training RandomForest (remaining useful life)...")
    t0 = time.time()
    estimator = LifeEstimator()
    estimator.fit(X_train, y_train_l)
    _, metrics_l = estimator.evaluate(X_test, y_test_l)
    print(f"  Time: {time.time() - t0:.2f}s")
    for k, v in metrics_l.items():
        print(f"  {k.upper()}: {v}")

    # Save models
    print("\n[5/5] Saving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    predictor.save(MODEL_DIR)
    estimator.save(MODEL_DIR)
    import pickle as _pkl
    with open(os.path.join(MODEL_DIR, "pipeline.pkl"), "wb") as _f:
        _pkl.dump(pipeline, _f)
    print(f"  Models saved to {MODEL_DIR}/")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
