
"""
Streaming benchmarks for NumCompute.

Compares different buffer sizes based on:
- Model accuracy
- Total retraining time
- Average processing time per chunk
"""

import time
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


def generate_data(n_samples=1000, random_state=42):
    """
    Generate a synthetic binary classification dataset.
    """
    rng = np.random.default_rng(random_state)

    X = rng.normal(0, 1, (n_samples, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    return X, y


def split_chunks(X, y, chunk_size=50):
    """
    Split data into smaller chunks.
    """
    X_chunks = []
    y_chunks = []

    for start in range(0, len(X), chunk_size):
        end = start + chunk_size

        X_chunks.append(X[start:end])
        y_chunks.append(y[start:end])

    return X_chunks, y_chunks


def benchmark_buffer_size(
    X_train,
    y_train,
    X_test,
    y_test,
    buffer_size,
    chunk_size=50
):
    """
    Measure accuracy and training time for one buffer size.
    """

    model = DecisionTreeClassifier(
        max_depth=4,
        buffer_size=buffer_size,
        random_state=42
    )

    X_chunks, y_chunks = split_chunks(
        X_train,
        y_train,
        chunk_size
    )

    start_time = time.perf_counter()

    for X_chunk, y_chunk in zip(X_chunks, y_chunks):
        model.partial_fit(X_chunk, y_chunk)

    total_time = time.perf_counter() - start_time

    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    return {
        "buffer_size": buffer_size,
        "accuracy": accuracy,
        "training_time": total_time,
        "average_chunk_time": total_time / len(X_chunks)
    }


def run_benchmarks():
    """
    Compare different buffer sizes.
    """

    X, y = generate_data()

    # Split into training and testing data
    X_train = X[:800]
    y_train = y[:800]

    X_test = X[800:]
    y_test = y[800:]

    buffer_sizes = [50, 100, 200, 400]

    results = []

    print("Streaming Buffer Size Benchmark")
    print("-" * 65)

    for buffer_size in buffer_sizes:

        result = benchmark_buffer_size(
            X_train,
            y_train,
            X_test,
            y_test,
            buffer_size
        )

        results.append(result)

        print(f"\nBuffer size: {result['buffer_size']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Total training time: {result['training_time']:.6f} seconds")
        print(
            f"Average chunk time: "
            f"{result['average_chunk_time']:.6f} seconds"
        )

    print("\nBenchmark Summary")
    print("-" * 65)
    print(
        f"{'Buffer Size':<15}"
        f"{'Accuracy':<15}"
        f"{'Training Time (s)':<20}"
    )

    for result in results:
        print(
            f"{result['buffer_size']:<15}"
            f"{result['accuracy']:<15.4f}"
            f"{result['training_time']:<20.6f}"
        )

    return results


if __name__ == "__main__":
    run_benchmarks()
