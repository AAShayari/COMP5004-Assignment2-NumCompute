
import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.preprocessing import (
    StandardScaler,
    Imputer,
    OneHotEncoder,
)
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute.stats import StreamingStats
from numcompute.metrics import StreamingMetrics


# --------------------------------------------------
# Shared sample data
# --------------------------------------------------

@pytest.fixture
def sample_data():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# --------------------------------------------------
# Decision tree tests
# --------------------------------------------------

def test_tree_fit_and_predict(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.array_equal(predictions, y)


def test_tree_score(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)

    assert model.score(X, y) == 1.0


def test_tree_partial_fit_buffers_small_chunk(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=10)

    model.partial_fit(X[:2], y[:2])

    assert model.buffered_samples_ == 2
    assert model.is_fitted_ is False


def test_tree_partial_fit_retrains_when_buffer_is_full(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=4)

    model.partial_fit(X[:2], y[:2])
    assert model.is_fitted_ is False

    model.partial_fit(X[2:4], y[2:4])

    assert model.is_fitted_ is True
    assert model.buffered_samples_ == 0


def test_tree_partial_fit_clears_buffer_after_retraining(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=3)

    model.partial_fit(X[:3], y[:3])

    assert model.is_fitted_ is True
    assert model.buffered_samples_ == 0


def test_tree_predict_before_fit_raises_error(sample_data):
    X, _ = sample_data
    model = DecisionTreeClassifier()

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_tree_handles_missing_feature_values():
    X = np.array([
        [1.0, 10.0],
        [2.0, np.nan],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert np.all(np.isin(predictions, [0, 1]))


def test_tree_rejects_invalid_criterion():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="invalid")


def test_tree_rejects_invalid_buffer_size():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(buffer_size=0)


def test_tree_rejects_invalid_max_depth():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=-1)


def test_tree_rejects_feature_count_mismatch(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


# --------------------------------------------------
# StandardScaler tests
# --------------------------------------------------

def test_scaler_partial_fit_updates_statistics():
    scaler = StandardScaler()

    scaler.partial_fit(np.array([[1.0], [3.0]]))
    scaler.partial_fit(np.array([[5.0], [7.0]]))

    assert scaler.count_[0] == 4
    assert scaler.mean_[0] == pytest.approx(4.0)


def test_scaler_transform_has_zero_mean():
    X = np.array([[1.0], [2.0], [3.0]])
    scaler = StandardScaler()
    transformed = scaler.fit_transform(X)

    assert np.mean(transformed) == pytest.approx(0.0)


def test_scaler_constant_feature_does_not_divide_by_zero():
    X = np.array([[5.0], [5.0], [5.0]])
    scaler = StandardScaler()
    transformed = scaler.fit_transform(X)

    assert np.all(np.isfinite(transformed))
    assert np.all(transformed == 0.0)


def test_scaler_rejects_changed_feature_count():
    scaler = StandardScaler()
    scaler.partial_fit(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        scaler.partial_fit(np.array([[1.0, 2.0, 3.0]]))


def test_scaler_rejects_infinity():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.partial_fit(np.array([[1.0], [np.inf]]))


# --------------------------------------------------
# Imputer tests
# --------------------------------------------------

def test_imputer_updates_mean_across_chunks():
    imputer = Imputer(strategy="mean")

    imputer.partial_fit(np.array([[1.0], [3.0]]))
    imputer.partial_fit(np.array([[5.0], [np.nan]]))

    assert imputer.statistics_[0] == pytest.approx(3.0)


def test_imputer_replaces_missing_values_with_mean():
    X = np.array([[1.0], [np.nan], [3.0]])
    imputer = Imputer(strategy="mean")
    transformed = imputer.fit_transform(X)

    assert transformed[1, 0] == pytest.approx(2.0)


def test_imputer_constant_strategy():
    X = np.array([[1.0], [np.nan], [3.0]])
    imputer = Imputer(strategy="constant", fill_value=-1)
    transformed = imputer.fit_transform(X)

    assert transformed[1, 0] == -1


def test_imputer_uses_fill_value_for_all_missing_column():
    X = np.array([[np.nan], [np.nan]])
    imputer = Imputer(strategy="mean", fill_value=7)
    transformed = imputer.fit_transform(X)

    assert np.all(transformed == 7)


def test_imputer_rejects_invalid_strategy():
    with pytest.raises(ValueError):
        Imputer(strategy="median")


# --------------------------------------------------
# OneHotEncoder tests
# --------------------------------------------------

def test_encoder_learns_categories():
    X = np.array([["red"], ["blue"], ["red"]], dtype=object)
    encoder = OneHotEncoder()
    encoder.fit(X)

    assert set(encoder.categories_[0]) == {"red", "blue"}


def test_encoder_creates_binary_columns():
    X = np.array([["red"], ["blue"], ["red"]], dtype=object)
    encoder = OneHotEncoder()
    transformed = encoder.fit_transform(X)

    assert transformed.shape == (3, 2)
    assert np.all(np.sum(transformed, axis=1) == 1)


def test_encoder_learns_new_categories_incrementally():
    encoder = OneHotEncoder()

    encoder.partial_fit(np.array([["red"]], dtype=object))
    encoder.partial_fit(np.array([["blue"]], dtype=object))

    assert set(encoder.categories_[0]) == {"red", "blue"}


def test_encoder_ignores_unknown_categories_by_default():
    train = np.array([["red"], ["blue"]], dtype=object)
    test = np.array([["green"]], dtype=object)

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(train)
    transformed = encoder.transform(test)

    assert transformed.shape == (1, 2)
    assert np.all(transformed == 0)


def test_encoder_can_raise_for_unknown_categories():
    train = np.array([["red"], ["blue"]], dtype=object)
    test = np.array([["green"]], dtype=object)

    encoder = OneHotEncoder(handle_unknown="error")
    encoder.fit(train)

    with pytest.raises(ValueError):
        encoder.transform(test)


# --------------------------------------------------
# Streaming statistics tests
# --------------------------------------------------

def test_streaming_stats_updates_mean():
    stats = StreamingStats()

    stats.update_stats(np.array([[1.0], [3.0]]))
    stats.update_stats(np.array([[5.0], [7.0]]))

    assert stats.mean()[0] == pytest.approx(4.0)


def test_streaming_stats_updates_variance():
    stats = StreamingStats()
    stats.update_stats(np.array([[1.0], [2.0], [3.0]]))

    assert stats.variance()[0] == pytest.approx(1.0)


def test_streaming_stats_calculates_quantiles():
    stats = StreamingStats()
    stats.update_stats(np.array([[1.0], [2.0], [3.0], [4.0]]))

    result = stats.quantiles([0.5])

    assert result[0, 0] == pytest.approx(2.5)


def test_streaming_stats_calculates_histogram():
    stats = StreamingStats()
    stats.update_stats(np.array([[1.0], [2.0], [3.0], [4.0]]))

    counts, edges = stats.histogram(bins=2)

    assert np.sum(counts) == 4
    assert len(edges) == 3


def test_streaming_stats_rejects_changed_feature_count():
    stats = StreamingStats()
    stats.update_stats(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        stats.update_stats(np.array([[1.0, 2.0, 3.0]]))


# --------------------------------------------------
# Streaming metrics tests
# --------------------------------------------------

def test_streaming_metrics_calculates_accuracy():
    metrics = StreamingMetrics()
    metrics.update(
        np.array([0, 1, 1]),
        np.array([0, 1, 0])
    )

    assert metrics.result()["accuracy"] == pytest.approx(2 / 3)


def test_streaming_metrics_accumulates_across_chunks():
    metrics = StreamingMetrics()

    metrics.update(np.array([0, 1]), np.array([0, 1]))
    metrics.update(np.array([0, 1]), np.array([1, 1]))

    assert metrics.result()["accuracy"] == pytest.approx(0.75)


def test_streaming_metrics_calculates_confusion_matrix():
    metrics = StreamingMetrics()
    metrics.update(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1])
    )

    matrix = metrics.result()["confusion_matrix"]

    assert matrix.shape == (2, 2)
    assert matrix.sum() == 4


def test_streaming_metrics_calculates_rolling_accuracy():
    metrics = StreamingMetrics(window_size=2)

    metrics.update(np.array([0, 1]), np.array([0, 1]))
    metrics.update(np.array([0]), np.array([1]))

    assert metrics.result()["rolling_accuracy"] == pytest.approx(0.5)


def test_streaming_metrics_reset_clears_results():
    metrics = StreamingMetrics()
    metrics.update(np.array([0, 1]), np.array([0, 1]))
    metrics.reset()

    assert metrics.result() == {}


def test_streaming_metrics_calculates_auc_when_scores_are_provided():
    metrics = StreamingMetrics()

    metrics.update(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        y_score=np.array([0.1, 0.9, 0.2, 0.8])
    )

    assert metrics.result()["auc"] == pytest.approx(1.0)


def test_streaming_metrics_auc_is_none_without_scores():
    metrics = StreamingMetrics()
    metrics.update(np.array([0, 1]), np.array([0, 1]))

    assert metrics.result()["auc"] is None


# --------------------------------------------------
# Pipeline tests
# --------------------------------------------------

def test_pipeline_buffers_chunks_before_retraining(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=4)
    pipeline = Pipeline([("tree", model)])

    pipeline.partial_fit(X[:2], y[:2])

    assert pipeline.buffered_samples_ == 2
    assert pipeline._is_fitted is False


def test_pipeline_retrains_when_buffer_is_full(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=4)
    pipeline = Pipeline([("tree", model)])

    pipeline.partial_fit(X[:2], y[:2])
    pipeline.partial_fit(X[2:4], y[2:4])

    assert pipeline._is_fitted is True
    assert pipeline.buffered_samples_ == 0


def test_pipeline_predicts_after_retraining(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=4)
    pipeline = Pipeline([("tree", model)])

    pipeline.partial_fit(X[:4], y[:4])
    predictions = pipeline.predict(X[:4])

    assert predictions.shape == y[:4].shape


def test_pipeline_works_with_preprocessing(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=4)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", model),
    ])

    pipeline.partial_fit(X[:2], y[:2])
    pipeline.partial_fit(X[2:4], y[2:4])

    predictions = pipeline.predict(X[:4])

    assert len(predictions) == 4


def test_pipeline_batch_fit_and_predict(sample_data):
    X, y = sample_data
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier()),
    ])

    pipeline.fit(X, y)

    assert pipeline.predict(X).shape == y.shape


def test_pipeline_rejects_empty_steps():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_rejects_duplicate_step_names():
    with pytest.raises(ValueError):
        Pipeline([
            ("step", StandardScaler()),
            ("step", DecisionTreeClassifier()),
        ])


def test_pipeline_rejects_inconsistent_input_features(sample_data):
    X, y = sample_data
    pipeline = Pipeline([("tree", DecisionTreeClassifier())])
    pipeline.fit(X, y)

    with pytest.raises(ValueError):
        pipeline.predict(np.array([[1.0, 2.0, 3.0]]))


# --------------------------------------------------
# StreamTrainer tests
# --------------------------------------------------

def test_stream_trainer_requires_model_or_pipeline():
    with pytest.raises(ValueError):
        StreamTrainer()


def test_stream_trainer_rejects_model_and_pipeline_together():
    model = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        StreamTrainer(model=model, pipeline=model)


def test_stream_trainer_trains_one_chunk(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier(buffer_size=3)
    trainer = StreamTrainer(model=model)

    trainer.fit_chunk(X[:3], y[:3])

    assert trainer.training_chunks == 1


def test_stream_trainer_scores_chunk(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier()
    model.fit(X, y)

    trainer = StreamTrainer(model=model)
    result = trainer.score_chunk(X, y)

    assert result["accuracy"] == pytest.approx(1.0)


def test_stream_trainer_tracks_cumulative_accuracy(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier()
    model.fit(X, y)

    trainer = StreamTrainer(model=model)
    trainer.score_chunk(X[:3], y[:3])
    trainer.score_chunk(X[3:], y[3:])

    assert trainer.get_cumulative_accuracy() == pytest.approx(1.0)


def test_stream_trainer_records_chunk_metrics(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier()
    model.fit(X, y)

    trainer = StreamTrainer(model=model)
    trainer.score_chunk(X, y)

    assert len(trainer.get_metrics()) == 1


def test_stream_trainer_estimates_memory(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier()
    model.fit(X, y)

    trainer = StreamTrainer(model=model)
    trainer.score_chunk(X, y)

    memory = trainer.memory_footprint()

    assert memory["tracked_memory_bytes"] > 0


def test_stream_trainer_reset_clears_metrics(sample_data):
    X, y = sample_data
    model = DecisionTreeClassifier()
    model.fit(X, y)

    trainer = StreamTrainer(model=model)
    trainer.score_chunk(X, y)
    trainer.reset_metrics()

    assert trainer.get_metrics() == []
    assert trainer.get_cumulative_accuracy() == 0.0

