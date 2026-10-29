import pytest
from my_classifier import TextClassifier

@pytest.fixture(scope="module")

def model_pipeline():
    return TextClassifier(model_path="distilbert-base-uncased")

def test_clean_text_handles_edge_cases(model_pipeline):
    assert model_pipeline.clean_text("  giush    ") == "giush"
    assert model_pipeline.clean_text(None) == ""
    assert model_pipeline.clean_text(['testing', 'list','instead','of','string']) == ""

def test_predict_returns_valid_probabilities(model_pipeline):
    result = model_pipeline.predict("Essastyczny test")

    assert "label" in result
    assert "confidence" in result

    assert result["label"] in ["neg", "pos"]

    assert 0.0 <= result["confidence"] <= 1.0

    probs_sum = sum(result["all_probabilities"])
    assert probs_sum == pytest.approx(1.0, rel=1e-3)

def test_predict_empty_text_raises_error(model_pipeline):
    with pytest.raises(ValueError):
        model_pipeline.predict("     ")
