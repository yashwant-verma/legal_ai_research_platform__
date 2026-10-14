from app.core.config import get_settings


def test_threshold_is_explicit_research_parameter():
    settings = get_settings()
    assert 0.0 < settings.dense_min_score < 1.0
