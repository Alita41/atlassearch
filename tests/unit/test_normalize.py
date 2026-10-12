from atlassearch.preprocessing.normalize import normalize_text, detect_script


def test_normalize_basic():
    assert normalize_text("Hello, WORLD!!!") == "hello world"


def test_urdu_variants():
    assert "ک" in normalize_text("كتاب")


def test_script_detection():
    assert detect_script("hello دنیا") == "mixed"
