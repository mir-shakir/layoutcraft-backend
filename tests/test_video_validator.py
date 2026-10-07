import pytest
from video.validator import validate_html, ValidationError

def test_valid_html():
    html = "<html><body><h1>Hello World</h1></body></html>"
    assert validate_html(html) == html

def test_rejects_external_scripts():
    html = "<html><head><script src='https://example.com/script.js'></script></head><body></body></html>"
    with pytest.raises(ValidationError, match="External scripts are not allowed"):
        validate_html(html)

def test_rejects_external_stylesheets():
    html = "<html><head><link rel='stylesheet' href='https://example.com/style.css'></head><body></body></html>"
    with pytest.raises(ValidationError, match="External stylesheets are not allowed"):
        validate_html(html)

def test_rejects_iframes():
    html = "<html><body><iframe src='https://example.com'></iframe></body></html>"
    with pytest.raises(ValidationError, match="Forbidden tag used: iframe"):
        validate_html(html)

def test_rejects_fetch():
    html = "<html><body><script>fetch('https://example.com')</script></body></html>"
    with pytest.raises(ValidationError, match=r"Forbidden JavaScript API used: fetch\("):
        validate_html(html)

def test_rejects_file_urls():
    html = "<html><body><img src='file:///etc/passwd'></body></html>"
    with pytest.raises(ValidationError, match="file:// URLs are not allowed"):
        validate_html(html)

def test_rejects_large_html():
    html = "a" * (2 * 1024 * 1024 + 1)
    with pytest.raises(ValidationError, match="HTML size must be less than or equal to 2 MB"):
        validate_html(html)
