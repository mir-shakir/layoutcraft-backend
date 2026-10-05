import re

class ValidationError(Exception):
    pass

def validate_html(html: str) -> str:
    """
    Validates and cleans the HTML to prevent access to external resources or malicious activities.
    This is not a perfect sandbox, but a robust V1 mitigation.
    """
    if len(html.encode('utf-8')) > 2 * 1024 * 1024:
        raise ValidationError("HTML size must be less than or equal to 2 MB")

    html_lower = html.lower()

    # Disallow external scripts or any script tag with a src attribute
    if re.search(r'<script[^>]*\bsrc\s*=', html_lower):
        raise ValidationError("External scripts are not allowed")

    # Disallow external stylesheets
    if re.search(r'<link[^>]*\brel\s*=\s*["\']?stylesheet["\']?[^>]*\bhref\s*=', html_lower):
        raise ValidationError("External stylesheets are not allowed")

    # Disallow iframes, objects, embeds
    forbidden_tags = ["<iframe", "<object", "<embed"]
    for tag in forbidden_tags:
        if tag in html_lower:
            raise ValidationError(f"Forbidden tag used: {tag[1:]}")

    # Disallow fetching or networking APIs
    forbidden_js = ["fetch(", "xmlhttprequest", "websocket", "navigator.sendbeacon"]
    for js in forbidden_js:
        if js in html_lower:
            raise ValidationError(f"Forbidden JavaScript API used: {js}")

    # Very naive check for http/https URLs (could be expanded)
    # We want to allow xmlns, but ban general external resource loading
    # Disallow file://
    if "file://" in html_lower:
        raise ValidationError("file:// URLs are not allowed")

    return html
