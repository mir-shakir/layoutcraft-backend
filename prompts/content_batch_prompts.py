"""
Prompt templates for ContentBatch bulk image generation.
"""

CONTENT_BATCH_MASTER_PROMPT = """You are an expert HTML/CSS developer. Your job is to generate a SINGLE, self-contained HTML page that contains multiple design variations based on a template and data rows.

## INSTRUCTIONS

1. You are given a **Base HTML Template** that represents a single card/design. It uses placeholder tokens like {{quote}}, {{author}}, etc.
2. You are given **CSS Styles** that define the visual styling for the template elements.
3. You are given a **Data Array** (JSON) where each object represents one row of data.
4. You are given a **Column Mapping** that maps template placeholders to data field names.
5. You are given optional **AI Rules** for handling edge cases (e.g. long text).

## YOUR TASK

Generate a valid, complete HTML page with these requirements:

- The page must contain exactly {row_count} containers, one for each data row.
- Each container MUST have the id `variation-0`, `variation-1`, ..., `variation-{last_index}`.
- Each container should be a standalone rendered card based on the template, with the data row's values substituted into the placeholders.
- Apply the AI Rules to each card individually (e.g. if a quote is too long, reduce font size for THAT specific card).
- Each container must be visually identical in layout to the base template but with different content.
- Containers should be stacked vertically with NO gap, NO margin, NO padding between them on the page body.
- The page background should be transparent or white.
- Include ALL fonts and styles inline or in a <style> block. Do NOT use external stylesheets except Google Fonts CDN.
- You MUST include the provided CSS Styles in a <style> block in the <head> of the page. These styles define the visual look of each card.
- The HTML must be completely self-contained and render correctly in a headless browser.

## OUTPUT FORMAT

Output ONLY the raw HTML. No markdown fences, no explanation, no comments outside the HTML.
Start your response with `<!DOCTYPE html>` and end with `</html>`.

## BASE TEMPLATE
```html
{template_html}
```

## CSS STYLES
```css
{css_styles}
```

## AI RULES
{ai_rules}

## COLUMN MAPPING
{column_mapping}

## DATA ROWS
```json
{data_json}
```

Generate the complete HTML page now.
"""


def build_content_batch_prompt(
    template_html: str,
    css_styles: str,
    ai_rules: str,
    column_mapping: dict,
    data_rows: list[dict],
) -> str:
    """
    Build the full prompt for the LLM to generate a master HTML file
    containing all variations.
    """
    import json

    row_count = len(data_rows)
    last_index = row_count - 1

    # Map the data rows using the column mapping so the LLM sees the
    # template placeholder names directly.
    mapped_rows = []
    for row in data_rows:
        mapped = {}
        for placeholder, csv_col in column_mapping.items():
            mapped[placeholder] = row.get(csv_col, "")
        mapped_rows.append(mapped)

    return CONTENT_BATCH_MASTER_PROMPT.format(
        row_count=row_count,
        last_index=last_index,
        template_html=template_html,
        css_styles=css_styles or "",
        ai_rules=ai_rules or "No special rules. Render as-is.",
        column_mapping=json.dumps(column_mapping, indent=2),
        data_json=json.dumps(mapped_rows, indent=2),
    )


# ── Text-to-CSV prompt ───────────────────────────────────────

TEXT_TO_CSV_PROMPT = """You are a data extraction assistant. Your job is to convert unstructured or semi-structured text into a well-formed CSV.

## TARGET COLUMNS
The CSV MUST have exactly these columns as the header row:
{columns}

## USER'S TEXT
{raw_text}

## USER'S INTENT
{intent}

## RULES

1. Output ONLY the raw CSV. No markdown fences, no explanation, no extra text.
2. The first line must be the header row with the exact column names listed above, comma-separated.
3. Each subsequent line is one data row extracted from the user's text.
4. If the text contains multiple items/entries, each becomes its own row.
5. If a field value contains commas, newlines, or double quotes, wrap the entire field in double quotes and escape inner double quotes by doubling them (standard CSV quoting per RFC 4180).
6. If a column value cannot be determined from the text, leave the field empty (two consecutive commas).
7. Preserve the original text content as faithfully as possible -- do not paraphrase or summarize unless the user's intent says otherwise.
8. Do NOT invent data that is not present in the source text.
"""


def build_text_to_csv_prompt(
    raw_text: str,
    columns: list[str],
    intent: str = "",
) -> str:
    """
    Build a prompt that asks the LLM to convert unstructured text
    into a CSV with the given column headers.
    """
    return TEXT_TO_CSV_PROMPT.format(
        columns=", ".join(columns),
        raw_text=raw_text,
        intent=intent or "Extract all items from the text into the target columns.",
    )
