Role: Senior Full Stack Developer.
Task: Build a "Bulk Image Generation" feature named ContentBatch.

Backend Requirements (FastAPI/Python):

Create a generic HTML template system. Store 3 hardcoded HTML/CSS templates in a templates.py file (Variables: {{text}}, {{subtext}}, {{image_url}}).

Create an endpoint POST /api/batch/preview that takes {template_id, row_data} and returns 1 image (for preview).

Create an endpoint POST /api/batch/process that takes {template_id, all_rows_data}.

Optimization: Use Playwright to open the page once. Iterate through all_rows_data, inject the HTML for that row, take a screenshot, save to memory.

Output: Zip all screenshots and return the ZIP file directly.

Frontend Requirements (Vanilla JS/HTML):

Create a batch.html page.

State Management: Use a simple global object to store parsedCSV and columnMappings.

CSV Parser: Implement a client-side CSV parser (handle commas and quotes).

Mapping UI: When CSV is parsed, show dropdowns for each Template Variable (e.g., "Title", "Subtitle") to select which CSV header maps to it.

Preview Logic: When user changes mapping, auto-call /preview with the first row of data.