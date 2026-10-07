# LayoutCraft Video MCP Server

This isolated subsystem converts self-contained HTML/CSS animations into H.264 MP4 videos using Chromium/Playwright and FFmpeg, exposing the functionality as a Model Context Protocol (MCP) server.

## Features
- **Deterministic Rendering**: Takes control of CSS Animations and `requestAnimationFrame` ensuring deterministic frame capture decoupled from wall-clock rendering.
- **MCP Exposed**: Securely exposes `render_video` over the standard `mcp` Streamable HTTP transport mechanism.
- **Isolated Validation**: Disables network execution (`<script src="...">`, `<base>`, external `<link>` CSS, `<iframe/embed/object>`, `<audio/video>`, etc) alongside browser-level isolation logic `offline=True`.

## Usage
The API server automatically handles `Streamable HTTP` under the endpoint `/mcp`. Tools discoverable via standard MCP protocol handshake on that endpoint.

### Available Tools:
`render_video`:
- `html` (str): Full HTML source of the document. Max 2MB. Only use data URIs for images, avoid external resources.
- `duration_seconds` (float): Output video duration. Max 30.0.
- `width` (int): default 1920
- `height` (int): default 1080
- `fps` (int): default 30

### Environment variables
- `VIDEO_RENDER_TIMEOUT_SECONDS`: Global timeout for rendering jobs (default 120s).
- `MAX_CONCURRENT_RENDERS`: Restricts system CPU consumption (default 1).
- `SUPABASE_URL` & `SUPABASE_SERVICE_KEY`: Utilized directly from `layoutcraft` backend constants.

## Local Development
1. Install Python dependencies: `pip install -r requirements.txt`.
2. Install Playwright: `playwright install-deps && playwright install`.
3. Ensure FFmpeg is available on your host: `apt-get install -y ffmpeg` (included in Dockerfile).
4. Start FastAPI server normally: `uvicorn index:app --port 8000`.
5. Access MCP streams endpoint at `http://localhost:8000/mcp`.
6. Run the test suite: `pytest`

### Docker Note
Ensure you build via the provided `Dockerfile`. FFmpeg has been added to it alongside Playwright configuration dependencies.
