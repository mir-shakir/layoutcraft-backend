from mcp.server.mcpserver import MCPServer
import mcp.types as types
from pydantic import ValidationError
import logging
import json

from video.renderer import render_video
from video.schemas import VideoRenderRequest

logger = logging.getLogger(__name__)

# Initialize MCPServer using the high-level API
mcp_server = MCPServer(
    name="layoutcraft-video-mcp",
    version="1.0.0",
    description="Renders self-contained HTML/CSS animations into a deterministic MP4 video."
)

@mcp_server.tool(
    name="render_video",
    description=(
        "Renders self-contained HTML/CSS animations into a deterministic MP4 video. "
        "Intended for visual explanations, presentations, animated diagrams, technical explainers. "
        "The HTML is supplied by the calling model. External network resources are blocked."
    )
)
async def tool_render_video(
    html: str,
    duration_seconds: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30
) -> str:
    """Execute the render_video tool."""
    try:
        req = VideoRenderRequest(
            html=html,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            fps=fps
        )
    except ValidationError as e:
        return json.dumps({"success": False, "error": f"Validation error: {str(e)}"})

    result = await render_video(req)
    return result.model_dump_json(exclude_none=True)
