import pytest
from mcp_server.server import mcp_server
import json
from unittest.mock import patch, MagicMock, AsyncMock
from mcp_server.server import tool_render_video

@pytest.mark.asyncio
async def test_tool_render_video_validation():
    # Test validation error
    result_str = await tool_render_video(
        html="<script src='test.js'></script>",
        duration_seconds=5.0,
        width=1920,
        height=1080,
        fps=30
    )
    result = json.loads(result_str)
    assert result["success"] is False
    assert "Validation error" in result["error"] or "External scripts" in result["error"]

@pytest.mark.asyncio
@patch("mcp_server.server.render_video")
async def test_tool_render_video_success(mock_render_video):
    from video.schemas import VideoRenderResponse

    mock_render_video.return_value = VideoRenderResponse(
        success=True,
        video_url="http://test.com/video.mp4",
        duration_seconds=5.0,
        width=1920,
        height=1080,
        fps=30,
        format="mp4",
        size_bytes=100
    )

    result_str = await tool_render_video(
        html="<html><body><h1>Test</h1></body></html>",
        duration_seconds=5.0,
        width=1920,
        height=1080,
        fps=30
    )
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["video_url"] == "http://test.com/video.mp4"
