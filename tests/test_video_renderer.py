import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from video.renderer import render_video
from video.schemas import VideoRenderRequest
import os
import uuid

@pytest.fixture
def valid_html():
    with open("tests/fixtures/demo_animation.html", "r") as f:
        return f.read()

@pytest.mark.asyncio
async def test_render_video_success(valid_html):
    # We will mock the external dependencies: FFmpeg encoder, Playwright, Storage
    with patch("video.renderer.VideoEncoder") as MockEncoder, \
         patch("video.renderer.async_playwright") as MockPlaywright, \
         patch("video.renderer.StorageAdapter") as MockStorage:

        # Setup mocks
        mock_encoder = MockEncoder.return_value
        mock_encoder.start = AsyncMock()
        mock_encoder.write_frame = AsyncMock()
        mock_encoder.finish = AsyncMock()

        mock_storage = MockStorage.return_value
        mock_storage.upload_video.return_value = "https://example.com/video.mp4"

        # We also need to mock file sizes for the output file before trying to return a file size
        with patch("os.path.getsize", return_value=1024):
            # Create a dummy output file so the cleanup/os checks pass
            with patch("os.makedirs"):
                request = VideoRenderRequest(
                    html=valid_html,
                    width=320,
                    height=180,
                    fps=5,
                    duration_seconds=1.0
                )
                response = await render_video(request)

        assert response.success is True
        assert response.video_url == "https://example.com/video.mp4"
        assert response.size_bytes == 1024

@pytest.mark.asyncio
async def test_render_video_invalid_html():
    request = VideoRenderRequest(
        html="<script src='http://evil.com'></script>",
        width=320,
        height=180,
        fps=5,
        duration_seconds=1.0
    )
    response = await render_video(request)
    assert response.success is False
    assert "External scripts are not allowed" in response.error
