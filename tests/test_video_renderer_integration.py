import pytest
import asyncio
import os
import subprocess
from unittest.mock import patch, MagicMock
from video.renderer import render_video
from video.schemas import VideoRenderRequest
import json

@pytest.fixture
def test_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .box { width: 50px; height: 50px; background: red; animation: slide 1s linear forwards; }
            @keyframes slide { from { transform: translateX(0); } to { transform: translateX(100px); } }
        </style>
    </head>
    <body>
        <div class="box"></div>
    </body>
    </html>
    """

def get_video_info(filepath):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)

@pytest.mark.asyncio
async def test_video_renderer_real_pipeline(test_html):
    # Mock only the storage adapter to prevent actual upload
    with patch("video.renderer.StorageAdapter") as MockStorage:
        mock_storage_instance = MagicMock()
        MockStorage.return_value = mock_storage_instance
        # Simulate an upload, we need the local file to exist so we intercept upload_video to just return the local file path instead.
        def mock_upload(output_path, job_id):
            return output_path
        mock_storage_instance.upload_video.side_effect = mock_upload

        request = VideoRenderRequest(
            html=test_html,
            width=320,
            height=180,
            fps=5,
            duration_seconds=1.0
        )

        # In order to inspect the file, we need to stop render_video from deleting the directory at the end.
        with patch("shutil.rmtree") as mock_rmtree:
            response = await render_video(request)

        assert response.success is True
        output_path = response.video_url # Because we mocked it to return output_path

        assert os.path.exists(output_path)

        # Verify with ffprobe
        info = get_video_info(output_path)
        assert info is not None

        streams = info.get("streams", [])
        assert len(streams) > 0
        video_stream = next((s for s in streams if s["codec_type"] == "video"), None)
        assert video_stream is not None

        assert video_stream["width"] == 320
        assert video_stream["height"] == 180
        # ffmpeg might output 5/1 or similar for r_frame_rate
        assert eval(video_stream["r_frame_rate"]) == 5.0

        format_info = info.get("format", {})
        duration = float(format_info.get("duration", 0))
        # 1.0 second duration approx
        assert 0.8 <= duration <= 1.2

        # Clean up the manual intercept
        os.remove(output_path)
        os.rmdir(os.path.dirname(output_path))
