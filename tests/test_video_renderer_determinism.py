import pytest
import asyncio
import os
import subprocess
import hashlib
from unittest.mock import patch, MagicMock
from video.renderer import render_video
from video.schemas import VideoRenderRequest

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

@pytest.fixture
def mock_storage_for_hash():
    # Helper to mock storage but return local path
    def _mock_upload(output_path, job_id):
        return output_path
    return _mock_upload

@pytest.mark.asyncio
async def test_video_renderer_deterministic_static():
    html = "<html><body><h1>Static</h1></body></html>"
    req = VideoRenderRequest(html=html, width=320, height=180, fps=5, duration_seconds=1.0)

    with patch("video.renderer.StorageAdapter") as MockStorage, patch("shutil.rmtree") as mock_rmtree:
        MockStorage.return_value.upload_video.side_effect = lambda op, jid: op
        res1 = await render_video(req)
        res2 = await render_video(req)

        assert res1.success is True and res2.success is True

        # Test hashes match
        hash1 = get_file_hash(res1.video_url)
        hash2 = get_file_hash(res2.video_url)
        assert hash1 == hash2

        os.remove(res1.video_url)
        os.rmdir(os.path.dirname(res1.video_url))
        os.remove(res2.video_url)
        os.rmdir(os.path.dirname(res2.video_url))

@pytest.mark.asyncio
async def test_video_renderer_deterministic_css():
    html = """
    <html><head><style>
        .box { width: 50px; height: 50px; background: red; animation: slide 1s linear forwards; }
        @keyframes slide { from { transform: translateX(0); } to { transform: translateX(100px); } }
    </style></head><body><div class="box"></div></body></html>
    """
    req = VideoRenderRequest(html=html, width=320, height=180, fps=5, duration_seconds=1.0)

    with patch("video.renderer.StorageAdapter") as MockStorage, patch("shutil.rmtree") as mock_rmtree:
        MockStorage.return_value.upload_video.side_effect = lambda op, jid: op
        res1 = await render_video(req)
        res2 = await render_video(req)

        assert res1.success is True and res2.success is True
        assert get_file_hash(res1.video_url) == get_file_hash(res2.video_url)

        os.remove(res1.video_url)
        os.rmdir(os.path.dirname(res1.video_url))
        os.remove(res2.video_url)
        os.rmdir(os.path.dirname(res2.video_url))

@pytest.mark.asyncio
async def test_video_renderer_deterministic_raf():
    html = """
    <html><body><div id="box" style="width:50px; height:50px; background:blue;"></div>
    <script>
        let box = document.getElementById('box');
        function tick(time) {
            box.style.transform = `translateX(${time / 10}px)`;
            requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    </script>
    </body></html>
    """
    req = VideoRenderRequest(html=html, width=320, height=180, fps=5, duration_seconds=1.0)

    with patch("video.renderer.StorageAdapter") as MockStorage, patch("shutil.rmtree") as mock_rmtree:
        MockStorage.return_value.upload_video.side_effect = lambda op, jid: op
        res1 = await render_video(req)
        res2 = await render_video(req)

        assert res1.success is True and res2.success is True
        assert get_file_hash(res1.video_url) == get_file_hash(res2.video_url)

        os.remove(res1.video_url)
        os.rmdir(os.path.dirname(res1.video_url))
        os.remove(res2.video_url)
        os.rmdir(os.path.dirname(res2.video_url))

@pytest.mark.asyncio
async def test_video_renderer_deterministic_multiple_css():
    html = """
    <html><head><style>
        .b1 { width: 50px; height: 50px; background: red; animation: a1 1s linear forwards; }
        .b2 { width: 50px; height: 50px; background: blue; animation: a2 1s linear forwards; animation-delay: 0.5s; }
        @keyframes a1 { to { transform: translateX(100px); } }
        @keyframes a2 { to { transform: translateY(100px); } }
    </style></head><body><div class="b1"></div><div class="b2"></div></body></html>
    """
    req = VideoRenderRequest(html=html, width=320, height=180, fps=5, duration_seconds=1.0)

    with patch("video.renderer.StorageAdapter") as MockStorage, patch("shutil.rmtree") as mock_rmtree:
        MockStorage.return_value.upload_video.side_effect = lambda op, jid: op
        res1 = await render_video(req)
        res2 = await render_video(req)

        assert res1.success is True and res2.success is True
        assert get_file_hash(res1.video_url) == get_file_hash(res2.video_url)

        os.remove(res1.video_url)
        os.rmdir(os.path.dirname(res1.video_url))
        os.remove(res2.video_url)
        os.rmdir(os.path.dirname(res2.video_url))
