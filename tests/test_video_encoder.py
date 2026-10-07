import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from video.encoder import VideoEncoder, EncoderError
import asyncio

@pytest.mark.asyncio
async def test_encoder_start_and_finish():
    encoder = VideoEncoder(width=1920, height=1080, fps=30, output_path="/tmp/test.mp4")

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdin.wait_closed = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        await encoder.start()

        mock_exec.assert_called_once()
        assert encoder._process is not None

        await encoder.write_frame(b"dummy_png_bytes")
        mock_process.stdin.write.assert_called_once_with(b"dummy_png_bytes")
        mock_process.stdin.drain.assert_called_once()

        await encoder.finish()
        mock_process.stdin.close.assert_called_once()
        mock_process.communicate.assert_called_once()

@pytest.mark.asyncio
async def test_encoder_finish_error():
    encoder = VideoEncoder(width=1920, height=1080, fps=30, output_path="/tmp/test.mp4")

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdin.wait_closed = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"ffmpeg error output"))
        mock_process.returncode = 1
        mock_exec.return_value = mock_process

        await encoder.start()

        with pytest.raises(EncoderError, match="FFmpeg failed with return code 1"):
            await encoder.finish()
