import asyncio
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class EncoderError(Exception):
    pass

class VideoEncoder:
    def __init__(self, width: int, height: int, fps: int, output_path: str):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self._process: asyncio.subprocess.Process | None = None

    async def start(self):
        """Starts the FFmpeg process expecting PNG frames from stdin via image2pipe."""
        command = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "image2pipe",
            "-vcodec", "png",
            "-r", str(self.fps),
            "-i", "-",  # Read from stdin
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p", # Broad compatibility
            "-s", f"{self.width}x{self.height}",
            self.output_path
        ]

        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        logger.info(f"Started FFmpeg process {self._process.pid} for {self.output_path}")

    async def write_frame(self, frame_bytes: bytes):
        if not self._process or not self._process.stdin:
            raise EncoderError("Encoder not started or stdin missing")
        self._process.stdin.write(frame_bytes)
        await self._process.stdin.drain()

    async def finish(self):
        if not self._process:
            return

        if self._process.stdin:
            self._process.stdin.close()
            await self._process.stdin.wait_closed()

        stdout, stderr = await self._process.communicate()

        if self._process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
            logger.error(f"FFmpeg failed: {error_msg}")
            raise EncoderError(f"FFmpeg failed with return code {self._process.returncode}")

        logger.info(f"Finished encoding {self.output_path}")
