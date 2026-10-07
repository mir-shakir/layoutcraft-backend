import asyncio
import os
import uuid
import shutil
import logging
from playwright.async_api import async_playwright, Browser, Page
from video.schemas import VideoRenderRequest, VideoRenderResponse
from video.validator import validate_html
from video.encoder import VideoEncoder
from video.storage import StorageAdapter

logger = logging.getLogger(__name__)

# Concurrency limiting
RENDER_SEMAPHORE = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_RENDERS", "1")))

async def _setup_page(browser: Browser, width: int, height: int, html: str) -> Page:
    context = await browser.new_context(
        viewport={'width': width, 'height': height},
        offline=True, # Disable network access
    )
    page = await context.new_page()
    await page.set_content(html, wait_until="networkidle")
    return page

async def render_video(request: VideoRenderRequest) -> VideoRenderResponse:
    if RENDER_SEMAPHORE.locked():
        return VideoRenderResponse(
            success=False,
            error="Renderer is busy. Please try again later."
        )

    async with RENDER_SEMAPHORE:
        job_id = str(uuid.uuid4())
        job_dir = f"/tmp/layoutcraft-video/{job_id}"
        os.makedirs(job_dir, exist_ok=True)

        output_path = os.path.join(job_dir, "output.mp4")

        try:
            # 1. Validate HTML
            validated_html = validate_html(request.html)

            # 2. Setup FFmpeg Encoder
            encoder = VideoEncoder(
                width=request.width,
                height=request.height,
                fps=request.fps,
                output_path=output_path
            )
            await encoder.start()

            # 3. Render Frames
            num_frames = int(request.duration_seconds * request.fps)
            frame_duration_ms = 1000.0 / request.fps

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await _setup_page(browser, request.width, request.height, validated_html)

                await page.evaluate("""
                    () => {
                        window._videoCurrentTime = 0;
                        const originalRequestAnimationFrame = window.requestAnimationFrame;
                        window.requestAnimationFrame = (callback) => {
                            return setTimeout(() => callback(window._videoCurrentTime), 0);
                        };
                        window.performance.now = () => window._videoCurrentTime;
                        window.Date.now = () => window._videoCurrentTime;

                        // Pause CSS animations and transitions
                        const style = document.createElement('style');
                        style.textContent = `
                            *, *::before, *::after {
                                animation-play-state: paused !important;
                            }
                        `;
                        document.head.appendChild(style);
                    }
                """)

                for frame_idx in range(num_frames):
                    current_time_ms = frame_idx * frame_duration_ms

                    # Advance time in browser
                    await page.evaluate(f"""
                        () => {{
                            window._videoCurrentTime = {current_time_ms};
                            // Seek CSS animations
                            document.getAnimations().forEach(anim => {{
                                anim.currentTime = {current_time_ms};
                            }});
                        }}
                    """)

                    # Capture frame as PNG
                    screenshot_bytes = await page.screenshot(
                        type="png",
                        omit_background=False
                    )

                    # Write frame to FFmpeg
                    await encoder.write_frame(screenshot_bytes)

                await browser.close()

            await encoder.finish()

            # 4. Upload to Storage
            storage = StorageAdapter()
            video_url = storage.upload_video(output_path, job_id)

            file_size = os.path.getsize(output_path)

            return VideoRenderResponse(
                success=True,
                video_url=video_url,
                duration_seconds=request.duration_seconds,
                width=request.width,
                height=request.height,
                fps=request.fps,
                format="mp4",
                size_bytes=file_size
            )

        except Exception as e:
            logger.error(f"Render failed: {str(e)}", exc_info=True)
            return VideoRenderResponse(
                success=False,
                error=str(e)
            )
        finally:
            # Clean up
            shutil.rmtree(job_dir, ignore_errors=True)
