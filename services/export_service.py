"""
Export service for LayoutCraft with premium format support
"""
import io
import base64
from PIL import Image
import cairosvg
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExportService:
    def __init__(self):
        self.supported_formats = {
            "png": self._export_png,
            "jpg": self._export_jpg,
            "jpeg": self._export_jpg,
            "webp": self._export_webp,
            "svg": self._export_svg
        }
    
    async def convert_image(self, image_bytes: bytes, format: str, quality: int = 95) -> bytes:
        """Convert image to specified format"""
        try:
            format = format.lower()
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            return await self.supported_formats[format](image_bytes, quality)
            
        except Exception as e:
            logger.error(f"Error converting image to {format}: {str(e)}")
            raise
    
    async def _export_png(self, image_bytes: bytes, quality: int = 95) -> bytes:
        """Export as PNG (lossless)"""
        # PNG is already the default format, return as-is
        return image_bytes
    
    async def _export_jpg(self, image_bytes: bytes, quality: int = 95) -> bytes:
        """Export as JPEG"""
        try:
            # Convert PNG to JPEG
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Convert RGBA to RGB if necessary
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                return output.getvalue()
                
        except Exception as e:
            logger.error(f"Error exporting to JPEG: {str(e)}")
            raise
    
    async def _export_webp(self, image_bytes: bytes, quality: int = 95) -> bytes:
        """Export as WebP"""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                output = io.BytesIO()
                img.save(output, format='WebP', quality=quality, optimize=True)
                return output.getvalue()
                
        except Exception as e:
            logger.error(f"Error exporting to WebP: {str(e)}")
            raise
    
    async def _export_svg(self, image_bytes: bytes, quality: int = 95) -> bytes:
        """Export as SVG (enterprise only)"""
        try:
            # For MVP, we'll embed the PNG as base64 in SVG
            # In production, you might want to use actual SVG generation
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
            
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <image href="data:image/png;base64,{img_base64}" width="{width}" height="{height}"/>
</svg>'''
            
            return svg_content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error exporting to SVG: {str(e)}")
            raise
    
    def get_content_type(self, format: str) -> str:
        """Get content type for format"""
        content_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "svg": "image/svg+xml"
        }
        return content_types.get(format.lower(), "application/octet-stream")
    
    def get_file_extension(self, format: str) -> str:
        """Get file extension for format"""
        extensions = {
            "png": ".png",
            "jpg": ".jpg",
            "jpeg": ".jpg",
            "webp": ".webp",
            "svg": ".svg"
        }
        return extensions.get(format.lower(), ".png")
