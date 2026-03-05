"""
ContentBatch service for database operations on cb_templates and cb_batches.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

from supabase import Client

logger = logging.getLogger(__name__)


class ContentBatchService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client

    # ── Templates ──────────────────────────────────────────────

    async def list_templates(self) -> List[Dict[str, Any]]:
        """Return all active templates."""
        try:
            response = (
                self.supabase.table("cb_templates")
                .select("id, name, html_structure, css_styles, ai_rules, thumbnail_url, created_at")
                .eq("is_active", True)
                .order("created_at", desc=False)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []

    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Return a single template by ID."""
        try:
            response = (
                self.supabase.table("cb_templates")
                .select("*")
                .eq("id", template_id)
                .eq("is_active", True)
                .single()
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error getting template {template_id}: {e}")
            return None

    # ── Batches ────────────────────────────────────────────────

    async def create_batch(
        self,
        user_id: str,
        template_id: str,
        input_data: list,
        column_mapping: dict,
    ) -> Optional[Dict[str, Any]]:
        """Create a new batch record with status 'processing'."""
        try:
            data = {
                "user_id": str(user_id),
                "template_id": str(template_id),
                "input_data": input_data,
                "column_mapping": column_mapping,
                "status": "processing",
            }
            response = self.supabase.table("cb_batches").insert(data).execute()
            if response.data:
                logger.info(f"Created batch {response.data[0]['id']} for user {user_id}")
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            return None

    async def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Return a batch by ID."""
        try:
            response = (
                self.supabase.table("cb_batches")
                .select("*")
                .eq("id", batch_id)
                .single()
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error getting batch {batch_id}: {e}")
            return None

    async def update_batch_completed(
        self, batch_id: str, result_zip_url: str
    ) -> bool:
        """Mark a batch as completed with the ZIP URL."""
        try:
            self.supabase.table("cb_batches").update(
                {"status": "completed", "result_zip_url": result_zip_url}
            ).eq("id", batch_id).execute()
            logger.info(f"Batch {batch_id} completed")
            return True
        except Exception as e:
            logger.error(f"Error completing batch {batch_id}: {e}")
            return False

    async def update_batch_failed(self, batch_id: str, error_log: str) -> bool:
        """Mark a batch as failed with an error log."""
        try:
            self.supabase.table("cb_batches").update(
                {"status": "failed", "error_log": error_log}
            ).eq("id", batch_id).execute()
            logger.info(f"Batch {batch_id} failed: {error_log}")
            return True
        except Exception as e:
            logger.error(f"Error failing batch {batch_id}: {e}")
            return False

    async def list_user_batches(self, user_id: str) -> List[Dict[str, Any]]:
        """Return all batches for a user, newest first, with template name."""
        try:
            response = (
                self.supabase.table("cb_batches")
                .select("id, template_id, status, result_zip_url, created_at, input_data, cb_templates(name)")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            batches = response.data or []
            for b in batches:
                tpl = b.pop("cb_templates", None)
                b["template_name"] = tpl["name"] if tpl and isinstance(tpl, dict) else ""
                b["row_count"] = len(b.get("input_data") or [])
                b.pop("input_data", None)
            return batches
        except Exception as e:
            logger.error(f"Error listing batches for user {user_id}: {e}")
            return []
