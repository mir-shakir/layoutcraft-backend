"""
Template service for LayoutCraft advanced template management
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from supabase import Client

logger = logging.getLogger(__name__)

class TemplateService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def create_template(self, user_id: str, template_data: Dict[str, Any]) -> Optional[str]:
        """Create a new custom template"""
        try:
            template_record = {
                "user_id": user_id,
                "name": template_data["name"],
                "description": template_data.get("description", ""),
                "prompt_template": template_data["prompt_template"],
                "default_width": template_data.get("default_width", 1200),
                "default_height": template_data.get("default_height", 630),
                "category": template_data.get("category", "general"),
                "tags": template_data.get("tags", []),
                "is_public": template_data.get("is_public", False),
                "created_at": datetime.now().isoformat()
            }
            
            response = self.supabase.table("custom_templates").insert(template_record).execute()
            
            if response.data:
                logger.info(f"Created template '{template_data['name']}' for user {user_id}")
                return response.data[0]["id"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating template: {str(e)}")
            return None
    
    async def get_user_templates(self, user_id: str, include_public: bool = True) -> List[Dict[str, Any]]:
        """Get user's templates and optionally public templates"""
        try:
            if include_public:
                # Get user's own templates plus public templates
                response = self.supabase.table("custom_templates").select("*").or_(
                    f"user_id.eq.{user_id},is_public.eq.true"
                ).order("created_at", desc=True).execute()
            else:
                # Get only user's templates
                response = self.supabase.table("custom_templates").select("*").eq(
                    "user_id", user_id
                ).order("created_at", desc=True).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting templates: {str(e)}")
            return []
    
    async def get_public_templates(self, category: str = None, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get public templates with optional category filter"""
        try:
            query = self.supabase.table("custom_templates").select(
                "*, user_profiles!inner(full_name)"
            ).eq("is_public", True)
            
            if category:
                query = query.eq("category", category)
            
            response = query.order("usage_count", desc=True).limit(limit).offset(offset).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting public templates: {str(e)}")
            return []
    
    async def get_featured_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured templates"""
        try:
            response = self.supabase.table("custom_templates").select(
                "*, user_profiles!inner(full_name)"
            ).eq("is_featured", True).eq("is_public", True).order(
                "usage_count", desc=True
            ).limit(limit).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting featured templates: {str(e)}")
            return []
    
    async def use_template(self, template_id: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Use a template and increment usage count"""
        try:
            # Get template
            template_response = self.supabase.table("custom_templates").select("*").eq("id", template_id).single().execute()
            
            if not template_response.data:
                return None
            
            template = template_response.data
            
            # Check access permissions
            if not template["is_public"] and template["user_id"] != user_id:
                return None
            
            # Increment usage count
            self.supabase.table("custom_templates").update({
                "usage_count": template["usage_count"] + 1
            }).eq("id", template_id).execute()
            
            logger.info(f"Template {template_id} used by user {user_id}")
            
            return template
            
        except Exception as e:
            logger.error(f"Error using template: {str(e)}")
            return None
    
    async def search_templates(self, query: str, category: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search public templates"""
        try:
            search_query = self.supabase.table("custom_templates").select(
                "*, user_profiles!inner(full_name)"
            ).eq("is_public", True)
            
            if category:
                search_query = search_query.eq("category", category)
            
            # Simple text search in name and description
            search_query = search_query.or_(
                f"name.ilike.%{query}%,description.ilike.%{query}%"
            )
            
            response = search_query.order("usage_count", desc=True).limit(limit).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error searching templates: {str(e)}")
            return []
    
    async def get_template_categories(self) -> List[Dict[str, Any]]:
        """Get all available template categories"""
        try:
            response = self.supabase.rpc("get_template_categories").execute()
            
            # If RPC doesn't exist, fall back to manual query
            if not response.data:
                response = self.supabase.table("custom_templates").select("category").eq("is_public", True).execute()
                
                categories = {}
                for item in response.data or []:
                    cat = item["category"]
                    categories[cat] = categories.get(cat, 0) + 1
                
                return [{"category": cat, "count": count} for cat, count in categories.items()]
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []
    
    async def update_template(self, template_id: str, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a template (owner only)"""
        try:
            # Verify ownership
            template_response = self.supabase.table("custom_templates").select("user_id").eq("id", template_id).single().execute()
            
            if not template_response.data or template_response.data["user_id"] != user_id:
                return False
            
            # Update template
            filtered_data = {k: v for k, v in update_data.items() if k in [
                "name", "description", "prompt_template", "default_width", 
                "default_height", "category", "tags", "is_public"
            ]}
            filtered_data["updated_at"] = datetime.now().isoformat()
            
            response = self.supabase.table("custom_templates").update(filtered_data).eq("id", template_id).execute()
            
            if response.data:
                logger.info(f"Updated template {template_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating template: {str(e)}")
            return False
    
    async def delete_template(self, template_id: str, user_id: str) -> bool:
        """Delete a template (owner only)"""
        try:
            response = self.supabase.table("custom_templates").delete().eq("id", template_id).eq("user_id", user_id).execute()
            
            if response.data:
                logger.info(f"Deleted template {template_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting template: {str(e)}")
            return False
