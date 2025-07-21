"""
Analytics service for LayoutCraft insights and reporting
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from supabase import Client
from collections import defaultdict

logger = logging.getLogger(__name__)

class AnalyticsService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get generation history for the period
            generations_response = self.supabase.table("generation_history").select("*").eq(
                "user_id", user_id
            ).gte("created_at", start_date.isoformat()).execute()
            
            generations = generations_response.data or []
            
            if not generations:
                return self._empty_analytics()
            
            # Calculate metrics
            total_generations = len(generations)
            
            # Generation times
            times = [g["generation_time_ms"] for g in generations if g["generation_time_ms"]]
            avg_generation_time = sum(times) / len(times) if times else 0
            
            # Model usage
            model_usage = defaultdict(int)
            for g in generations:
                model_usage[g["model_used"]] += 1
            
            # Dimension usage
            dimension_usage = defaultdict(int)
            for g in generations:
                dim = f"{g['width']}x{g['height']}"
                dimension_usage[dim] += 1
            
            # Daily activity
            daily_activity = defaultdict(int)
            for g in generations:
                date = datetime.fromisoformat(g["created_at"].replace("Z", "+00:00")).date()
                daily_activity[date.isoformat()] += 1
            
            # Peak hours
            hourly_activity = defaultdict(int)
            for g in generations:
                hour = datetime.fromisoformat(g["created_at"].replace("Z", "+00:00")).hour
                hourly_activity[hour] += 1
            
            peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else 0
            
            # Success rate (assume all stored generations are successful)
            success_rate = 100.0
            
            return {
                "period_days": days,
                "total_generations": total_generations,
                "avg_generation_time_ms": round(avg_generation_time),
                "success_rate": success_rate,
                "most_used_model": max(model_usage.items(), key=lambda x: x[1])[0],
                "most_used_dimensions": max(dimension_usage.items(), key=lambda x: x[1])[0],
                "peak_hour": peak_hour,
                "daily_activity": dict(daily_activity),
                "model_distribution": dict(model_usage),
                "dimension_distribution": dict(dimension_usage),
                "hourly_distribution": dict(hourly_activity)
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {str(e)}")
            return self._empty_analytics()
    
    async def get_popular_prompts(self, limit: int = 10, days: int = 7) -> List[Dict[str, Any]]:
        """Get most popular prompts across all users"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            response = self.supabase.table("generation_history").select(
                "prompt"
            ).gte("created_at", start_date.isoformat()).execute()
            
            if not response.data:
                return []
            
            # Count prompt frequency
            from collections import Counter
            prompt_counts = Counter(item["prompt"] for item in response.data)
            
            # Return top prompts (anonymized)
            popular_prompts = []
            for prompt, count in prompt_counts.most_common(limit):
                # Truncate long prompts
                display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
                popular_prompts.append({
                    "prompt": display_prompt,
                    "usage_count": count,
                    "popularity_score": round((count / len(response.data)) * 100, 2)
                })
            
            return popular_prompts
            
        except Exception as e:
            logger.error(f"Error getting popular prompts: {str(e)}")
            return []
    
    async def get_platform_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get platform-wide analytics (admin only)"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Total users
            users_response = self.supabase.table("user_profiles").select("id, subscription_tier, created_at").execute()
            total_users = len(users_response.data) if users_response.data else 0
            
            # Subscription distribution
            tier_distribution = defaultdict(int)
            new_users = 0
            
            for user in users_response.data or []:
                tier_distribution[user["subscription_tier"]] += 1
                user_created = datetime.fromisoformat(user["created_at"].replace("Z", "+00:00"))
                if user_created >= start_date:
                    new_users += 1
            
            # Generations in period
            generations_response = self.supabase.table("generation_history").select(
                "id, model_used, generation_time_ms, created_at"
            ).gte("created_at", start_date.isoformat()).execute()
            
            total_generations = len(generations_response.data) if generations_response.data else 0
            
            # Average generation time
            times = [g["generation_time_ms"] for g in generations_response.data or [] if g["generation_time_ms"]]
            avg_generation_time = sum(times) / len(times) if times else 0
            
            # Daily generation counts
            daily_generations = defaultdict(int)
            for g in generations_response.data or []:
                date = datetime.fromisoformat(g["created_at"].replace("Z", "+00:00")).date()
                daily_generations[date.isoformat()] += 1
            
            return {
                "period_days": days,
                "total_users": total_users,
                "new_users": new_users,
                "total_generations": total_generations,
                "avg_generation_time_ms": round(avg_generation_time),
                "tier_distribution": dict(tier_distribution),
                "daily_generations": dict(daily_generations),
                "generations_per_user": round(total_generations / max(total_users, 1), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting platform analytics: {str(e)}")
            return {}
    
    async def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            response = self.supabase.table("generation_history").select(
                "generation_time_ms, model_used, created_at"
            ).gte("created_at", start_date.isoformat()).execute()
            
            if not response.data:
                return {}
            
            generations = response.data
            
            # Overall performance
            times = [g["generation_time_ms"] for g in generations if g["generation_time_ms"]]
            
            if not times:
                return {}
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Performance by model
            model_performance = defaultdict(list)
            for g in generations:
                if g["generation_time_ms"]:
                    model_performance[g["model_used"]].append(g["generation_time_ms"])
            
            model_avg_times = {}
            for model, times in model_performance.items():
                model_avg_times[model] = sum(times) / len(times)
            
            # Performance trends (daily averages)
            daily_performance = defaultdict(list)
            for g in generations:
                if g["generation_time_ms"]:
                    date = datetime.fromisoformat(g["created_at"].replace("Z", "+00:00")).date()
                    daily_performance[date.isoformat()].append(g["generation_time_ms"])
            
            daily_avg_times = {}
            for date, times in daily_performance.items():
                daily_avg_times[date] = sum(times) / len(times)
            
            return {
                "period_days": days,
                "total_samples": len(times),
                "avg_generation_time_ms": round(avg_time),
                "min_generation_time_ms": min_time,
                "max_generation_time_ms": max_time,
                "model_performance": {k: round(v) for k, v in model_avg_times.items()},
                "daily_performance": {k: round(v) for k, v in daily_avg_times.items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure"""
        return {
            "period_days": 30,
            "total_generations": 0,
            "avg_generation_time_ms": 0,
            "success_rate": 0,
            "most_used_model": None,
            "most_used_dimensions": None,
            "peak_hour": 0,
            "daily_activity": {},
            "model_distribution": {},
            "dimension_distribution": {},
            "hourly_distribution": {}
        }
