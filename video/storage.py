import os
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class StorageAdapter:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required for storage")

        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.bucket_name = "generations" # Reusing the existing bucket from LayoutCraft

    def upload_video(self, file_path: str, job_id: str) -> str:
        """
        Uploads a video to Supabase storage and returns the public URL.
        """
        file_name = f"videos/{job_id}.mp4"

        with open(file_path, "rb") as f:
            self.supabase.storage.from_(self.bucket_name).upload(
                path=file_name,
                file=f,
                file_options={"content-type": "video/mp4"}
            )

        return self.supabase.storage.from_(self.bucket_name).get_public_url(file_name)
