from dotenv import load_dotenv
import os

load_dotenv()

print("Environment Variables:")
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
# print(f"SUPABASE_KEY: {os.getenv('SUPABASE_KEY')}")
print(f"SUPABASE_SERVICE_KEY: {os.getenv('SUPABASE_SERVICE_KEY')}")
print(f"JWT_SECRET: {os.getenv('JWT_SECRET')}")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
