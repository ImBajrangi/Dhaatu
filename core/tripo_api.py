"""
Tripo3D Cloud API Client for Professional 3D Generation
This module handles the communication with Tripo3D's cloud API to generate
production-quality 3D models from images.
"""
import os
import time
import requests
import tempfile

# API Configuration
TRIPO_API_BASE = "https://api.tripo3d.ai/v2/openapi"
TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY", "")

class TripoAPIClient:
    def __init__(self):
        self.api_key = TRIPO_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def is_configured(self):
        return bool(self.api_key)
    
    def upload_image(self, image_path):
        """Upload image to Tripo3D and get a file token."""
        url = f"{TRIPO_API_BASE}/upload"
        with open(image_path, "rb") as f:
            files = {"file": f}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(url, headers=headers, files=files)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", {}).get("image_token")
        return None
    
    def create_task(self, image_token):
        """Create a 3D generation task."""
        url = f"{TRIPO_API_BASE}/task"
        payload = {
            "type": "image_to_model",
            "file": {"type": "jpg", "file_token": image_token}
        }
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", {}).get("task_id")
        return None
    
    def poll_task(self, task_id, timeout=300):
        """Poll task status until complete or timeout."""
        url = f"{TRIPO_API_BASE}/task/{task_id}"
        start = time.time()
        
        while time.time() - start < timeout:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json().get("data", {})
                status = data.get("status")
                
                if status == "success":
                    return data.get("output", {}).get("model")
                elif status in ("failed", "cancelled"):
                    return None
            
            time.sleep(3)
        
        return None
    
    def download_model(self, model_url, suffix=".glb"):
        """Download the generated model to a temp file."""
        response = requests.get(model_url)
        if response.status_code == 200:
            out_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
            with open(out_path, "wb") as f:
                f.write(response.content)
            return out_path
        return None
    
    def generate(self, image_path):
        """Full pipeline: upload -> create task -> poll -> download."""
        if not self.is_configured():
            return None, "TRIPO_API_KEY not set in environment"
        
        print("Uploading image to Tripo3D...")
        image_token = self.upload_image(image_path)
        if not image_token:
            return None, "Failed to upload image"
        
        print("Creating 3D generation task...")
        task_id = self.create_task(image_token)
        if not task_id:
            return None, "Failed to create task"
        
        print(f"Waiting for task {task_id} to complete...")
        model_url = self.poll_task(task_id)
        if not model_url:
            return None, "Task failed or timed out"
        
        print("Downloading generated 3D model...")
        model_path = self.download_model(model_url)
        if not model_path:
            return None, "Failed to download model"
        
        return model_path, None

# Singleton instance
tripo_client = TripoAPIClient()
