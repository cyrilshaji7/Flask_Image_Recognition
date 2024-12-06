from locust import HttpUser, between, task
import os

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # Random wait between 1-3 seconds

    @task(1)
    def load_main(self):
        """Test the main page loading"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def predict_image_file(self):
        """Test image prediction"""
        image_path = os.path.join('test_images', '0', 'Sign 0 (21).jpeg')
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': ('test_image.jpeg', f, 'image/jpeg')}
                with self.client.post("/prediction", files=files, catch_response=True) as response:
                    if response.status_code != 200:
                        response.failure(f"Got status code {response.status_code}")
                        
        except FileNotFoundError:
            print("Test image not found! Please check test_images folder.")
            self.environment.runner.quit()