
import requests
import sys

url = "https://mr-system-4zz9.onrender.com"

try:
    print(f"Checking {url}...")
    r = requests.get(url)
    print(f"Status: {r.status_code}")
    print(f"Content: {r.text[:200]}")
    
    print(f"\nChecking {url}/docs...")
    r = requests.get(f"{url}/docs")
    print(f"Status: {r.status_code}")

    print(f"\nChecking {url}/auth/login...")
    # 405 Method Not Allowed is expected for GET on login, 404 means route doesn't exist
    r = requests.get(f"{url}/auth/login") 
    print(f"Status: {r.status_code}")
    
except Exception as e:
    print(f"Error: {e}")
