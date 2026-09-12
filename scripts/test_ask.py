import httpx
import json

url = "http://localhost:8000/ask"
payload = {"question": "What are the important articles for Indian people"}

try:
    print(f"Sending POST to {url}...")
    response = httpx.post(url, json=payload, timeout=60.0)
    response.raise_for_status()
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))
except httpx.HTTPError as e:
    print(f"HTTP error occurred: {e}")
except Exception as e:
    print(f"Error: {e}")
