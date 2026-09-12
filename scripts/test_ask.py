"""Test script for verifying the FastAPI /ask endpoint with sample queries."""

import json
from typing import Any
import httpx

DEFAULT_URL: str = "http://localhost:8000/ask"
DEFAULT_QUESTION: str = "What are the important articles for Indian people"


def send_test_query(
    question: str = DEFAULT_QUESTION,
    url: str = DEFAULT_URL,
    timeout: float = 60.0,
) -> None:
    """Send a POST request to the /ask endpoint and print the JSON response.

    Args:
        question: Constitutional question to send to the backend.
        url: Full URL to the /ask endpoint.
        timeout: HTTP request timeout in seconds.
    """
    payload: dict[str, str] = {"question": question}

    try:
        print(f"Sending POST to {url}...")
        response: httpx.Response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        print("\nResponse:")
        print(json.dumps(data, indent=2))
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    send_test_query()
