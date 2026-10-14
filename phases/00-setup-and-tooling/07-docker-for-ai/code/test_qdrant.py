"""
Verify connection to Qdrant Vector Database (Exercise 2)
Run inside container or network: python test_qdrant.py
"""

import sys
import requests
from qdrant_client import QdrantClient


def test_qdrant_connection(host: str = "qdrant", port: int = 6333):
    print(f"Connecting to Qdrant at http://{host}:{port}...")

    # 1. Test via direct HTTP endpoint (Exercise 2 requirement: http://qdrant:6333/collections)
    url = f"http://{host}:{port}/collections"
    try:
        response = requests.get(url, timeout=5)
        print(f"HTTP GET {url} status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"HTTP request failed: {e}")

    # 2. Test via QdrantClient
    try:
        client = QdrantClient(host=host, port=port, timeout=5)
        collections = client.get_collections()
        print(f"QdrantClient collections: {collections}")
        print("Successfully connected to Qdrant!")
        return True
    except Exception as e:
        print(f"QdrantClient error: {e}")
        return False


if __name__ == "__main__":
    host_arg = sys.argv[1] if len(sys.argv) > 1 else "qdrant"
    port_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 6333
    success = test_qdrant_connection(host=host_arg, port=port_arg)
    sys.exit(0 if success else 1)
