"""HTTP-level tests against the real Flask API and native BM25 binary."""
from api.app import app


def test_search_returns_native_bm25_results():
    with app.test_client() as client:
        response = client.post("/v1/search", json={"query": "machine learning", "mode": "bm25", "top_k": 3})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["mode"] == "bm25"
        assert payload["results"]
        assert all("rank" in result for result in payload["results"])


def test_search_rejects_bad_request_payloads():
    with app.test_client() as client:
        assert client.post("/v1/search", data="not-json").status_code == 400
        assert client.post("/v1/search", json=["not", "an", "object"]).status_code == 400
        assert client.post("/v1/search", json={"query": "hello", "top_k": "bad"}).status_code == 400
        assert client.post("/v1/search", json={"query": "hello", "mode": "invalid"}).status_code == 400
        assert client.get("/v1/autocomplete?limit=bad").status_code == 400
