import requests

api_key = "test123"
questions = ["Rewrite me the following question: What should I learn to be the best programmer?", "Write me a python function to add 2 numbers", "What is the capital of France?"]

for question in questions:
    print("=" * 100)
    print(f"Q: {question}")
    resp = requests.post("http://127.0.0.1:8000/proxy", json={"api_key": api_key, "query": question})
    print(f"A: {resp.json()}")

