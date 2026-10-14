"""
Simple Flask API Server for AI Engineering (Exercise 3)
Run inside container: python app.py
Exposed on: http://0.0.0.0:5000
"""

from flask import Flask, jsonify, request
import torch

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "healthy",
        "service": "AI Development Container API",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    })


@app.route("/predict", methods=["POST"])
def dummy_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    return jsonify({
        "input": text,
        "token_count": len(text.split()),
        "message": "Model inference mockup successful!"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
