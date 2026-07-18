"""
Flask web app wrapping query_pipe.answer_query() in a simple chat UI.

Run with:
    python app.py
Then open http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Loading the models/DB client happens once, at import time, inside
# query_pipe.py - so the first request after startup will be fast(er)
# instead of paying that cost per-message.
from query_pipe import answer_query  # noqa: E402


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        result = answer_query(message)
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500

    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)