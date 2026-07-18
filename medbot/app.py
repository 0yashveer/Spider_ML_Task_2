import os
import logging

from flask import Flask, render_template, request, jsonify
from query_pipe import answer_query

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000  # adjust to taste; guards against huge/abusive payloads


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    # Renamed from "messages" -> "message" since a single string is being
    # sent, not a conversation history. If you actually want to send
    # multi-turn history to answer_query, this needs a different shape
    # (e.g. a list of {"role": ..., "content": ...} dicts) -- let me know
    # and I'll adjust the contract + answer_query call accordingly.
    message = data.get("message")
    history = data.get("history") or []

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "empty message"}), 400
    
    if not isinstance(history, list):
        history = []
    history = [
        h for h in history
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)
    ][-8:]

    message = message.strip()

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"message too long (max {MAX_MESSAGE_LENGTH} chars)"}), 400

    try:
        result = answer_query(message, history=history)
    except Exception:
        # Log the full traceback server-side, but don't leak internals to the client
        logger.exception("answer_query failed")
        return jsonify({"error": "something went wrong while processing your request"}), 500

    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)