import json
import queue
import threading

from flask import Flask, Response, render_template, request, jsonify

import rag_graph

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", demo_mode=rag_graph.DEMO_MODE)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "mode": "demo" if rag_graph.DEMO_MODE else "real"})


@app.route("/api/ask")
def ask():
    question = (request.args.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    q: "queue.Queue" = queue.Queue()
    SENTINEL = object()

    def on_event(evt):
        q.put(evt)

    def worker():
        try:
            rag_graph.run_graph_streaming(question, on_event)
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        yield "retry: 2000\n\n"
        while True:
            evt = q.get()
            if evt is SENTINEL:
                break
            yield f"data: {json.dumps(evt)}\n\n"

    return Response(stream(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
