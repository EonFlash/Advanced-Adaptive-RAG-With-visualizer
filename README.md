# Adaptive RAG — Live Reasoning Visualizer

A real-time, animated visualizer for an Adaptive/Self-Correcting RAG graph
(decide-retrieval → retrieve → relevance filter → generate → IsSUP verify/revise
loop → IsUSE check → query rewrite loop), built with **Flask only** (no Node
backend) and a dependency-free vanilla JS + SVG frontend.

Every node lights up, pulses, and streams a live trace entry the instant it
runs in the real LangGraph graph — including retry loops (`revise_answer`)
and query-rewrite loops (`rewrite_question`) — via Server-Sent Events.

## Two modes

| Mode | When it's used | Requirements |
|---|---|---|
| **Demo** | Default. Auto-enabled if Azure env vars or a `Legal_Docs/` folder are missing. | None — works instantly, anywhere, including a fresh Vercel deploy. |
| **Live** | Auto-enabled when all `AZURE_OPENAI_*` env vars are set **and** a `Legal_Docs/` folder with PDFs exists next to `app.py`. | `pip install -r requirements-real.txt`, Azure OpenAI access, a `Legal_Docs/` folder of PDFs. |

Demo mode runs the *exact same graph topology* with simulated timings and
occasional randomized retries/rewrites so the visualizer always looks alive —
this is what you want running on your public resume link.

## Run locally

```bash
pip install -r requirements.txt
# optional, for live mode:
pip install -r requirements-real.txt
python app.py
```

Open http://localhost:5000.

## Deploy to Vercel

```bash
vercel deploy
```

`vercel.json` is already configured to run `app.py` as a Python serverless
function and serve `static/` directly. Demo mode needs zero environment
variables. For live mode on Vercel, add the `AZURE_OPENAI_*` env vars in the
project settings — note that Vercel's serverless functions have an execution
time limit, so very long retry/rewrite chains in live mode may time out; demo
mode has no such limit since it's fully simulated.

## Project structure

```
app.py              Flask app + /api/ask SSE streaming endpoint
rag_graph.py         Graph logic: real LangGraph pipeline + demo simulation,
                     both instrumented to emit step events
templates/index.html Visualizer page
static/css/style.css Dark "control room" theme
static/js/app.js     SVG graph rendering + SSE client + animations
```

## Customizing for live mode

`rag_graph.py` mirrors your original `Adaptive_Rag.py` node-for-node (same
prompts, same routing, same MAX_RETRIES / MAX_REWRITE_TRIES). Drop your
`Legal_Docs/` PDFs alongside `app.py`, set the five `AZURE_OPENAI_*` env vars,
install `requirements-real.txt`, and the app automatically switches out of
demo mode — no code changes needed.
