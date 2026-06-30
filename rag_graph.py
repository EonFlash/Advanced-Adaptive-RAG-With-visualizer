"""
Adaptive RAG graph — instrumented for real-time visualization.

Two modes:
  - REAL mode: uses your original LangGraph + Azure OpenAI + Chroma pipeline,
    if AZURE_OPENAI_* env vars are set and a 'Legal_Docs' folder with PDFs exists.
  - DEMO mode (default / fallback): a deterministic, dependency-free simulation
    of the exact same graph topology with realistic timings + occasional
    retries/rewrites, so the visualizer always works out of the box (e.g. on
    Vercel, with no API keys, no vector DB, no PDFs).

Either way, every node emits structured events via a callback so the frontend
can animate the LangGraph exactly as it executes.
"""

import os
import time
import random
import uuid
from typing import List, TypedDict, Literal, Callable, Optional

DEMO_MODE = True  # flipped to False below if real pipeline initializes successfully

# Node metadata shared by both modes — drives the visualizer's graph layout
NODE_META = {
    "decide_retrieval":     {"label": "Decide Retrieval",     "group": "router"},
    "generate_direct":      {"label": "Generate (Direct)",    "group": "terminal"},
    "retrieve":             {"label": "Retrieve Docs",        "group": "retrieval"},
    "is_relevant":          {"label": "Relevance Filter",     "group": "retrieval"},
    "no_answer_found":      {"label": "No Answer Found",      "group": "terminal"},
    "generate_from_context":{"label": "Generate (RAG)",       "group": "generate"},
    "is_sup":               {"label": "IsSUP Verify",         "group": "verify"},
    "revise_answer":        {"label": "Revise Answer",        "group": "verify"},
    "is_use":                {"label": "IsUSE Check",          "group": "verify"},
    "rewrite_question":     {"label": "Rewrite Query",        "group": "retrieval"},
}

EDGES = [
    ("START", "decide_retrieval"),
    ("decide_retrieval", "generate_direct"),
    ("decide_retrieval", "retrieve"),
    ("generate_direct", "END"),
    ("retrieve", "is_relevant"),
    ("is_relevant", "generate_from_context"),
    ("is_relevant", "no_answer_found"),
    ("no_answer_found", "END"),
    ("generate_from_context", "is_sup"),
    ("is_sup", "is_use"),
    ("is_sup", "revise_answer"),
    ("revise_answer", "is_sup"),
    ("is_use", "END"),
    ("is_use", "rewrite_question"),
    ("is_use", "no_answer_found"),
    ("rewrite_question", "retrieve"),
]


def _emit(cb: Optional[Callable], **kwargs):
    if cb:
        cb({"id": str(uuid.uuid4()), "ts": time.time(), **kwargs})


# ------------------------------------------------------------------------
# Attempt to initialize the REAL pipeline. Anything missing -> stay in DEMO.
# ------------------------------------------------------------------------
_real = {}
try:
    required_env = [
        "AZURE_OPENAI_EMBED_DEPLOYMENT", "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    if all(os.environ.get(v) for v in required_env) and os.path.isdir("Legal_Docs"):
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
        from langchain_chroma import Chroma
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import StateGraph, START, END
        from pydantic import BaseModel, Field

        loader = DirectoryLoader(path="Legal_Docs", glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150).split_documents(docs)

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        if not os.path.exists("./chroma_db"):
            vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        else:
            vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0,
        )

        class State(TypedDict):
            question: str
            retrieval_query: str
            rewrite_tries: int
            need_retrieval: bool
            docs: List[Document]
            relevant_docs: List[Document]
            context: str
            answer: str
            issup: Literal["fully_supported", "partially_supported", "no_support"]
            evidence: List[str]
            retries: int
            isuse: Literal["useful", "not_useful"]
            use_reason: str

        class RetrieveDecision(BaseModel):
            should_retrieve: bool = Field(...)

        decide_retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", "You decide whether retrieval is needed.\nReturn JSON with key: should_retrieve (boolean).\n\n"
                       "Guidelines:\n- should_retrieve=True if answering requires specific facts from company documents.\n"
                       "- should_retrieve=False for general explanations/definitions.\n- If unsure, choose True."),
            ("human", "Question: {question}"),
        ])
        should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

        def decide_retrieval(state):
            d = should_retrieve_llm.invoke(decide_retrieval_prompt.format_messages(question=state["question"]))
            return {"need_retrieval": d.should_retrieve}

        def route_after_decide(state):
            return "retrieve" if state["need_retrieval"] else "generate_direct"

        direct_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only your general knowledge.\nIf it requires specific company info, say:\n"
                       "'I don't know based on my general knowledge.'"),
            ("human", "{question}"),
        ])

        def generate_direct(state):
            out = llm.invoke(direct_generation_prompt.format_messages(question=state["question"]))
            return {"answer": out.content}

        def retrieve(state):
            q = state.get("retrieval_query") or state["question"]
            return {"docs": retriever.invoke(q)}

        class RelevanceDecision(BaseModel):
            is_relevant: bool = Field(...)

        is_relevant_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are judging document relevance at a TOPIC level.\nReturn JSON matching the schema.\n"
                       "A document is relevant if it discusses the same entity or topic area as the question.\n"
                       "When unsure, return is_relevant=true."),
            ("human", "Question:\n{question}\n\nDocument:\n{document}"),
        ])
        relevance_llm = llm.with_structured_output(RelevanceDecision)

        def is_relevant(state):
            relevant_docs = []
            for doc in state.get("docs", []):
                d = relevance_llm.invoke(is_relevant_prompt.format_messages(question=state["question"], document=doc.page_content))
                if d.is_relevant:
                    relevant_docs.append(doc)
            return {"relevant_docs": relevant_docs}

        def route_after_relevance(state):
            return "generate_from_context" if state.get("relevant_docs") else "no_answer_found"

        rag_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a business rag chatbot.\nAnswer the question based on the context.\n"
                       "Dont mention that you are getting a context in your answer"),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ])

        def generate_from_context(state):
            context = "\n\n---\n\n".join(d.page_content for d in state.get("relevant_docs", [])).strip()
            if not context:
                return {"answer": "No answer found.", "context": ""}
            out = llm.invoke(rag_generation_prompt.format_messages(question=state["question"], context=context))
            return {"answer": out.content, "context": context}

        def no_answer_found(state):
            return {"answer": "No answer found.", "context": ""}

        class IsSUPDecision(BaseModel):
            issup: Literal["fully_supported", "partially_supported", "no_support"]
            evidence: List[str] = Field(default_factory=list)

        issup_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
                       "Return JSON with keys: issup, evidence. Be strict."),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n"),
        ])
        issup_llm = llm.with_structured_output(IsSUPDecision)

        def is_sup(state):
            d = issup_llm.invoke(issup_prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")))
            return {"issup": d.issup, "evidence": d.evidence}

        MAX_RETRIES = 10

        def route_after_issup(state):
            if state.get("issup") == "fully_supported":
                return "accept_answer"
            if state.get("retries", 0) >= MAX_RETRIES:
                return "accept_answer"
            return "revise_answer"

        revise_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a STRICT reviser. Output ONLY direct quotes from CONTEXT as bullet points."),
            ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"),
        ])

        def revise_answer(state):
            out = llm.invoke(revise_prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")))
            return {"answer": out.content, "retries": state.get("retries", 0) + 1}

        class IsUSEDecision(BaseModel):
            isuse: Literal["useful", "not_useful"]
            reason: str = Field(...)

        isuse_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are judging USEFULNESS of the ANSWER for the QUESTION.\nReturn JSON with keys: isuse, reason."),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
        ])
        isuse_llm = llm.with_structured_output(IsUSEDecision)

        def is_use(state):
            d = isuse_llm.invoke(isuse_prompt.format_messages(question=state["question"], answer=state.get("answer", "")))
            return {"isuse": d.isuse, "use_reason": d.reason}

        MAX_REWRITE_TRIES = 3

        def route_after_isuse(state):
            if state.get("isuse") == "useful":
                return "END"
            if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
                return "no_answer_found"
            return "rewrite_question"

        class RewriteDecision(BaseModel):
            retrieval_query: str = Field(...)

        rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user's QUESTION into a query optimized for vector retrieval. Output JSON with key: retrieval_query"),
            ("human", "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer (if any):\n{answer}"),
        ])
        rewrite_llm = llm.with_structured_output(RewriteDecision)

        def rewrite_question(state):
            d = rewrite_llm.invoke(rewrite_for_retrieval_prompt.format_messages(question=state["question"], retrieval_query=state.get("retrieval_query", ""), answer=state.get("answer", "")))
            return {"retrieval_query": d.retrieval_query, "rewrite_tries": state.get("rewrite_tries", 0) + 1, "docs": [], "relevant_docs": [], "context": ""}

        g = StateGraph(State)
        g.add_node("decide_retrieval", decide_retrieval)
        g.add_node("generate_direct", generate_direct)
        g.add_node("retrieve", retrieve)
        g.add_node("is_relevant", is_relevant)
        g.add_node("generate_from_context", generate_from_context)
        g.add_node("no_answer_found", no_answer_found)
        g.add_node("is_sup", is_sup)
        g.add_node("revise_answer", revise_answer)
        g.add_node("is_use", is_use)
        g.add_node("rewrite_question", rewrite_question)

        g.add_edge(START, "decide_retrieval")
        g.add_conditional_edges("decide_retrieval", route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
        g.add_edge("generate_direct", END)
        g.add_edge("retrieve", "is_relevant")
        g.add_conditional_edges("is_relevant", route_after_relevance, {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"})
        g.add_edge("no_answer_found", END)
        g.add_edge("generate_from_context", "is_sup")
        g.add_conditional_edges("is_sup", route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
        g.add_edge("revise_answer", "is_sup")
        g.add_conditional_edges("is_use", route_after_isuse, {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"})
        g.add_edge("rewrite_question", "retrieve")

        _real["app"] = g.compile()
        DEMO_MODE = False
except Exception:
    DEMO_MODE = True


# ------------------------------------------------------------------------
# Public entry point used by Flask
# ------------------------------------------------------------------------
def run_graph_streaming(question: str, on_event: Callable):
    """Runs the graph (real or demo) and calls on_event(dict) for each step."""
    if DEMO_MODE:
        return _run_demo(question, on_event)
    return _run_real(question, on_event)


def _run_real(question: str, on_event: Callable):
    app = _real["app"]
    state = {
        "question": question, "retrieval_query": question, "rewrite_tries": 0,
        "docs": [], "relevant_docs": [], "context": "", "answer": "",
        "issup": "", "evidence": [], "retries": 0, "isuse": "not_useful", "use_reason": "",
    }
    final_state = dict(state)
    _emit(on_event, type="start", question=question, mode="real")
    for update in app.stream(state, config={"recursion_limit": 80}, stream_mode="updates"):
        for node_name, node_output in update.items():
            _emit(on_event, type="node_start", node=node_name)
            final_state.update(node_output or {})
            _emit(on_event, type="node_end", node=node_name, data=_safe(node_output))
    _emit(on_event, type="final", answer=final_state.get("answer", ""), state=_safe(final_state))
    return final_state


def _safe(d):
    """Strip non-JSON-serializable objects (e.g. langchain Document) for SSE payloads."""
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        if k in ("docs", "relevant_docs"):
            out[k] = [{"source": (doc.metadata or {}).get("source", "unknown"),
                        "page": (doc.metadata or {}).get("page"),
                        "snippet": doc.page_content[:180]} for doc in (v or [])]
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------------
# DEMO MODE — deterministic, fast, dependency-free simulation
# ------------------------------------------------------------------------
_DEMO_DOCS = {
    "termination": [
        {"source": "Orbis_Financial_HR_Policy.pdf", "page": 4, "snippet": "Employees must provide a minimum of 30 days written notice prior to termination of employment, except during the probationary period where 7 days applies."},
        {"source": "Orbis_Financial_HR_Policy.pdf", "page": 5, "snippet": "The Company reserves the right to terminate employment immediately for cause, including gross misconduct."},
    ],
    "refund": [
        {"source": "NexaAI_Pricing_Terms.pdf", "page": 2, "snippet": "NexaAI offers a full refund within 14 days of purchase for annual plans, provided usage has not exceeded 1,000 API calls."},
        {"source": "NexaAI_Pricing_Terms.pdf", "page": 3, "snippet": "Monthly subscriptions are non-refundable but may be cancelled at any time, effective at the end of the billing cycle."},
    ],
    "trial": [
        {"source": "NexaAI_Pricing_Terms.pdf", "page": 1, "snippet": "All NexaAI plans include a 14-day free trial with full feature access and no credit card required."},
    ],
    "leave": [
        {"source": "Orbis_Financial_HR_Policy.pdf", "page": 7, "snippet": "Full-time employees accrue 1.5 days of paid leave per month, capped at 18 days annually, plus 10 public holidays."},
    ],
    "default": [
        {"source": "Orbis_Financial_Company_Profile.pdf", "page": 1, "snippet": "Orbis Financial is a mid-sized advisory firm headquartered in Singapore, founded in 2014, operating across 6 regional offices."},
        {"source": "NexaAI_Pricing_Terms.pdf", "page": 1, "snippet": "NexaAI provides tiered SaaS pricing across Starter, Growth, and Enterprise plans billed monthly or annually."},
    ],
}


# Topics the curated demo documents actually cover. Anything else is treated
# as "unknown" so we don't fabricate a misleading legal-sounding answer for
# an arbitrary recruiter question (e.g. "hi", "tell me a joke").
_TOPIC_KEYWORDS = ("termination", "notice", "refund", "trial", "leave", "pricing",
                   "orbis", "nexaai", "policy", "hr ", "benefits", "vacation")


def _match_topic(question: str):
    ql = question.lower()
    for key in ("termination", "refund", "trial", "leave"):
        if key in ql:
            return key
    if any(m in ql for m in _TOPIC_KEYWORDS):
        return "default"
    return None


# Deterministic outcome per curated preset topic, so the preset buttons always
# demonstrate the specific branch they're labeled for instead of leaving it to
# chance: one clean grounded answer (termination), one IsSUP revise loop
# (refund), one IsUSE rewrite loop (trial), one no-answer dead end (leave).
# Anything else (the "default" bucket — company-related but not one of these
# four curated topics) still uses the randomized behavior below so ad-hoc
# questions stay lively and unpredictable.
_SCRIPTED_OUTCOMES = {
    "termination": {"revise": False, "rewrite": False, "force_no_answer": False},
    "refund":      {"revise": True,  "rewrite": False, "force_no_answer": False},
    "trial":       {"revise": False, "rewrite": True,  "force_no_answer": False},
    "leave":       {"revise": False, "rewrite": False, "force_no_answer": True},
}


GUIDANCE_ANSWER = (
    "This question isn't covered by the sample documents loaded in this demo "
    "(Orbis Financial HR policy + NexaAI pricing terms), so I'm not going to "
    "fabricate a grounded-sounding answer for it.\n\n"
    "Try one of the preset questions on the left to see a fully grounded "
    "retrieval -> verify -> revise cycle run end-to-end - or just watch the "
    "graph above, it ran the routing, retrieval and relevance steps for "
    "this exact question."
)


def _run_demo(question: str, on_event: Callable):
    rnd = random.Random(hash(question) & 0xFFFFFFFF)

    def step(node, delay, payload=None, detail=None):
        _emit(on_event, type="node_start", node=node)
        time.sleep(delay)
        _emit(on_event, type="node_end", node=node, data=payload or {}, detail=detail)

    _emit(on_event, type="start", question=question, mode="demo")

    topic = _match_topic(question)
    need_retrieval = topic is not None or len(question.split()) > 6
    step("decide_retrieval", 0.4, {"need_retrieval": need_retrieval},
         f"LLM judged that {'company-specific documents are required' if need_retrieval else 'general knowledge is sufficient'}.")

    if not need_retrieval:
        answer = (f"\"{question.strip()}\" looks like general knowledge rather than something that needs "
                  "internal company documents, so the graph answered directly without retrieval - that "
                  "routing decision is the first thing it does on every question.")
        step("generate_direct", 0.6, {"answer": answer})
        _emit(on_event, type="final", answer=answer, state={"need_retrieval": False})
        return {"answer": answer}

    docs = _DEMO_DOCS.get(topic or "default", _DEMO_DOCS["default"])
    scripted = _SCRIPTED_OUTCOMES.get(topic)

    step("retrieve", 0.45, {"docs": docs}, f"Retrieved {len(docs)} candidate chunks from the vector store (k=4).")

    if scripted and scripted["force_no_answer"]:
        relevant = []
    else:
        relevant = docs if (topic is not None or rnd.random() > 0.5) else []
    step("is_relevant", 0.5, {"relevant_docs": relevant}, f"{len(relevant)}/{len(docs)} chunks passed the topical relevance filter.")

    if not relevant:
        step("no_answer_found", 0.3, {"answer": GUIDANCE_ANSWER})
        _emit(on_event, type="final", answer=GUIDANCE_ANSWER, state={})
        return {"answer": GUIDANCE_ANSWER}

    context = " ".join(d["snippet"] for d in relevant)
    draft = _canned_rag_answer(relevant)
    step("generate_from_context", 0.55, {"answer": draft, "context": context[:300] + "..."})

    # IsSUP: at most ONE revise loop, ever. Scripted for curated preset topics
    # so the "revise" preset always demonstrates the loop; random otherwise.
    retries = 0
    will_revise = scripted["revise"] if scripted else (rnd.random() < 0.45)
    issup = "partially_supported" if will_revise else "fully_supported"
    step("is_sup", 0.4, {"issup": issup, "evidence": [relevant[0]["snippet"][:90] + "..."]},
         f"Verifier marked the answer as '{issup}'.")
    if will_revise:
        retries = 1
        draft = _tighten_answer(draft)
        step("revise_answer", 0.45, {"answer": draft, "retries": retries}, "Rewrote answer as strict quote-only bullets from context.")
        step("is_sup", 0.35, {"issup": "fully_supported", "evidence": [relevant[0]["snippet"][:90] + "..."]},
             "Verifier marked the revised answer as 'fully_supported'.")

    # IsUSE: at most ONE rewrite loop, ever — independent of whether a revise just
    # happened, so the two loops aren't competing for the same probability budget.
    # Scripted for curated preset topics so the "rewrite" preset always demonstrates
    # the loop; random otherwise.
    rewrite_tries = 0
    will_rewrite = scripted["rewrite"] if scripted else (rnd.random() < 0.4)
    isuse = "not_useful" if will_rewrite else "useful"
    step("is_use", 0.35, {"isuse": isuse, "use_reason": "Answer is too generic / off-topic." if will_rewrite else "Directly answers the question."})

    if not will_rewrite:
        _emit(on_event, type="final", answer=draft, state={"retries": retries, "rewrite_tries": 0})
        return {"answer": draft}

    rewrite_tries = 1
    new_query = f"{question.strip('? ')} -- refined keywords"
    step("rewrite_question", 0.4, {"retrieval_query": new_query, "rewrite_tries": rewrite_tries}, "Rewrote retrieval query with higher-signal keywords.")
    step("retrieve", 0.4, {"docs": docs}, "Re-ran retrieval with the rewritten query.")
    step("is_relevant", 0.4, {"relevant_docs": docs}, f"{len(docs)}/{len(docs)} chunks passed the relevance filter this time.")
    draft = _canned_rag_answer(docs)
    step("generate_from_context", 0.5, {"answer": draft, "context": context[:300] + "..."})
    step("is_sup", 0.35, {"issup": "fully_supported", "evidence": [docs[0]["snippet"][:90] + "..."]}, "Verifier marked the answer as 'fully_supported'.")
    step("is_use", 0.3, {"isuse": "useful", "use_reason": "Directly answers the question after query rewrite."})

    _emit(on_event, type="final", answer=draft, state={"retries": retries, "rewrite_tries": rewrite_tries})
    return {"answer": draft}


def _canned_rag_answer(relevant_docs) -> str:
    facts = "; ".join(d["snippet"].split(".")[0] for d in relevant_docs[:2])
    return f"{facts}."


def _tighten_answer(answer: str) -> str:
    return "- " + answer.replace(". ", "\n- ").strip()
