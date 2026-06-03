import csv
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# Add the project root to the Python path so we can import 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.pipeline import iterative_rag

MODE_OPTIONS = {
    "Fast answer": {
        "mode": "single",
        "condition_id": "B",
        "label": "Hybrid, single-pass",
    },
    "Deep search": {
        "mode": "iterative",
        "condition_id": "D",
        "label": "Hybrid, iterative",
    },
}
DEFAULT_MODE_LABEL = "Fast answer"
DEEPEVAL_RESULTS_PATH = Path(project_root) / "results" / "deepeval_fastapi.csv"

# Set page configuration
st.set_page_config(
    page_title="Codebase Intelligence Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("Codebase Intelligence Assistant")
st.markdown("Ask questions about your codebase, and the assistant will find the relevant snippets and generate a grounded answer.")

eval_summary = {}
if DEEPEVAL_RESULTS_PATH.exists():
    with DEEPEVAL_RESULTS_PATH.open(encoding="utf-8", newline="") as handle:
        eval_summary = {row["condition_id"]: row for row in csv.DictReader(handle)}

selected_mode_label = st.sidebar.selectbox(
    "Answer mode",
    options=list(MODE_OPTIONS),
    index=list(MODE_OPTIONS).index(DEFAULT_MODE_LABEL),
)
selected_mode = MODE_OPTIONS[selected_mode_label]
selected_summary = eval_summary.get(selected_mode["condition_id"], {})
st.sidebar.caption(f"{selected_mode['condition_id']}: {selected_mode['label']}")
if selected_summary:
    st.sidebar.metric("Answer correctness", selected_summary.get("answer_correctness", "n/a"))
    st.sidebar.metric("Faithfulness", selected_summary.get("faithfulness", "n/a"))
    st.sidebar.metric("Refusal quality", selected_summary.get("refusal_quality", "n/a"))
    st.sidebar.caption("DeepEval judge results on the FastAPI ablation.")


def render_retrieval_journey(
    trace: object,
    *,
    answer: str,
    mode_label: str,
    condition_label: str,
) -> None:
    if not isinstance(trace, Mapping) or not trace:
        return

    with st.expander("Retrieval journey"):
        st.caption(f"{mode_label} · {condition_label}")
        columns = st.columns(4)
        columns[0].markdown("**Question**")
        columns[0].caption(_first_pass_query(trace))
        columns[1].markdown("**Hybrid retrieval**")
        columns[1].caption("GraphCodeBERT + BM25 -> RRF")
        columns[2].markdown("**Generation**")
        columns[2].caption("Qwen2.5-Coder over retrieved snippets")
        columns[3].markdown("**Citations**")
        columns[3].caption(f"{_final_chunk_count(trace)} snippet(s)")

        total_ms = trace.get("total_ms")
        if total_ms is not None:
            st.caption(f"Total pipeline time: {_format_ms(total_ms)}")

        _render_pass_trace("Pass 1", trace.get("pass_1"))
        _render_pass_trace("Pass 2", trace.get("pass_2"))

        st.markdown("**Final answer**")
        st.markdown(answer)


def _render_pass_trace(label: str, pass_trace: object) -> None:
    if not isinstance(pass_trace, Mapping) or not pass_trace:
        return

    st.markdown(f"**{label}**")
    if pass_trace.get("skipped"):
        st.info(f"Skipped: {pass_trace['skipped']}")
        return

    query = pass_trace.get("query")
    if query:
        st.markdown("Retrieval query")
        st.code(str(query), language="text")

    timing = []
    if pass_trace.get("retrieval_ms") is not None:
        timing.append(f"retrieval {_format_ms(pass_trace['retrieval_ms'])}")
    if pass_trace.get("generation_ms") is not None:
        timing.append(f"generation {_format_ms(pass_trace['generation_ms'])}")
    if timing:
        st.caption(" · ".join(timing))

    chunks = pass_trace.get("retrieved_chunks", [])
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)) and chunks:
        st.markdown("Retrieved snippets")
        for index, chunk in enumerate(chunks[:5], start=1):
            if isinstance(chunk, Mapping):
                st.markdown(f"{index}. {_chunk_label(chunk)}")
        if len(chunks) > 5:
            st.caption(f"{len(chunks) - 5} more snippet(s) available in Retrieved snippets.")


def _first_pass_query(trace: Mapping[object, object]) -> str:
    pass_1 = trace.get("pass_1")
    if isinstance(pass_1, Mapping):
        query = pass_1.get("query")
        if query:
            return _short_text(str(query), limit=120)
    return "n/a"


def _final_chunk_count(trace: Mapping[object, object]) -> int:
    pass_2 = trace.get("pass_2")
    pass_1 = trace.get("pass_1")
    final_pass = pass_2 if isinstance(pass_2, Mapping) and not pass_2.get("skipped") else pass_1
    if not isinstance(final_pass, Mapping):
        return 0
    chunks = final_pass.get("retrieved_chunks", [])
    return len(chunks) if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)) else 0


def _chunk_label(chunk: Mapping[object, object]) -> str:
    filepath = chunk.get("filepath", "Unknown")
    function_name = chunk.get("function_name")
    start_line = chunk.get("start_line", "?")
    end_line = chunk.get("end_line", "?")
    location = f"`{filepath}`"
    if function_name:
        location += f" · `{function_name}`"
    debug = _retrieval_debug_label(chunk.get("retrieval_debug"))
    if debug:
        return f"{location} · lines {start_line}-{end_line} · {debug}"
    return f"{location} · lines {start_line}-{end_line}"


def _retrieval_debug_label(raw_debug: object) -> str:
    if not isinstance(raw_debug, Mapping):
        return ""
    parts = []
    if raw_debug.get("dense_rank") is not None:
        parts.append(f"dense #{raw_debug['dense_rank']}")
    if raw_debug.get("bm25_rank") is not None:
        parts.append(f"BM25 #{raw_debug['bm25_rank']}")
    if raw_debug.get("rrf_score") is not None:
        parts.append(f"RRF {_format_score(raw_debug['rrf_score'])}")
    return ", ".join(parts)


def _format_score(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_ms(value: object) -> str:
    try:
        return f"{float(value):.1f} ms"
    except (TypeError, ValueError):
        return str(value)


def _short_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "confidence_details" in message and message["confidence_details"]:
            details = message["confidence_details"]
            score = details.get("confidence", 0.0)
            raw_answers = details.get("raw_answers", [])
            if raw_answers:
                with st.expander(f"Confidence Analysis (Score: {score:.4f})"):
                    for i, ans in enumerate(raw_answers):
                        st.markdown(f"**Answer Variant {i+1}:**")
                        st.info(ans)
                        
        if message.get("grounded") is False:
            claims = ", ".join([f"`{c}`" for c in message.get("ungrounded_claims", [])])
            st.warning(f"⚠️ **Grounding Warning:** The AI mentioned {claims}, but these could not be found in the retrieved code. This might be a hallucination.")

        render_retrieval_journey(
            message.get("trace"),
            answer=message["content"],
            mode_label=message.get("mode_label", ""),
            condition_label=message.get("condition_label", ""),
        )

        # If there are retrieved chunks saved in the message history, display them in an expander
        if "chunks" in message and message["chunks"]:
            with st.expander("Retrieved snippets"):
                for idx, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"**{idx}. {chunk.get('filepath', 'Unknown')}** (lines {chunk.get('start_line', '?')}-{chunk.get('end_line', '?')})")
                    if chunk.get("function_name"):
                        st.markdown(f"*Function/Class: `{chunk['function_name']}`*")
                    # Optionally show the source code snippet
                    if chunk.get("source"):
                        st.code(chunk["source"], language=chunk.get("language", "python"))

# Accept user input
if prompt := st.chat_input("Ask a question about the codebase..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching codebase and generating answer..."):
            try:
                # The default is condition B from the DeepEval-backed ablation.
                result = iterative_rag(prompt, mode=selected_mode["mode"], confidence=False)
                
                answer = result.get("answer", "No answer generated.")
                chunks = result.get("retrieved_chunks", [])
                
                # Only show a confidence badge when the confidence layer is enabled.
                confidence_level = result.get("confidence_level")
                badge = ""
                if confidence_level == "high":
                    badge = "🟢 "
                elif confidence_level == "medium":
                    badge = "🟡 "
                elif confidence_level == "low":
                    badge = "🔴 "
                final_answer = f"{badge}{answer}"
                
                # Display the main answer
                st.markdown(final_answer)
                
                # Display the confidence analysis
                confidence_details = result.get("confidence_details", {})
                if confidence_details:
                    score = confidence_details.get("confidence", 0.0)
                    raw_answers = confidence_details.get("raw_answers", [])
                    if raw_answers:
                        with st.expander(f"Confidence Analysis (Score: {score:.4f})"):
                            for i, ans in enumerate(raw_answers):
                                st.markdown(f"**Answer Variant {i+1}:**")
                                st.info(ans)

                # Display grounding warning if applicable
                is_grounded = result.get("grounded", True)
                ungrounded_claims = result.get("ungrounded_claims", [])
                if not is_grounded:
                    claims = ", ".join([f"`{c}`" for c in ungrounded_claims])
                    st.warning(f"⚠️ **Grounding Warning:** The AI mentioned {claims}, but these could not be found in the retrieved code. This might be a hallucination.")

                render_retrieval_journey(
                    result.get("trace"),
                    answer=final_answer,
                    mode_label=selected_mode_label,
                    condition_label=selected_mode["label"],
                )

                # Display the retrieved snippets in an expander
                if chunks:
                    with st.expander("Retrieved snippets"):
                        for idx, chunk in enumerate(chunks, 1):
                            st.markdown(f"**{idx}. {chunk.get('filepath', 'Unknown')}** (lines {chunk.get('start_line', '?')}-{chunk.get('end_line', '?')})")
                            if chunk.get("function_name"):
                                st.markdown(f"*Function/Class: `{chunk['function_name']}`*")
                            if chunk.get("source"):
                                st.code(chunk["source"], language=chunk.get("language", "python"))
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_answer,
                    "chunks": chunks,
                    "confidence_details": confidence_details,
                    "grounded": is_grounded,
                    "ungrounded_claims": ungrounded_claims,
                    "trace": result.get("trace", {}),
                    "mode_label": selected_mode_label,
                    "condition_label": selected_mode["label"],
                })
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
