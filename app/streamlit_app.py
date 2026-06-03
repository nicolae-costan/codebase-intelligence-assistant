import csv
import os
import sys
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
                    "ungrounded_claims": ungrounded_claims
                })
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
