# 🧠 The Complete Architecture Walkthrough (Code Deep-Dive)

This document provides a deeply detailed, step-by-step explanation of exactly what happened behind the scenes when you asked: *"What does the Greeter class do?"*

It traces the data from the moment you hit "Enter" on your keyboard to the moment the green 🟢 badge appeared on your screen, explaining both the conceptual architecture and the exact Python code that executed it.

---

## Step 1: The User Interface (`app/streamlit_app.py`)

When you type your question into the chat box, the journey begins in the frontend.

*   **The Tool:** **Streamlit**. Streamlit is a Python library that instantly turns Python scripts into interactive web applications. It constantly reruns the script from top-to-bottom every time you interact with it.
*   **What happens:** The code captures your string `"What does the Greeter class do?"`. It draws a loading spinner and hands your question off to the `iterative_rag` orchestrator.

**The Code (`streamlit_app.py`):**
```python
# 1. Accept user input
if prompt := st.chat_input("Ask a question about the codebase..."):
    
    # 2. Draw the chat bubble
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Trigger the backend pipeline and show a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching codebase and generating answer..."):
            
            # This is the handoff to the backend!
            result = iterative_rag(prompt, mode="iterative", confidence=True)
```

## Step 2: The Orchestrator (`src/pipeline.py`)

This file acts as the traffic cop. It doesn't do the heavy lifting itself, but it knows exactly which files to call and in what order. 

*   **What happens:** Streamlit calls `iterative_rag`. The orchestrator realizes you want the "iterative" mode (which requires two passes) and that you want the "confidence" check enabled. It kicks off Pass 1.

**The Code (`src/pipeline.py`):**
```python
def iterative_rag(query: str, mode: str = "single", confidence: bool = False, ...) -> dict:
    
    # Trigger Pass 1 (Drafting and initial search)
    pass_1 = _run_pass(
        retriever_fn=retriever_fn,
        generator_fn=generate_fn,
        retrieval_query=query,
        generation_query=query,
    )
    
    # If mode is iterative, trigger Pass 2 (Refining)
    if mode == "iterative":
        pass_2 = iterative_pass(query=query, partial_answer=pass_1.answer, ...)
    
    # If confidence is True, trigger the HonestCoder paranoia layer
    if confidence:
        _apply_confidence(result, query=query, generator_fn=generate_fn, ...)
        
    return result
```

## Step 3: The Hallucination Draft (`src/generator.py`)

Before we even search your actual code files, we do a neat trick called **HyDE (Hypothetical Document Embeddings)**. 

*   **The Tool:** **Ollama** running the **Qwen2.5-Coder (7 Billion parameters)** model. 
*   **What happens:** The orchestrator sends your question to `generator.py` with `is_draft=True`. It asks the AI to generate a hypothetical, fake answer to your question without looking at the code. This fake answer generates highly relevant keywords that make our actual codebase search much more accurate.

**The Code (`src/generator.py`):**
```python
def generate(context_chunks: Sequence[dict], query: str, is_draft: bool = False, ...) -> str:
    
    # If it's a draft, ignore the codebase entirely and just hallucinate
    if is_draft:
        prompt = (
            "You are an expert developer. Please write a hypothetical, plausible snippet of code "
            f"or documentation that would answer the following query: {query}"
        )
    else:
        # Otherwise, build a massive prompt containing the actual retrieved code chunks
        prompt = build_context_prompt(context_chunks, query)
        
    # Send the prompt to Ollama running locally on port 11434
    response = client.chat.completions.create(
        model="qwen2.5-coder:7b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## Step 4: The Hybrid Codebase Search (`src/retriever.py`)

Now we take your original question AND the fake draft answer, and we go hunting in your codebase. We use two completely different search engines at the exact same time (a Hybrid search).

*   **Search Engine A (BM25 Keyword Hunter):** `src/bm25_index.py` scans your project for exact keyword matches (e.g., the exact word "Greeter").
*   **Search Engine B (ChromaDB "Vibe" Hunter):** `src/index_dense.py` uses the **GraphCodeBERT** model to convert your question into a mathematical vector. It searches for code with a similar *meaning*, even if the exact words are missing.
*   **The Merger:** `src/retriever.py` uses **Reciprocal Rank Fusion (RRF)** to mash the two lists together, pushing the best files (like `sample.py`) to the absolute top.

**The Code (`src/retriever.py`):**
```python
def hybrid_search(query: str, top_k: int = 5) -> list[Chunk]:
    
    # 1. Ask BM25 for the top keyword matches
    sparse_results = search_sparse(query, top_k=top_k * 2)
    
    # 2. Ask ChromaDB/GraphCodeBERT for the top semantic meaning matches
    dense_results = search_dense(query, top_k=top_k * 2)
    
    # 3. Merge the two lists using math (RRF)
    merged_chunks = rrf_merge(sparse_results, dense_results, top_k=top_k)
    
    return merged_chunks
```

## Step 5: Reading the Code & Final Answer (`src/pipeline.py` & `src/generator.py`)

We finally have the top 5 code snippets from the Hybrid search. Now we need the AI to read them and answer your question.

*   **What happens:** The orchestrator takes the 5 snippets of `sample.py` and the partial answer from Pass 1, bundles them together, and runs the generator again.
*   **The Output:** Ollama reads the actual code and generates **Answer Variant 1**: *"The Greeter class is used to build greeting messages..."*

**The Code (`src/pipeline.py`):**
```python
def iterative_pass(query: str, partial_answer: str, retriever, generator_fn) -> _PassResult:
    # Combine the user's question with the hallucinated draft answer to make the ultimate search query
    refined_query = f"{query}\n\nPartial answer:\n{partial_answer}"
    
    # Run the retriever using the ultimate query, then generate the final answer
    return _run_pass(
        retrieval_query=refined_query, 
        generation_query=query, 
        retriever_fn=retriever, 
        generator_fn=generator_fn
    )
```

## Step 6: The Paranoia Layer (`src/confidence.py`)

Because you enabled `confidence=True`, the system doesn't just blindly trust Answer Variant 1. We need to double-check its work to prevent hallucinations.

*   **What happens:** `confidence.py` secretly talks to Ollama two more times. It asks the exact same question with the exact same code, but it changes the **Temperature** (0.0, 0.5, 0.9) to make the AI more "creative" or "unhinged".
*   **The Math:** We use the **Sentence Transformers (all-MiniLM-L6-v2)** AI model to mathematically calculate how similar the 3 answers are. Because all three answers were highly similar, the math calculates a final score of **0.7604**, officially awarding a **High Confidence ("high")** rating.

**The Code (`src/confidence.py`):**
```python
def estimate_confidence(query: str, context_chunks: list[dict], generator_fn: Callable, ...) -> ConfidenceResult:
    
    # 1. Ask Ollama the exact same question 3 times at different temperatures
    temps = [0.0, 0.5, 0.9]
    answers = [generator_fn(context_chunks, query, temperature=t) for t in temps]
    
    # 2. Load the SentenceTransformer model to do the math
    model = _load_similarity_model()
    embeddings = model.encode(answers)
    
    # 3. Calculate Cosine Similarity between Variant 1 & Variant 2, and Variant 1 & Variant 3
    similarity_0_1 = util.cos_sim(embeddings[0], embeddings[1]).item()
    similarity_0_2 = util.cos_sim(embeddings[0], embeddings[2]).item()
    
    # 4. Average them out for the final score!
    confidence = (similarity_0_1 + similarity_0_2) / 2
    
    # 5. Assign the badge level
    level = "high" if confidence >= threshold else "low"
    
    return ConfidenceResult(answer=answers[0], confidence=confidence, level=level, raw_answers=answers)
```

## Step 7: The Final Display (`app/streamlit_app.py`)

The orchestrator packages everything up: Answer 1, the 5 code snippets, the 0.7604 score, and the 3 variant texts. It sends this massive package back to the Streamlit frontend.

*   **What happens:** Streamlit sees the "high" rating, slaps a green 🟢 badge onto the answer, prints the text, hides the variants inside the "Confidence Analysis" expander, hides the code inside the "Retrieved snippets" expander, and saves it all to `st.session_state`.

**The Code (`app/streamlit_app.py`):**
```python
# Format the final answer string with the badge
badge = "🟢" if confidence_level == "high" else "🔴"
final_answer = f"{badge} {answer}"

# Display the main answer on the webpage
st.markdown(final_answer)

# Display the confidence analysis variants in an expander
with st.expander(f"Confidence Analysis (Score: {score:.4f})"):
    for i, ans in enumerate(raw_answers):
        st.info(ans)

# Display the retrieved snippets in an expander
with st.expander("Retrieved snippets"):
    for chunk in chunks:
        st.code(chunk["source"], language="python")

# Save permanently to session state so it survives screen refreshes
st.session_state.messages.append({
    "role": "assistant", 
    "content": final_answer,
    "chunks": chunks,
    "confidence_details": confidence_details
})
```
