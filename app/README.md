# Streamlit Chat Interface Explanation

This document provides a detailed breakdown of the `streamlit_app.py` file, what was implemented, how the code is structured, and a brief overview of how Streamlit works under the hood.

## What Was Done
For Task 9 (T9), we built a fully decoupled web frontend for the Codebase Intelligence Assistant using Streamlit. The UI allows users to ask questions about the codebase in a chat-like interface. It connects to the `iterative_rag` pipeline, displays a loading spinner during the slow retrieval and LLM generation steps, formats the final answer with a confidence badge, and neatly tucks the cited source code into a collapsible expander.

## How the Code Works (Block by Block)

Here is an explanation of what each major section in `streamlit_app.py` does:

### 1. Path Resolution
```python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```
When running a script inside a subdirectory (like the `app/` folder), Python doesn't automatically know how to import modules from the parent folder. This block dynamically finds the parent project folder and adds it to Python's internal search path (`sys.path`) so `from src.pipeline import iterative_rag` works flawlessly.

### 2. Page Configuration
```python
st.set_page_config(...)
```
This is a built-in Streamlit function that sets the browser tab title, the favicon (🧠), and the layout style of the page. It must always be the first Streamlit command in the script.

### 3. Session State Initialization
```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```
Streamlit is stateless by default (it forgets everything on every rerun). To build a chat app, we use a special dictionary called `st.session_state` to permanently remember the chat history (`messages`). If it doesn't exist yet, we initialize it as an empty list.

### 4. Rendering Past Messages
```python
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # ... logic for drawing expanders
```
Because Streamlit refreshes the screen from scratch every time you do something, we have to loop through our saved chat history and explicitly redraw every past message and its accompanying code snippets onto the screen.

### 5. Capturing User Input
```python
if prompt := st.chat_input("Ask a question about the codebase..."):
```
This renders a sleek chat box at the bottom of the screen. When the user types a message and hits Enter, the code inside the `if` block executes, and `prompt` contains the user's text.

### 6. Calling the Backend Pipeline
```python
with st.spinner("Searching codebase and generating answer..."):
    result = iterative_rag(prompt, mode="iterative", confidence=True)
```
`st.spinner` pauses the UI and shows a loading animation. Inside this block, we call the `iterative_rag` orchestrator from T7. We pass `confidence=True` to activate the HonestCoder layer (T10).

### 7. Badges and Expanders
Once `iterative_rag` returns the result dictionary:
*   **Confidence Badge:** We read `result.get("confidence_level")` and prepend a 🟢, 🟡, or 🔴 to the answer.
*   **Confidence Analysis Expander:** We extract `result.get("confidence_details")` and use `st.expander` to display the exact numerical confidence score and all 3 raw answer variants that were secretly generated in the background, allowing the user to verify how the AI changed its output across different temperatures.
*   **Source Code Expander:** We use `with st.expander("Retrieved snippets"):` to create another collapsible UI box. Inside, we loop over `chunks` and use `st.code()` to display the exact source code that the AI used to generate its answer, ensuring transparency and trustworthiness.
*   **Saving State:** Finally, we append the AI's final answer, the retrieved chunks, and the confidence details back into `st.session_state.messages` so they survive the next screen refresh.

---

## How Streamlit Works 

Streamlit's architecture is unique compared to traditional web frameworks like React or Vue:

1. **Top-to-Bottom Execution:** Every time a user interacts with the app (e.g., typing in the chat box, clicking a button, opening an expander), Streamlit **reruns the entire Python script from line 1 to the end**.
2. **Declarative UI:** You don't write HTML or manage DOM elements. You simply declare what should be on the screen sequentially (e.g., `st.title`, then `st.markdown`, then `st.chat_message`).
3. **Session State is King:** Because of the constant top-to-bottom reruns, any variable you define normally (like `x = 5`) gets wiped out and reset on every interaction. To make data persist across clicks (like chat history), you *must* store it inside `st.session_state`.
