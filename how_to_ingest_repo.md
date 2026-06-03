# 📥 How to Analyze a New GitHub Repository

If you want the AI to analyze a completely different codebase (like an open-source project or your own work), you need to download it and rebuild the databases. 

Here is the exact step-by-step process:

## Step 1: Stop the Streamlit App
If your Streamlit app is currently running in your terminal, stop it by pressing `Ctrl + C`.

## Step 2: Download the Repository
We have a built-in script that will automatically `git clone` a repository from GitHub and place it cleanly into our `data/raw/` directory.

Run this command in your terminal, replacing the URL with the GitHub project you want to analyze:
```bash
python -m src.ingest --url https://github.com/username/project-name.git
```
*(When it finishes, it will print out the exact folder path where it saved the code, which will look something like `data/raw/project-name`)*

## Step 3: Build the Databases
Now we need to run the Chunker, build the BM25 Sparse Index, and build the ChromaDB Dense Index. 

Instead of running all three scripts manually, we have a combined "builder" script that does everything at once. Run this command, making sure to replace `project-name` with the exact folder name from Step 2:

```bash
python -m src.build_index --repo data/raw/project-name
```

> [!WARNING]
> This step can take a long time! The script has to read every single Python file, break it down into chunks, and pass every chunk through the `GraphCodeBERT` AI model to generate vectors. Depending on how big the repository is, this could take anywhere from 1 minute to 30+ minutes on a local computer.

## Step 4: Restart the App
Once the database builder finishes, your new code is locked and loaded. Just start the Streamlit UI again:

```bash
streamlit run app/streamlit_app.py
```

When the webpage opens, any question you ask will now be searching through the new repository you just downloaded!
