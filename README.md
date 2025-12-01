# Chatbot_using_NLP
implementation of chatbot using NLP

Quick setup
1. Create virtual environment and activate (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Download NLTK punkt once (optional if not present):

```powershell
python -m nltk.downloader punkt
```

4. Run the Streamlit UI:

```powershell
streamlit run chatbot_embeddings_ui.py
```

Notes:
- Do not commit the `.venv/` directory. Use `.gitignore` to exclude it.
- If pushing to a remote, pull/rebase remote changes first: `git pull --rebase origin main`.
