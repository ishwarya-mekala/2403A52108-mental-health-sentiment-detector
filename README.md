## Mental Health Sentiment Detection (TF-IDF + Logistic Regression)

This project is a small NLP web application that detects the sentiment of mental-health related text as **positive**, **neutral**, or **negative** using a **TF-IDF** text representation and a **Logistic Regression** classifier.

### How to run locally

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser (typically at `http://localhost:8501`).

### Deployment (Streamlit Cloud)

1. Push this folder to a public GitHub repository.
2. Go to Streamlit Cloud and create a new app linked to your repo.
3. Set the entry point to `app.py` and ensure `requirements.txt` is detected.
4. Deploy; the app URL will look like `https://your-app-name.streamlit.app`.

