"""
app.py — Hugging Face Spaces Entry Point
Runs the complete Streamlit application on Port 7860.
Compatible with Hugging Face Spaces Free Tier (Gradio SDK).
"""
import sys
import os
import subprocess

def prepare_environment():
    """Pre-download required NLTK datasets if missing."""
    try:
        import nltk
        nltk_pkgs = [
            "punkt",
            "punkt_tab",
            "stopwords",
            "wordnet",
            "averaged_perceptron_tagger",
            "vader_lexicon",
        ]
        for pkg in nltk_pkgs:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    except Exception as e:
        print(f"NLTK setup notice: {e}")

if __name__ == "__main__":
    prepare_environment()

    port = os.environ.get("PORT", "7860")
    print(f"Starting WhatsApp Analytics Streamlit app on port {port}...")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
    ]

    subprocess.run(cmd)
