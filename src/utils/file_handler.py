import os
import shutil
from pathlib import Path

def save_uploaded_file(uploaded_file, destination_folder: str) -> str:
    """
    Save uploaded file to destination folder.

    Args:
        uploaded_file: File object from Streamlit
        destination_folder: Where to save the file

    Returns:
        Path to saved file
    """
    os.makedirs(destination_folder, exist_ok=True)
    file_path = os.path.join(destination_folder, uploaded_file.name)

    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def get_all_chat_files(folder: str) -> list:
    """Get all .txt files from a folder"""
    if not os.path.exists(folder):
        return []

    return [f for f in os.listdir(folder) if f.endswith('.txt')]

def clean_old_files(folder: str, keep_latest: int = 5):
    """Keep only latest N files in folder"""
    if not os.path.exists(folder):
        return

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
    files.sort(key=os.path.getctime, reverse=True)

    for old_file in files[keep_latest:]:
        os.remove(old_file)
