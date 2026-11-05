"""Shared default settings. Every script imports its default file paths from here,
so you can change a path in one place. Command-line options still override them."""

SEED = 42

CLEAN_CSV = "clean/clean_text.csv"    # output of clean_text.py
EMB_DIR = "embeddings"                # output of extract_text_embeddings.py
MODEL_DIR = "models"                  # trained models are saved here
RESULTS_DIR = "results"               # predictions and evaluation reports are saved here
