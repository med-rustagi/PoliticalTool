# Text analysis: how to set up and run

Put the contents of this package in the root of the PoliticalTool repo
(so you have `src/text/`, `tests/`, and `requirements-text.txt`).
Rename `gitignore.txt` to `.gitignore` (with the dot) and use it to replace the current `gitignore`.
Delete the old extensionless files `clean-text`, `data-loader`, `extract-embeddings`: the same code,
correctly named, is in `src/text/`.

## 1. Install
    pip3 install -r requirements-text.txt
(Only numpy, pandas, scikit-learn, joblib are needed for everything except embeddings.)

## 2. Check that everything works (fake data, takes seconds)
    python3 tests/test_text_pipeline.py
You should see "9 of 9 tests passed".

## 3. Try it on fake demo data
    python3 src/text/make_demo_data.py --out demo_data/demo_posts_FAKE.csv
    python3 src/text/clean_text.py --input demo_data/demo_posts_FAKE.csv --id-col shortcode --text-col caption --label-col stance --group-col username --out-dir clean/
    python3 src/text/baseline_tfidf.py
    python3 src/text/evaluate.py --pred results/predictions_tfidf.csv
    python3 src/text/text_tool.py --text "healthcare and climate action for workers"

## 4. Run on your real data
Your raw file needs, per post: an ID, the caption text, a label (e.g. left/right), and ideally the
account name. Then use the same commands as step 3 with your file and column names.

## 5. Optional: embedding model (needs internet, torch, transformers)
    python3 src/text/extract_text_embeddings.py --input clean/clean_text.csv --out-dir embeddings/ --limit 50
    python3 src/text/extract_text_embeddings.py --input clean/clean_text.csv --out-dir embeddings/
    python3 src/text/train_classifier.py
    python3 src/text/evaluate.py --pred results/predictions_tfidf.csv results/predictions_embedding_logreg.csv
This part has not been run on real data yet.

## 6. Choose the "unclear" cutoff
`evaluate.py` prints a table: for each confidence cutoff, how many posts the tool would still answer and
how accurate it is on those. Pick a cutoff you are comfortable with and pass it to the tool:
    python3 src/text/text_tool.py --text "..." --min-confidence 0.7

## Using the tool from a backend
    import sys; sys.path.insert(0, "src/text")
    from text_tool import analyze_text
    analyze_text("some caption")     # dict: label or "unclear", confidence, probabilities, top_words, reason

## Reading results responsibly
- If labels come from the ACCOUNT (e.g. party), a model can score well by recognizing account style,
  hashtags, or names. Check explain.py output and the per-account results in evaluate.py.
- Always compare with the baseline and with "always guess the most common label".
- Results are model estimates, not verdicts.
