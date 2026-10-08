# ChiEAC Grant Intelligence - free hosted demo
Independent fellowship prototype for feedback. No organizational endorsement implied.

## Deploy
Upload this folder's contents to a GitHub repository. In Streamlit Community Cloud select that repository, branch main, entrypoint app.py and Python 3.11. No secrets or billing information are needed for this demo.
Do not upload the parent desktop project or reviewer data.

## Run
pip install -r requirements.txt
streamlit run app.py

## Features and limits
Live TF-IDF keyword search; actual precomputed nomic-embed-text scores for three fixed program descriptions; 14 source-linked research records; HTML brief exports; session-only reviewer feedback; a clearly labeled recorded Llama extraction example.
Neural inference and Llama extraction do not run on the hosted server. No paid API calls, live imports, shared writes or persistent reviewer store.
Source snapshots are from October 7, 2026 and may become stale. Similarity does not determine eligibility. Benchmark labels are provisional assistant-authored judgments.

Review feedback is lost when a session ends unless downloaded. The app does not authenticate reviewer identity. The full local app remains separate.
