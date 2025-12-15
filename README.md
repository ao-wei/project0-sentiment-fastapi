# Project 0: Sentiment Classification + FastAPI

A from-scratch NLP engineering mini project:

Text -> fine-tune a sentiment model -> save/load -> FastAPI inference service -> Streamlit web demo.

## 0. Prerequisites

- macOS / Linux / Windows
- Conda (Miniconda/Anaconda)
- Git
- Internet access (first run will download the IMDb dataset + pretrained model)

## 1. Setup (Conda)

Create environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate proj0-nlp
```

If you already have an environment and want to install deps into it:

```bash
conda activate proj0-nlp
pip install -U pip
pip install -r requirements.txt
```

## 2. Train the model
This will fine-tune distilroberta-base on IMDb and save the model to:

- models/imdb-distilroberta/

Run:
```bash
conda activate proj0-nlp
python train.py
```

Notes:

-  The models/ directory is ignored by git (large files). You must train once to generate it.
-  First training run may take longer due to downloads and cache building.

## 3. Local inference(CLI)

After training, test inference:

```bash
conda activate proj0-nlp
python inference.py --text "This movie was fantastic. I loved it."
python inference.py --text "This film was boring and a waste of time."
```

Or interactive mode:

```bash
python inference.py
```

## 4.Run FastAPI service

Start the API server:

```bash
conda activate proj0-nlp
uvicorn app.main:app --reload
```

Open:

- Swagger docs: http://127.0.0.1:8000/docs

- Health check: http://127.0.0.1:8000/

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I really enjoyed this movie."}'
```

## 5. Run Streamlit demo

Keep FastAPI running in Terminal A, then in Terminal B:

```bash
conda activate proj0-nlp
streamlit run frontend/streamlit_app.py
```

Open the UI (Streamlit will print a URL, usually):

- http://localhost:8501

The Streamlit app calls FastAPI /predict and displays label, score, and probabilities.

## 6. Project structure
```text
project0-sentiment-fastapi/
├── app/
│   └── main.py                # FastAPI service (POST /predict)
├── frontend/
│   └── streamlit_app.py       # Streamlit UI demo
├── train.py                   # training script (Trainer)
├── inference.py               # load model + predict() + CLI
├── models/                    # saved model (ignored by git)
├── data/                      # optional data cache (ignored by git)
├── environment.yml            # conda environment spec
└── requirements.txt           # optional pip dependency list
```

## 7.Common issues

- Cannot reach FastAPI from Streamlit:
  - Make sure FastAPI is running: uvicorn app.main:app --reload
  - Check the base URL in Streamlit sidebar is http://127.0.0.1:8000
- First run downloads are slow / network issues:
  - Retry, or check your network/proxy settings.
- Apple Silicon acceleration:
  - On macOS, PyTorch may use MPS if available. The code will print the device during inference.
