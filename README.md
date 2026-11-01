# Archaeological retrieval models

This repository contains scripts to fine-tune a BGE dense retriever, train a BERT backbone for a ColBERT adaptation experiment, and compare the resulting models with lexical and neural retrieval baselines. It also provides the evaluation results and an interactive explorer of the top-10 rankings.

## Published models and research data

The trained retrieval model is available on Hugging Face:

- **Domain-adapted BGE:** [smanousopoulos/archaeo-bge](https://huggingface.co/smanousopoulos/archaeo-bge)

The source code and associated research data, including training pairs, evaluation queries, document descriptions, and relevance judgments, are archived on [Zenodo](https://doi.org/10.5281/zenodo.23671089).

For details of the training procedures, experimental design, and interpretation of the results, see the accompanying research paper.

## Evaluation results

Six retrieval configurations were evaluated using **34 archaeological queries**, **160 document descriptions**, and graded relevance judgments (**0 = irrelevant**, **1 = partially relevant**, **2 = highly relevant**).

The configurations comprise BM25, pretrained and domain-adapted ColBERT, pretrained and domain-adapted BGE, and an mxbai-rerank cross-encoder.

### Interactive top-10 ranking explorer

**[Explore the top-10 rankings](docs/index.html)** (open the HTML file locally, or publish `docs/` using GitHub Pages).

The explorer allows readers to select an evaluation query, compare the six models' top-10 rankings, inspect relevance grades and abbreviated document descriptions, and examine per-query performance. It is a static HTML page and does not require a Python server.

> After enabling GitHub Pages under **Settings → Pages**, replace this relative link with the live Pages URL. Links to `docs/index.html` on GitHub show the source rather than running the interactive page.

### Overall retrieval effectiveness

| Model | P@10 | R@10 | F1@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.200 | 0.240 | 0.186 | 0.279 |
| ColBERT pretrained | 0.318 | 0.361 | 0.291 | 0.420 |
| ColBERT domain-adapted | 0.429 | 0.507 | 0.402 | 0.589 |
| Cross-encoder (mxbai-rerank) | 0.453 | 0.544 | 0.430 | 0.623 |
| BGE pretrained | 0.379 | 0.438 | 0.350 | 0.504 |
| **BGE domain-adapted** | **0.471** | **0.560** | **0.442** | **0.627** |

### Downloadable results

- [Overall evaluation metrics](results/overall_metrics.csv)
- [Results by query category](results/category_metrics.csv)
- [Complete per-query metrics](results/per_query_metrics.csv)
- [Top-10 rankings and relevance grades](results/top10_rankings.csv)

The per-query file reports Precision@10, Recall@10, F1@10, and nDCG@10 for each query-model combination. The top-10 export contains individual ranked results, model scores, and gold relevance grades. Document excerpts are **truncated** in the original evaluation log; the exported ranking file does not replace the complete evaluation document collection.

These results should be interpreted as exploratory, given the relatively small, researcher-curated evaluation collection. The Zenodo record provides the versioned research archive. Ensure the published Zenodo release includes the result files linked here before citing it as their location.

## Quick Start: Semantic Search

Use the published [domain-adapted BGE model](https://huggingface.co/smanousopoulos/archaeo-bge) to rank archaeological document descriptions against a natural-language query.

No training dataset or GPU is required for this example. An internet connection is needed the first time the model is downloaded.

### Install dependencies

```bash
pip install sentence-transformers numpy
```

### Run an example search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "smanousopoulos/archaeo-bge"
model = SentenceTransformer(MODEL_NAME)

query = "prehistoric drinking vessel"

documents = [
    "A Late Helladic kylix with a shallow bowl and two handles.",
    "A Roman bronze coin depicting the emperor Justinian on the obverse.",
    "Palestinian amphora with broad, bag-shaped body and short collar rim with plain rounded lip.",
    "A Mycenaean goblet dating to the 12th century BCE.",
    "A Classical black-glazed kantharos",
]

query_embedding = model.encode(query, normalize_embeddings=True)
document_embeddings = model.encode(documents, normalize_embeddings=True)

# Cosine similarity for normalized vectors.
scores = document_embeddings @ query_embedding
ranked_indices = np.argsort(scores)[::-1]

print(f"\nQuery: {query}\n")
print("Ranked results:\n")

for rank, index in enumerate(ranked_indices, start=1):
    print(f"{rank}. Score: {scores[index]:.4f}")
    print(f"   {documents[index]}\n")
```

The script downloads the model from Hugging Face, encodes one query and **five** example document descriptions, and ranks them by cosine similarity. To try your own examples, change the `query` and `documents` variables, or save this snippet as `example_search.py`.

**Note:** Similarity scores indicate relative semantic similarity, not probabilities of relevance.

## Installation

Use a Python environment with a compatible PyTorch/CUDA installation if you plan to train or evaluate models on a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if the default installation is not appropriate for your system. The training scripts use `fp16=True`, and model comparison defaults to CUDA.

## Input files

Run commands from the repository root. The training scripts expect:

```text
data/
├── general_pairs.csv       # General positive query-document pairs
└── dataset_pairs.csv       # Archaeological positive query-document pairs
```

Each training CSV must contain `query` and `document` columns. Additional original metadata columns may be retained.

## Train the BGE model

```bash
python supervised-learn-bge-2-stage.py
```

The script starts from `BAAI/bge-large-en-v1.5` and trains in two stages:

1. General pairs (`data/general_pairs.csv`)
2. Archaeological dataset pairs (`data/dataset_pairs.csv`)

The final model is saved to:

```text
bge_results/bge_large_stage2/
```

## Train the backbone for the ColBERT experiment

```bash
python supervised-learn-colbert-transplant-2-stage.py
```

**Important:** This script fine-tunes a Sentence Transformers BERT encoder for **backbone transplantation into ColBERT**. It does **not** train the complete ColBERT model end-to-end.

The script starts from `bert-base-uncased` and uses the opposite training order to the BGE experiment:

1. Archaeological dataset pairs (`data/dataset_pairs.csv`)
2. General pairs (`data/general_pairs.csv`)

The final stage model is saved to:

```text
colbert_transplant_two_stage_runs/dataset_then_general/stage2/
```

During comparison, `compare-models.py` loads the pretrained `colbert-ir/colbertv2.0` checkpoint and attempts to load the adapted BERT backbone into it. Inspect the printed missing/unexpected weight keys when reproducing this adaptation.

## Evaluating the models

Provide the separately distributed evaluation CSV at:

```text
data/evaluation/evaluation_data.csv
```

Expected columns:

```text
query_id,query,category,doc,relevance
```

The evaluation CSV and trained model checkpoints are **not included** in this source package. Complete the training steps above, or modify the paths in the `models_to_test` dictionary in `compare-models.py` to point to your checkpoints.

Run:

```bash
python compare-models.py
```

The comparison script evaluates BM25, original and adapted BGE, original and adapted ColBERT, and a cross-encoder baseline. Its generated outputs are:

```text
compare_results/per_query_model_results.csv
compare_results/overall_summary.csv
compare_results/category_summary.csv
compare_results/query_counts_by_category.csv
```

The `results/` directory contains the curated results accompanying the paper, whereas `compare_results/` holds outputs generated by an evaluation run. Do not assume the two directories reflect identical experiments unless their model revisions and settings match.

## Repository layout

```text
.
├── README.md
├── example_search.py
├── requirements.txt
├── compare-models.py
├── supervised-learn-bge-2-stage.py
├── supervised-learn-colbert-transplant-2-stage.py
├── docs/
│   └── index.html                # Interactive ranking explorer
├── results/
│   ├── overall_metrics.csv
│   ├── category_metrics.csv
│   ├── per_query_metrics.csv
│   └── top10_rankings.csv
├── scripts/
│   └── parse_results_log.py     # Optional log-to-results converter
└── data/
    └── evaluation/
        └── evaluation_data.csv  # Separately distributed
```

To publish the interactive explorer, enable **GitHub Pages** for the repository, choosing **Deploy from a branch** → your default branch → **`/docs`**. Once the site is live, use its URL in the explorer link above.

## Reproducibility notes

- Training uses `MultipleNegativesRankingLoss` and validation-based checkpoint selection.
- The training scripts currently split training and validation **by row**. Documents with multiple query variants can therefore appear in both sets. A document-grouped split is preferable for document-disjoint validation.
- Both training scripts define a custom document-unique batch sampler. This does not eliminate all semantic false negatives.
- Check the BM25 handling of zero/nonpositive scores in the exact revision of `compare-models.py` used for the published evaluation. Changes to how zero-score results are retained can affect comparisons.
- Record Python and dependency versions, GPU/CUDA configuration, random seeds, and pretrained model revisions for exact reproduction.
- The training scripts were inspected but not run through full GPU training as part of the package review.
- The interactive explorer displays excerpts from console output rather than full document texts. Use the released evaluation dataset for complete descriptions and relevance judgments.

## License

The source code in this repository is released under the Apache License 2.0.

Datasets and model checkpoints may be subject to separate licenses, as specified in their respective directories or model cards. Third-party archaeological texts and pretrained model components remain subject to their original licenses and applicable rights.

See the [LICENSE](LICENSE) file for details.