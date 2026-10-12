# bert4rec-features

Code for the paper **"When Does Item Content Help? A Multi-Seed Study of Genre, Metadata and Plot Features in BERT4Rec Across Training-History Retention Levels"** (Biswajit Nayak).

The study trains five BERT4Rec variants that share one Transformer backbone and differ only in the item input representation, at four history-retention levels, and evaluates them with full-catalogue Recall@10 and NDCG@10.

| Variant | Item representation |
| --- | --- |
| `M1_id` | ID embedding only |
| `M2_genre` | ID + genre feature |
| `M3_metadata` | ID + director, lead cast, runtime |
| `M4_plot` | ID + plot embedding (all-MiniLM-L6-v2, 384-d, frozen) |
| `M5_all` | ID + genre + metadata + plot |

Retention levels: 100%, 50%, 25%, 10% of the most recent 20 training interactions per user (20, 10, 5, 2 items).

## 1. Repository layout

| File | Purpose |
| --- | --- |
| `metadata_fetch.py` | Retrieves director and lead cast for every movie from the TMDB API |
| `build_dataset.py` | Builds the five CSV files read by the training code from the raw sources |
| `config.py` | All paths and hyperparameters, including `SEEDS` and `SPARSITY_LEVELS` |
| `data_utils.py` | Loading, item vocabulary, leave-one-out split, retention (sparsity) protocol |
| `features.py` | Genre, metadata and plot feature encoders |
| `dataset.py` | Masked-item training dataset and evaluation datasets |
| `model.py` | BERT4Rec with the content-fusion item encoder |
| `train.py` | Training loop, early stopping, sampled and full-catalogue evaluation |
| `run_all.py` | Runs every variant x retention x seed and writes the result tables |
| `analyze_results.py` | Recomputes means, SDs, paired t-tests, Holm correction and Friedman tests |
| `plot_curves.py` | Training and validation curves for the 100% retention runs |

## 2. Environment

Python 3.9+ and a CUDA GPU are expected (`DEVICE = "cuda"` in `config.py`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`sentence-transformers` is **required** to reproduce the paper. If it is missing, `features.py` silently falls back to a TF-IDF + SVD plot encoder and the M4 and M5 results will not match. After starting a run, check that `outputs_publication/experiment_metadata.txt` and the console line `Plot encoder actually used:` mention the sentence encoder (`all-MiniLM-L6-v2`).

## 3. Data

Three public sources are used. **The combined data is not redistributed** (MovieLens and TMDB have their own terms), so you rebuild it locally.

1. **MovieLens-25M**: download from GroupLens, https://grouplens.org/datasets/movielens/25m/, and unzip. You need `ratings.csv` and `links.csv`.
2. **TMDB Movies Dataset 2023 (Kaggle)**: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies. You need its main CSV (the file name varies between versions, for example `TMDB_movie_dataset_v11.csv`). Columns used: `id`, `title`, `runtime`, `overview`, `genres`.
3. **TMDB API key** (free account at https://www.themoviedb.org): used to fetch director and lead cast.

### 3.1 Fetch director and lead cast

```bash
export TMDB_API_KEY=your_v3_api_key        # Windows PowerShell: $env:TMDB_API_KEY="..."
python metadata_fetch.py --links ml-25m/links.csv --out link2.csv
```

- `director` = first crew member whose job is "Director".
- `lead` = the two top-billed cast members joined as `"Actor A, Actor B"` (one categorical value).
- The script resumes from an existing `link2.csv`. It makes roughly 60,000 requests.
- Never commit your API key. Alternatively set `TMDB_BEARER_TOKEN` (v4 read token).

### 3.2 Build the five CSV files

```bash
python build_dataset.py --ratings ml-25m/ratings.csv --links2 link2.csv \
    --tmdb path/to/TMDB_movie_dataset.csv --out data
```

Rules: join on the TMDB id; drop every movie with a null title, genres, overview, runtime, director or lead; keep movies present in both ratings and the cleaned movie table; drop ratings of removed movies.

Output in `data/`: `Final_Ratings.csv`, `model_id.csv`, `model_genre.csv`, `model_metadata.csv`, `model_plot.csv`.

**Expected counts** (printed at the end of the script and by `python data_utils.py`):

| Quantity | Value |
| --- | --- |
| Users | 162,541 |
| Items | 55,388 |
| Interactions | 24,925,850 |
| Density | 0.2769% |

If your counts differ, the item and user numbers in the paper will not apply to your run. TMDB content changes over time, so small differences are possible if you retrieve it at a later date.

Sanity check of the data pipeline:

```bash
python data_utils.py
```

## 4. Running the experiments

`config.py` sets `DATA_DIR = "./data"` and `OUTPUT_DIR = "./outputs_publication"`. The paper uses:

- retention 10%, 25%, 50%: seeds **42, 62, 82**
- retention 100%: seed **42 only** (limited GPU compute)

`run_all.py` resumes: finished (variant, retention, seed) runs listed in `results_by_seed.csv` are skipped, and interrupted runs continue from their checkpoint. So the paper's setting is two passes over the same `OUTPUT_DIR`.

**Pass 1** (edit `config.py`):

```python
SEEDS = [42]
SPARSITY_LEVELS = [1.0, 0.50, 0.25, 0.10]
```

```bash
python run_all.py
```

**Pass 2** (edit `config.py`, same `OUTPUT_DIR`):

```python
SEEDS = [42, 62, 82]
SPARSITY_LEVELS = [0.50, 0.25, 0.10]
```

```bash
python run_all.py
```

Seed 42 runs from pass 1 are skipped, so pass 2 only trains seeds 62 and 82 at the three lower retention levels.

Compute: individual runs took about 0.9 to 2.3 hours at 10 to 50% retention on a single GPU, and 100% runs are longer, so the full set (50 runs) is on the order of 80 GPU-hours. To check the setup first, use `SPARSITY_LEVELS = [0.10]`, `SEEDS = [42]` and a small `NUM_EPOCHS`.

Important settings (already the paper's values in `config.py`): hidden size 128, 2 layers, 4 heads, feed-forward 256, dropout 0.2, mask probability 0.2, batch size 128, Adam with learning rate 1e-3, up to 200 epochs, validation every 4 epochs, patience 20 checks, 100 sampled negatives for validation.

### Outputs (in `outputs_publication/`)

| File | Content |
| --- | --- |
| `results_by_seed.csv` | One row per run: full and sampled Recall@10 and NDCG@10, best epoch, parameters, time |
| `results_summary.csv` | Mean and SD per variant and retention level |
| `paper_table_full_catalog.csv` | Full-catalogue Recall@10 and NDCG@10, mean and SD |
| `experiment_metadata.txt` | Epochs, patience, seeds, plot encoder actually used |
| `checkpoints/` | Model checkpoints and per-run training histories |

## 5. Statistics and figures

```bash
python analyze_results.py outputs_publication/results_by_seed.csv --out analysis
```

Prints, per retention level: mean and SD, relative change versus `M1_id`, per-seed winner, paired t-tests versus `M1_id` (nominal and Holm-corrected) and the Friedman test. With three seeds the tests have only 2 degrees of freedom, so p-values are nominal; see the paper for the interpretation.

```bash
python plot_curves.py          # training curves of the 100% retention runs (seed 42)
```

## 6. Expected results

Full-catalogue Recall@10 (mean ± SD over seeds 42, 62, 82; 100%: seed 42 only). Small differences are expected from GPU non-determinism and library versions.

| Variant | 10% | 25% | 50% | 100% (1 seed) |
| --- | --- | --- | --- | --- |
| M1 ID | 0.0728 ± 0.0098 | 0.1199 ± 0.0017 | 0.1629 ± 0.0024 | 0.1879 |
| M2 genre | 0.0657 ± 0.0068 | 0.1326 ± 0.0016 | 0.1642 ± 0.0007 | 0.1885 |
| M3 metadata | 0.0704 ± 0.0130 | 0.1286 ± 0.0053 | 0.1611 ± 0.0014 | 0.1878 |
| M4 plot | 0.0673 ± 0.0024 | 0.1290 ± 0.0017 | 0.1639 ± 0.0019 | 0.1866 |
| M5 all | 0.0690 ± 0.0021 | 0.1286 ± 0.0059 | 0.1637 ± 0.0021 | 0.1882 |

## 7. Notes that affect reproduction

- **Genre feature.** `features.py` splits the genre string on `|`, but the genres in the data are comma-separated (for example `"Action, Adventure"`). Each distinct genre combination therefore becomes one category (5,028 in the paper), so the genre feature is a genre-combination identifier, not a multi-hot vector over single genres. The paper's results use this behaviour. Changing the separator changes the M2 and M5 results.
- **Lead cast** is likewise a pair of two actors treated as one categorical value.
- **Rating values are ignored.** Only `userId`, `movieId` and `timestamp` are read; every rating is an implicit interaction.
- **Retention** keeps the most recent items of the 20-item training window, and the reduced history is also the context at validation and test time.
- **Seeds** change initialisation, masking, batch order and sampled negatives, not the data split (leave-one-out is deterministic).
- **Determinism:** the code requests deterministic PyTorch algorithms (warnings only) and uses mixed precision, so small GPU-level differences can remain.
- **Disk and memory:** `Final_Ratings.csv` is about 650 MB.

## 8. Citation

```bibtex
@misc{nayak2026itemcontent,
  title  = {When Does Item Content Help? A Multi-Seed Study of Genre, Metadata and Plot Features in BERT4Rec Across Training-History Retention Levels},
  author = {Nayak, Biswajit},
  year   = {2026},
  note   = {arXiv preprint, identifier to be added}
}
```

## 9. Data attribution and licence

- MovieLens-25M: F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015. Follow the GroupLens usage terms.
- TMDB Movies Dataset 2023 by asaniczka (Kaggle).
- This product uses the TMDB API but is not endorsed or certified by TMDB.
- Do not commit raw data, retrieved TMDB content or API keys to this repository.
- Code licence: MIT (see `LICENSE`).
