import numpy as np

LAST_PLOT_ENCODER_INFO = "unknown"


def encode_genres(feat_table, num_items):
    all_genres = set()
    for g in feat_table["genres"].dropna():
        for tok in str(g).split("|"):
            tok = tok.strip()
            if tok:
                all_genres.add(tok)
    genre_list = sorted(all_genres)
    genre2idx = {g: i for i, g in enumerate(genre_list)}
    num_genres = len(genre_list)


    mat = np.zeros((num_items + 2, num_genres), dtype=np.float32)
    for item_idx, row in feat_table.iterrows():
        g = row.get("genres")
        if isinstance(g, str) and g.strip():
            for tok in g.split("|"):
                tok = tok.strip()
                if tok in genre2idx:
                    mat[item_idx, genre2idx[tok]] = 1.0
    return mat, genre2idx


def encode_metadata(feat_table, num_items):
    directors = feat_table["director"].fillna("__UNK__").astype(str) if "director" in feat_table else None
    leads = feat_table["lead"].fillna("__UNK__").astype(str) if "lead" in feat_table else None
    runtimes = feat_table["runtime"] if "runtime" in feat_table else None

    def build_vocab(series):
        vals = sorted(series.unique().tolist())
        v2i = {v: i + 1 for i, v in enumerate(vals)}  
        v2i["__UNK__"] = v2i.get("__UNK__", 0)
        return v2i

    director2idx = build_vocab(directors) if directors is not None else {"__UNK__": 0}
    lead2idx = build_vocab(leads) if leads is not None else {"__UNK__": 0}

    director_ids = np.zeros(num_items + 2, dtype=np.int64)
    lead_ids = np.zeros(num_items + 2, dtype=np.int64)
    runtime_norm = np.zeros(num_items + 2, dtype=np.float32)

    if runtimes is not None:
        rt = runtimes.astype(float)
        mean, std = np.nanmean(rt), np.nanstd(rt) + 1e-6
    else:
        mean, std = 0.0, 1.0

    for item_idx, row in feat_table.iterrows():
        if directors is not None:
            director_ids[item_idx] = director2idx.get(row.get("director", "__UNK__"), 0)
        if leads is not None:
            lead_ids[item_idx] = lead2idx.get(row.get("lead", "__UNK__"), 0)
        if runtimes is not None and not np.isnan(row.get("runtime", np.nan)):
            runtime_norm[item_idx] = (row["runtime"] - mean) / std

    return {
        "director_ids": director_ids,
        "lead_ids": lead_ids,
        "runtime_norm": runtime_norm,
        "num_directors": len(director2idx),
        "num_leads": len(lead2idx),
    }


def encode_plot(feat_table, num_items, cache_path=None):
    global LAST_PLOT_ENCODER_INFO
    overviews = feat_table["overview"].fillna("").astype(str)
    texts_by_item = {idx: t for idx, t in overviews.items()}

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ordered_items = sorted(texts_by_item.keys())
        embs = model.encode([texts_by_item[i] for i in ordered_items],
                             show_progress_bar=False, convert_to_numpy=True)
        dim = embs.shape[1]
        mat = np.zeros((num_items + 2, dim), dtype=np.float32)
        for i, item_idx in enumerate(ordered_items):
            mat[item_idx] = embs[i]
        LAST_PLOT_ENCODER_INFO = f"sentence-transformers: all-MiniLM-L6-v2 (dim={dim})"
        print(f"[encode_plot] used {LAST_PLOT_ENCODER_INFO}")
        return mat
    except Exception as e:
        print(f"[encode_plot] sentence-transformers unavailable ({e}); "
              f"falling back to TF-IDF + SVD")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        ordered_items = sorted(texts_by_item.keys())
        corpus = [texts_by_item[i] if texts_by_item[i].strip() else "unknown" for i in ordered_items]
        tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
        X = tfidf.fit_transform(corpus)
        dim = min(128, X.shape[1] - 1, X.shape[0] - 1)
        dim = max(dim, 2)
        svd = TruncatedSVD(n_components=dim, random_state=42)
        reduced = svd.fit_transform(X).astype(np.float32)

        mat = np.zeros((num_items + 2, dim), dtype=np.float32)
        for i, item_idx in enumerate(ordered_items):
            mat[item_idx] = reduced[i]
        LAST_PLOT_ENCODER_INFO = f"TF-IDF + TruncatedSVD fallback (dim={dim}, max_features=20000, stop_words=english)"
        print(f"[encode_plot] used {LAST_PLOT_ENCODER_INFO}")
        return mat


if __name__ == "__main__":
    import data_utils as du
    ratings, ids, genre, metadata, plot = du.load_raw_tables()
    item2idx, idx2item, num_items, mask_token, pad_token = du.build_item_vocab(ratings, ids)
    feat_table = du.build_item_feature_table(ids, genre, metadata, plot, item2idx)

    genre_mat, genre2idx = encode_genres(feat_table, num_items)
    print("genre matrix:", genre_mat.shape, "num_genres:", len(genre2idx))

    meta = encode_metadata(feat_table, num_items)
    print("metadata:", {k: (v.shape if hasattr(v, "shape") else v) for k, v in meta.items()})

    plot_mat = encode_plot(feat_table, num_items)
    print("plot matrix:", plot_mat.shape)
