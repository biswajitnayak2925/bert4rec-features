import os
import copy
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from model import BERT4Rec
from dataset import TrainDataset, EvalDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if cfg.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device, top_k=None):
    top_k = cfg.TOP_K if top_k is None else top_k
    model.eval(); recalls, ndcgs = [], []
    for batch in loader:
        x = batch["input_ids"].to(device)
        candidates = batch["candidates"].to(device)
        logits = model.project(model.encode(x)[:, -1, :])
        scores = torch.gather(logits, 1, candidates)
        ranks = (scores.argsort(dim=1, descending=True) == 0).nonzero()[:, 1]
        recalls.extend((ranks < top_k).float().cpu().tolist())
        ndcgs.extend(torch.where(ranks < top_k,
                                  1.0 / torch.log2(ranks.float() + 2.0),
                                  torch.zeros_like(ranks, dtype=torch.float)).cpu().tolist())
    return float(np.mean(recalls)), float(np.mean(ndcgs))


@torch.no_grad()
def evaluate_full_ranking(model, loader, device, top_k=None):
    top_k = cfg.TOP_K if top_k is None else top_k
    model.eval(); dataset = loader.dataset
    recalls, ndcgs = [], []; row_idx = 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        candidates = batch["candidates"].to(device)
        targets = candidates[:, 0]
        logits = model.project(model.encode(x)[:, -1, :])
        logits[:, 0] = float("-inf")
        for i in range(x.size(0)):
            uid = dataset.uids[row_idx]; row_idx += 1
            exclude = dataset.interacted[uid] - {int(targets[i].item())}
            if exclude:
                idx = torch.tensor(list(exclude), device=device, dtype=torch.long)
                logits[i, idx] = float("-inf")
        target_scores = torch.gather(logits, 1, targets[:, None])
        ranks = (logits > target_scores).sum(dim=1)
        recalls.extend((ranks < top_k).float().cpu().tolist())
        ndcgs.extend(torch.where(ranks < top_k,
                                  1.0 / torch.log2(ranks.float() + 2.0),
                                  torch.zeros_like(ranks, dtype=torch.float)).cpu().tolist())
    return float(np.mean(recalls)), float(np.mean(ndcgs))


def _inprogress_path(path):
    if path is None: return None
    root, ext = os.path.splitext(path)
    return f"{root}_inprogress{ext}"


def train_one_config(variant, split, num_items, mask_token, pad_token,
                     genre_mat=None, meta=None, plot_mat=None, verbose=True,
                     checkpoint_path=None, resume=True, max_seq_len=None, seed=None):
    seed = cfg.SEEDS[0] if seed is None else seed
    seed_everything(seed)
    device = get_device()
    effective_max_seq_len = cfg.MAX_SEQ_LEN if max_seq_len is None else max_seq_len

    train_ds = TrainDataset(split, num_items, mask_token, pad_token, effective_max_seq_len)
    val_ds = EvalDataset(split, num_items, mask_token, pad_token, "val", effective_max_seq_len, seed=seed + 1)
    test_ds = EvalDataset(split, num_items, mask_token, pad_token, "test", effective_max_seq_len, seed=seed + 2)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    model = BERT4Rec(num_items, mask_token, pad_token, variant, genre_mat, meta, plot_mat,
                     max_seq_len=effective_max_seq_len).to(device)
    total_params, trainable_params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_recall = -1.0
    best_val_ndcg = -1.0
    best_epoch = 0
    best_state = None
    patience_left = cfg.PATIENCE
    history = []

    inprogress_path = _inprogress_path(checkpoint_path)
    start_epoch = 1
    if resume and inprogress_path and os.path.exists(inprogress_path):
        try:
            ckpt = torch.load(inprogress_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_recall = ckpt["best_val_recall"]
            best_val_ndcg = ckpt["best_val_ndcg"]
            best_epoch = ckpt["best_epoch"]
            best_state = ckpt["best_state"]
            patience_left = ckpt["patience_left"]
            history = ckpt["history"]
            if verbose:
                print(f"  [{variant} seed={seed}] found in-progress checkpoint -> "
                      f"resuming from epoch {start_epoch} (best_val_recall so far = {best_val_recall:.4f})")
        except Exception as e:
            if verbose:
                print(f"  [{variant} seed={seed}] WARNING: in-progress checkpoint at "
                      f"{inprogress_path} could not be loaded ({type(e).__name__}: {e}). "
                      f"Starting this run from epoch 1 instead.")
            try:
                os.remove(inprogress_path)
            except OSError:
                pass
            start_epoch = 1
            best_val_recall = -1.0
            best_val_ndcg = -1.0
            best_epoch = 0
            best_state = None
            patience_left = cfg.PATIENCE
            history = []

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        model.train(); total_loss = 0.0; n_batches = 0
        for batch in train_loader:
            x = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                hidden = model.encode(x)
                mask = labels != -100
                selected_logits = model.project(hidden[mask])
                loss = nn.functional.cross_entropy(selected_logits, labels[mask])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        row = {"epoch": epoch, "train_loss": train_loss,
               "val_recall": np.nan, "val_ndcg": np.nan}
        stop = False
        if epoch % cfg.EVAL_EVERY == 0:
            val_recall, val_ndcg = evaluate(model, val_loader, device)
            row.update(val_recall=val_recall, val_ndcg=val_ndcg)
            improved = val_recall > best_val_recall
            if improved:
                best_val_recall, best_val_ndcg = val_recall, val_ndcg
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = cfg.PATIENCE
            else:
                patience_left -= 1
                stop = patience_left <= 0
            if verbose:
                print(f"  [{variant} seed={seed}] epoch {epoch:3d} | loss={train_loss:.4f} | "
                      f"val R@{cfg.TOP_K}={val_recall:.4f} | val NDCG={val_ndcg:.4f} | "
                      f"patience={patience_left}")
        history.append(row)

        if checkpoint_path and epoch % cfg.EVAL_EVERY == 0:
            os.makedirs(os.path.dirname(inprogress_path), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "best_val_recall": best_val_recall,
                        "best_val_ndcg": best_val_ndcg, "best_epoch": best_epoch,
                        "best_state": best_state, "patience_left": patience_left,
                        "history": history}, inprogress_path)
        if stop:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
    model.load_state_dict(best_state)

    sampled_r, sampled_n = evaluate(model, test_loader, device)
    if verbose: print(f"  [{variant} seed={seed}] full-catalog test ranking...")
    full_r, full_n = evaluate_full_ranking(model, test_loader, device)

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "variant": variant,
                    "num_items": num_items, "mask_token": mask_token, "pad_token": pad_token,
                    "seed": seed, "best_epoch": best_epoch,
                    "best_val_recall": best_val_recall, "best_val_ndcg": best_val_ndcg,
                    "total_params": total_params, "trainable_params": trainable_params,
                    "max_seq_len": effective_max_seq_len}, checkpoint_path)
        history_path = os.path.splitext(checkpoint_path)[0] + "_history.csv"
        import pandas as pd
        pd.DataFrame(history).to_csv(history_path, index=False)
        if inprogress_path and os.path.exists(inprogress_path): os.remove(inprogress_path)

    return {"variant": variant, "seed": seed,
            "best_epoch": best_epoch, "best_val_recall": best_val_recall,
            "best_val_ndcg": best_val_ndcg, "total_params": total_params,
            "trainable_params": trainable_params,
            f"test_recall@{cfg.TOP_K}_sampled": sampled_r,
            f"test_ndcg@{cfg.TOP_K}_sampled": sampled_n,
            f"test_recall@{cfg.TOP_K}_full": full_r,
            f"test_ndcg@{cfg.TOP_K}_full": full_n,
            "max_seq_len_used": effective_max_seq_len}
