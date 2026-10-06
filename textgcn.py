"""Text GCN implementation based on Yao, Mao & Luo (AAAI 2019).

Commands: prepare, train, results, audit. See README.md for usage.
Includes graph construction, both models, checkpoint resume and reporting.
"""


# ========================================================================
# Common
# ========================================================================
"""Shared artifact utilities. No external packages needed for data download."""
import hashlib
import json
import os
from pathlib import Path

EXPECTED = {
    "R8": (7674, 5485, 2189, 8, 7688, 65.72),
    "R52": (9100, 6532, 2568, 52, 8892, 69.82),
    "ohsumed": (7400, 3357, 4043, 23, 14157, 135.82),
    "mr": (10662, 7108, 3554, 2, 18764, 20.39),
    "20ng": (18846, 11314, 7532, 20, 42757, 221.26),
}
PAPER = {
    "tfidf_lr": dict(zip(EXPECTED, [93.74, 86.95, 54.66, 74.59, 83.19])),
    "textgcn": dict(zip(EXPECTED, [97.07, 93.56, 68.36, 76.74, 86.34])),
}

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def load_dataset(root, name):
    p = Path(root) / "data" / name
    meta = read_json(p / "dataset.json")
    if digest(p / "clean.txt") != meta["clean_sha256"]:
        raise ValueError("Clean text changed; rerun prepare_data in a fresh directory")
    docs = (p / "clean.txt").read_text(encoding="utf-8").splitlines()
    if len(docs) != len(meta["labels"]):
        raise ValueError("Document/label alignment mismatch")
    return docs, meta


# ========================================================================
# Prepare Data
# ========================================================================
"""Download authors' cleaned corpora and original labels/splits at one commit.

Uses the supplied cleaned files to avoid changes in stop-word resources or cleaning.
Does not fit preprocessing to labels, alter the test split, or drop empty documents.
"""
import argparse
from collections import Counter
from pathlib import Path
import urllib.request


REPO = "yao8839836/text_gcn"
SOURCE_COMMIT = "962223652e9bb164ac2d83cd09fc7b8845ce860b"

def split_name(value):
    names = {"train": "train", "training": "train", "20news-bydate-train": "train",
             "test": "test", "20news-bydate-test": "test"}
    if value not in names:
        raise ValueError(f"Unknown source split: {value}")
    return names[value]

def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "TextGCN-student-replication"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()

def prepare(root, datasets):
    root = Path(root)
    provenance = root / "data" / "source.json"
    if provenance.exists():
        source = read_json(provenance)
    else:
        source = {"repository": REPO, "commit": SOURCE_COMMIT,
                  "preprocessing": "Authors' supplied .clean.txt files, used without modification"}
        write_json(provenance, source)
    for name in datasets:
        p = root / "data" / name
        p.mkdir(parents=True, exist_ok=True)
        paths = {"clean.txt": f"data/corpus/{name}.clean.txt", "metadata.txt": f"data/{name}.txt"}
        urls = {}
        for local, remote in paths.items():
            url = f"https://raw.githubusercontent.com/{REPO}/{source['commit']}/{remote}"
            urls[local] = url
            if not (p / local).exists():
                payload = fetch(url)
                tmp = p / (local + ".download")
                tmp.write_bytes(payload)
                tmp.replace(p / local)
        docs = (p / "clean.txt").read_text(encoding="utf-8").splitlines()
        rows = [line.split("\t") for line in (p / "metadata.txt").read_text(encoding="utf-8").splitlines()]
        if any(len(r) != 3 for r in rows):
            raise ValueError(f"Unexpected metadata format for {name}")
        splits = [split_name(r[1]) for r in rows]
        labels = sorted({r[2] for r in rows})
        train = [i for i, value in enumerate(splits) if value == "train"]
        test = [i for i, value in enumerate(splits) if value == "test"]
        counts = Counter(w for doc in docs for w in doc.split())
        actual = (len(docs), len(train), len(test), len(labels), len(counts))
        if len(rows) != len(docs) or actual != EXPECTED[name][:5]:
            raise ValueError(f"{name}: expected {EXPECTED[name][:5]}, found {actual}; inspect source before training")
        avg = sum(counts.values()) / len(docs)
        if abs(avg - EXPECTED[name][5]) > 0.02:
            raise ValueError(f"{name}: average length {avg} does not match paper")
        meta = {"dataset": name, "source": source, "urls": urls,
                "clean_sha256": digest(p / "clean.txt"), "metadata_sha256": digest(p / "metadata.txt"),
                "document_ids": [r[0] for r in rows], "class_names": labels,
                "labels": [labels.index(r[2]) for r in rows],
                "train_indices": train, "test_indices": test,
                "statistics": {"documents": len(docs), "original_train": len(train), "test": len(test),
                               "classes": len(labels), "vocabulary": len(counts),
                               "nodes": len(docs) + len(counts), "average_length": avg,
                               "empty_documents": sum(not d.strip() for d in docs)}}
        write_json(p / "dataset.json", meta)
        print(name, meta["statistics"], flush=True)



# ========================================================================
# Build Graph
# ========================================================================
"""Sparse graph builder with disk-backed pair counts and streaming windows.

released_code mode intentionally counts repeated token pairs inside each window,
matching build_graph.py upstream. paper mode counts each distinct pair once/window.
Only released_code is used by the default experiment. SQLite avoids keeping all
windows and all pair-count Python objects in RAM simultaneously.
"""
import argparse
from array import array
from collections import Counter
from itertools import combinations
import math
from pathlib import Path
import sqlite3
import time


def windows(tokens, size):
    if size < 1:
        raise ValueError("Window size must be positive")
    if len(tokens) <= size:
        yield tokens
    else:
        for i in range(len(tokens) - size + 1):
            yield tokens[i:i + size]

def pair_counts(window, mode):
    counts = Counter(window)
    if mode not in ("paper", "released_code"):
        raise ValueError(mode)
    for a, b in combinations(sorted(counts), 2):
        yield (a, b), (counts[a] * counts[b] if mode == "released_code" else 1)

def pmi(pair_count, window_count, count_a, count_b):
    return math.log(pair_count * window_count / (count_a * count_b))

def build(root, name, window_size=20, pmi_mode="released_code"):
    import numpy as np
    import scipy.sparse as sp
    docs, meta = load_dataset(root, name)
    folder = Path(root) / "graphs" / name
    folder.mkdir(parents=True, exist_ok=True)
    settings = {"builder_version": 1, "window_size": window_size, "pmi_mode": pmi_mode,
                "clean_sha256": meta["clean_sha256"]}
    if (folder / "graph.json").exists():
        saved = read_json(folder / "graph.json")
        if saved["settings"] != settings:
            raise ValueError("Graph settings changed: use another --root")
        if digest(folder / "adjacency.npz") != saved["sha256"]:
            raise ValueError("Graph checksum mismatch")
        return saved
    started = time.perf_counter()
    vocab = sorted({w for doc in docs for w in doc.split()})
    ids = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)
    n = len(docs)
    df = Counter()
    wf = Counter()
    pending = Counter()
    total = 0
    db = folder / "pair_counts.sqlite"
    con = sqlite3.connect(db)
    con.execute("DROP TABLE IF EXISTS pairs")
    con.execute("CREATE TABLE pairs (a INTEGER, b INTEGER, count INTEGER, PRIMARY KEY(a,b)) WITHOUT ROWID")
    def flush():
        con.executemany("INSERT INTO pairs VALUES (?,?,?) ON CONFLICT(a,b) DO UPDATE SET count=count+excluded.count",
                        ((a, b, c) for (a, b), c in pending.items()))
        con.commit()
        pending.clear()
    try:
        for i, doc in enumerate(docs):
            tokens = [ids[w] for w in doc.split()]
            df.update(set(tokens))
            for window in windows(tokens, window_size):
                total += 1
                wf.update(set(window))
                for pair, count in pair_counts(window, pmi_mode):
                    pending[pair] += count
                if len(pending) >= 250000:
                    flush()
            if (i + 1) % 1000 == 0:
                print(f"{name}: counted {i+1}/{n} documents", flush=True)
        flush()
        rows, cols, weights = array("q"), array("q"), array("f")
        def add(a, b, weight):
            rows.extend((a, b)); cols.extend((b, a)); weights.extend((weight, weight))
        ww = dw = 0
        for a, b, count in con.execute("SELECT a,b,count FROM pairs"):
            weight = pmi(count, total, wf[a], wf[b])
            if weight > 0:
                add(n + a, n + b, weight)
                ww += 1
        for i, doc in enumerate(docs):
            for word, count in Counter(doc.split()).items():
                j = ids[word]
                weight = count * math.log(n / df[j])
                if weight > 0:
                    add(i, n + j, weight)
                    dw += 1
        adj = sp.coo_matrix((np.asarray(weights), (np.asarray(rows), np.asarray(cols))), shape=(n+v, n+v)).tocsr()
        adj = adj + sp.eye(n+v, dtype=np.float32, format="csr")
        inv = np.power(np.asarray(adj.sum(1)).ravel(), -0.5)
        adj = (sp.diags(inv) @ adj @ sp.diags(inv)).astype(np.float32).tocsr()
        sp.save_npz(folder / "adjacency.npz", adj)
        write_json(folder / "vocabulary.json", vocab)
        result = {"settings": settings, "sha256": digest(folder / "adjacency.npz"),
                  "node_order": "all documents in source order, then sorted vocabulary",
                  "nodes": n+v, "nonzeros_including_self_loops": adj.nnz,
                  "word_word_edges_undirected": ww, "document_word_edges_undirected": dw,
                  "windows": total, "build_seconds": time.perf_counter()-started}
        write_json(folder / "graph.json", result)
    finally:
        con.close()
    db.unlink(missing_ok=True)
    return result



# ========================================================================
# Train Models
# ========================================================================
"""Two independent trainers. Test evaluation happens only after training ends."""
import csv
import os
import random
import time
import warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def split_indices(meta, seed, fraction):
    original = np.array(meta["train_indices"], dtype=np.int64)
    order = np.random.default_rng(seed).permutation(original)
    size = int(len(order) * fraction)
    if not 0 < size < len(order):
        raise ValueError("Validation split must leave nonempty training and validation sets")
    train, val = order[size:].copy(), order[:size].copy()
    # Preserve the original random split unless it removes every training example
    # of a rare class. Swap one validation example for a training example from
    # a class with multiple remaining examples. Never touch the test split.
    from collections import Counter
    labels = np.asarray(meta["labels"])
    counts = Counter(labels[train].tolist())
    missing = sorted(set(labels[original].tolist()) - set(counts))
    for label in missing:
        candidates = [i for i, doc in enumerate(train) if counts[labels[doc]] > 1]
        if not candidates:
            raise ValueError("Training size cannot retain every class; reduce validation_fraction")
        vi = int(np.flatnonzero(labels[val] == label)[0])
        ti = candidates[0]
        counts[labels[train[ti]]] -= 1
        train[ti], val[vi] = val[vi], train[ti]
        counts[label] += 1
    return train, val, np.array(meta["test_indices"], dtype=np.int64)

def code_early_stop(losses, patience):
    # Matches upstream epoch > early_stopping, where epoch is zero-based.
    return len(losses) > patience + 1 and losses[-1] > float(np.mean(losses[-patience-1:-1]))

def evaluate_output(folder, meta, test, predictions, probabilities):
    y = np.asarray(meta["labels"])[test]
    labels = list(range(len(meta["class_names"])))
    metrics = {"accuracy": float(accuracy_score(y, predictions)),
               "macro_f1": float(f1_score(y, predictions, labels=labels, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(y, predictions, labels=labels, average="weighted", zero_division=0))}
    write_json(folder / "classification_report.json", classification_report(
        y, predictions, labels=labels, target_names=meta["class_names"], output_dict=True, zero_division=0))
    matrix = confusion_matrix(y, predictions, labels=labels)
    with (folder / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted"] + meta["class_names"])
        for name, row in zip(meta["class_names"], matrix):
            writer.writerow([name] + row.tolist())
    with (folder / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document_index", "document_id", "true_label", "predicted_label", "confidence"])
        for i, truth, pred, prob in zip(test, y, predictions, probabilities):
            writer.writerow([int(i), meta["document_ids"][i], meta["class_names"][truth],
                             meta["class_names"][pred], float(max(prob))])
    np.savez_compressed(folder / "probabilities.npz", document_indices=test, probabilities=probabilities)
    return metrics

def train_lr(docs, meta, split, seed, config, folder):
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train, val, test = split
    y = np.asarray(meta["labels"])
    # Explicit reconstruction: fit vocabulary and IDF on labelled training texts only.
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                 norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
    started = time.perf_counter()
    x = vectorizer.fit_transform([docs[i] for i in train])
    model = LogisticRegression(C=config["C"], solver="lbfgs", max_iter=config["max_iter"], random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y[train])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Logistic regression did not converge; increase max_iter in a new experiment root")
    seconds = time.perf_counter() - started
    joblib.dump({"vectorizer": vectorizer, "classifier": model, "classes": meta["class_names"]}, folder / "model.joblib")
    probs_small = model.predict_proba(vectorizer.transform([docs[i] for i in test]))
    probs = np.zeros((len(test), len(meta["class_names"])))
    probs[:, model.classes_] = probs_small
    metrics = evaluate_output(folder, meta, test, model.classes_[probs_small.argmax(1)], probs)
    metrics.update(training_seconds=seconds, epochs=None, selected_checkpoint="model.joblib",
                   validation_accuracy=float(model.score(vectorizer.transform([docs[i] for i in val]), y[val])))
    return metrics

def make_gcn(nodes, hidden, classes, dropout):
    import torch
    class TextGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w0 = torch.nn.Parameter(torch.empty(nodes, hidden))
            self.w1 = torch.nn.Parameter(torch.empty(hidden, classes))
            torch.nn.init.xavier_uniform_(self.w0)
            torch.nn.init.xavier_uniform_(self.w1)

        def forward(self, adjacency):
            # Identity features are implicit. Dropout on sparse identity drops rows
            # of W0 together, NOT individual elements of the learned embeddings.
            mask = torch.nn.functional.dropout(torch.ones((nodes, 1), device=self.w0.device),
                                              p=dropout, training=self.training)
            h = torch.relu(torch.sparse.mm(adjacency, self.w0 * mask))
            h = torch.nn.functional.dropout(h, p=dropout, training=self.training)
            return torch.sparse.mm(adjacency, h @ self.w1)
    return TextGCN()

def atomic_torch_save(obj, path):
    import torch
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def train_gcn(adj, meta, split, seed, config, folder, device, run_signature):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Select a GPU runtime or use cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    coo = adj.tocoo()
    a = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
                                torch.from_numpy(coo.data), size=coo.shape, check_invariants=True).coalesce().to(device)
    model = make_gcn(adj.shape[0], config["hidden"], len(meta["class_names"]), config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0, eps=1e-8)
    y = torch.tensor(meta["labels"], dtype=torch.long, device=device)
    train, val, test = split
    ti = torch.tensor(train, device=device); vi = torch.tensor(val, device=device)
    history, start, elapsed = [], 0, 0.0
    best_loss = float("inf")
    finished = False
    last_path = folder / "last.pt"
    if last_path.exists():
        # Only load checkpoints produced by this project; pickle is not safe for untrusted files.
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck["signature"] != run_signature:
            raise ValueError("Checkpoint/config mismatch")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        history = ck["history"]; start = ck["epoch"] + 1
        best_loss = ck["best_loss"]; elapsed = ck["training_seconds"]; finished = ck["finished"]
        torch.set_rng_state(ck["torch_rng"])
        if device.startswith("cuda") and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"]); random.setstate(ck["python_rng"])
    for epoch in range(start, config["epochs"]):
        if finished:
            break
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        model.train(); optimizer.zero_grad()
        out = model(a)
        loss = torch.nn.functional.cross_entropy(out[ti], y[ti])
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(a)
            vl = float(torch.nn.functional.cross_entropy(logits[vi], y[vi]).item())
            va = float((logits[vi].argmax(1) == y[vi]).float().mean().item())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - tick
        history.append({"epoch": epoch+1, "train_loss": float(loss.item()), "validation_loss": vl,
                        "validation_accuracy": va})
        improved = vl < best_loss
        best_loss = min(best_loss, vl)
        finished = code_early_stop([r["validation_loss"] for r in history], config["early_stopping"]) or epoch+1 == config["epochs"]
        ck = {"signature": run_signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "epoch": epoch, "history": history, "best_loss": best_loss, "finished": finished,
              "training_seconds": elapsed, "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
              "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
              "config": config, "seed": seed, "classes": meta["class_names"]}
        atomic_torch_save(ck, last_path)
        if improved:
            atomic_torch_save(ck, folder / "best_validation.pt")
        write_json(folder / "history.json", history)
        print(f"epoch={epoch+1} loss={loss.item():.4f} val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
    # Repair a history file if interruption occurred between checkpoint/history writes.
    write_json(folder / "history.json", history)
    # Use LAST state, not best_validation, to match original evaluation semantics.
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(a)[torch.tensor(test, device=device)], dim=1).cpu().numpy()
    metrics = evaluate_output(folder, meta, test, probabilities.argmax(1), probabilities)
    metrics.update(training_seconds=elapsed, epochs=len(history), selected_checkpoint="last.pt",
                   validation_accuracy=history[-1]["validation_accuracy"])
    return metrics


# ========================================================================
# Run Experiments
# ========================================================================
"""Plan or run experiments; finished runs are skipped and interrupted GCNs resume."""
import argparse
import importlib.metadata
import platform
import sys
import threading
import time
from pathlib import Path


def run(config_path, root, datasets=None, models=None, seeds=None, device=None, dry_run=False):
    config = read_json(config_path)
    datasets = datasets or config["datasets"]
    models = models or config["models"]
    seeds = seeds or config["seeds"]
    if not set(datasets) <= set(config["datasets"]) or not set(models) <= set(config["models"]):
        raise ValueError("Dataset/model selection is outside config")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    device = device or config["device"]
    plan = [(d, m, s) for d in datasets for m in models for s in seeds]
    print(f"{len(plan)} requested runs: {len(datasets)} datasets x {len(models)} models x {len(seeds)} seeds")
    if dry_run:
        for item in plan:
            print(*item)
        return
    import numpy as np
    import psutil
    import scipy.sparse as sp


    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source_hashes = {Path(__file__).name: digest(Path(__file__))}
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "scipy", "scikit-learn", "torch", "matplotlib", "joblib", "psutil"]}
    environment = {"python": sys.version, "platform": platform.platform(), "versions": versions,
                   "device": device, "source_hashes": source_hashes}
    # Changing the list of seeds is allowed; changing experiment semantics is not.
    protocol = {k: v for k, v in config.items() if k not in ["seeds", "device"]}
    for name in datasets:
        docs, meta = load_dataset(root, name)
        split = split_indices(meta, config["validation_seed"], config["validation_fraction"])
        if len(set(np.array(meta["labels"])[split[0]])) != len(meta["class_names"]):
            raise ValueError("Training subset lacks a class; explicitly revise validation protocol")
        split_path = root / "splits" / f"{name}.json"
        split_record = dict(zip(["train", "validation", "test"], [x.tolist() for x in split]))
        split_record["validation_seed"] = config["validation_seed"]
        if split_path.exists() and read_json(split_path) != split_record:
            raise ValueError("Saved split differs; use a new root")
        write_json(split_path, split_record)
        adjacency = graph_meta = None
        for model in models:
            if model == "textgcn":
                graph_meta = build(root, name, **config["graph"])
                adjacency = sp.load_npz(root / "graphs" / name / "adjacency.npz")
            for seed in seeds:
                folder = root / "runs" / name / model / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                settings = {"protocol": protocol, "dataset": name, "model": model, "seed": seed,
                            "data_sha256": meta["clean_sha256"], "labels_sha256": meta["metadata_sha256"],
                            "split": split_record, "graph": graph_meta, "environment": environment}
                signature = fingerprint(settings)
                manifest = folder / "manifest.json"
                if manifest.exists() and read_json(manifest)["signature"] != signature:
                    raise ValueError(f"Run settings or source changed: {folder}; use a new root")
                if (folder / "metrics.json").exists():
                    print("SKIP completed", name, model, seed, flush=True)
                    continue
                write_json(manifest, {"signature": signature, **settings})
                print("START", name, model, seed, flush=True)
                process = psutil.Process()
                peak = [process.memory_info().rss]
                stop = threading.Event()
                def sample():
                    while not stop.wait(0.1):
                        peak[0] = max(peak[0], process.memory_info().rss)
                monitor = threading.Thread(target=sample, daemon=True)
                monitor.start()
                started = time.perf_counter()
                try:
                    if model == "textgcn":
                        import torch
                        if device.startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        metrics = train_gcn(adjacency, meta, split, seed, config[model], folder, device, signature)
                        metrics["gpu_peak_allocated_mb"] = (torch.cuda.max_memory_allocated()/2**20
                                                            if device.startswith("cuda") else None)
                    else:
                        metrics = train_lr(docs, meta, split, seed, config[model], folder)
                    metrics.update(dataset=name, model=model, seed=seed, signature=signature,
                                   invocation_wall_seconds=time.perf_counter()-started,
                                   process_peak_rss_mb=max(peak[0], process.memory_info().rss)/2**20,
                                   n_train=len(split[0]), n_validation=len(split[1]), n_test=len(split[2]))
                    # Written last: acts as a completion marker. Earlier output files may be partial.
                    write_json(folder / "metrics.json", metrics)
                finally:
                    stop.set(); monitor.join()
                print("DONE", name, model, seed, metrics["accuracy"], flush=True)



# ========================================================================
# Make Results
# ========================================================================
"""Aggregate completed runs into auditable CSV/JSON tables and PNG/PDF figures."""
import argparse
import csv
from pathlib import Path
import statistics


def csv_out(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def summarize(root, config_path="config.json"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    cfg = read_json(config_path)
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    runs = [read_json(p) for p in sorted((root / "runs").glob("*/*/seed_*/metrics.json"))]
    if not runs:
        raise ValueError("No completed runs. Train a model first.")
    csv_out(out / "per_run.csv", [{k: r.get(k) for k in ["dataset", "model", "seed", "accuracy", "macro_f1",
            "weighted_f1", "training_seconds", "epochs", "process_peak_rss_mb", "gpu_peak_allocated_mb"]} for r in runs])
    summary = []
    for dataset in cfg["datasets"]:
        for model in cfg["models"]:
            selected = [r for r in runs if r["dataset"] == dataset and r["model"] == model]
            if not selected:
                continue
            acc = [r["accuracy"]*100 for r in selected]
            row = {"dataset": dataset, "model": model, "runs": len(selected),
                   "seeds": ";".join(str(r["seed"]) for r in selected),
                   "accuracy_mean_pct": statistics.mean(acc),
                   "accuracy_sample_std_pp": statistics.stdev(acc) if len(acc)>1 else None,
                   "paper_accuracy_pct": PAPER[model][dataset],
                   "difference_from_paper_pp": statistics.mean(acc)-PAPER[model][dataset],
                   "macro_f1_mean": statistics.mean(r["macro_f1"] for r in selected),
                   "weighted_f1_mean": statistics.mean(r["weighted_f1"] for r in selected),
                   "mean_training_seconds": statistics.mean(r["training_seconds"] for r in selected),
                   "max_process_rss_mb": max(r["process_peak_rss_mb"] for r in selected)}
            summary.append(row)
    csv_out(out / "summary.csv", summary)
    write_json(out / "summary.json", summary)
    observed = {(r["dataset"], r["model"], r["seed"]) for r in runs}
    missing = [(d,m,s) for d in cfg["datasets"] for m in cfg["models"] for s in cfg["seeds"] if (d,m,s) not in observed]
    write_json(out / "completion.json", {"planned_base_runs": len(cfg["datasets"])*len(cfg["models"])*len(cfg["seeds"]),
                                         "completed_runs_in_root": len(runs), "missing_base_runs": missing})
    stats = []
    for d in cfg["datasets"]:
        p = root / "data" / d / "dataset.json"
        if p.exists():
            stats.append({"dataset": d, **read_json(p)["statistics"]})
    csv_out(out / "dataset_statistics.csv", stats)
    md = ["# Text GCN Implementation — Results", "", "Accuracy is percent; ± is sample standard deviation across runs (ddof=1).",
          "The published means are reference values, not newly reproduced results. No significance claim is made.", "",
          "| Dataset | Model | Runs | Our accuracy | Paper | Difference (pp) |",
          "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        sd = r["accuracy_sample_std_pp"]
        value = f"{r['accuracy_mean_pct']:.2f}" + (f" ± {sd:.2f}" if sd is not None else " (one run)")
        md.append(f"| {r['dataset']} | {r['model']} | {r['runs']} | {value} | {r['paper_accuracy_pct']:.2f} | {r['difference_from_paper_pp']:+.2f} |")
    md += ["", f"Missing base-plan runs: {len(missing)}. See completion.json.", "",
           "Fixed validation split; repeated LR fits may be identical. Their zero spread is not evidence of general uncertainty being zero."]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), dpi=180)
        fig.savefig(out / (name + ".pdf"))
        plt.close(fig)
    # Missing results remain gaps, not zero-valued bars.
    x = np.arange(len(cfg["datasets"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, model in enumerate(cfg["models"]):
        rows = {r["dataset"]: r for r in summary if r["model"] == model}
        if not rows:
            continue
        positions = [i for i,d in enumerate(cfg["datasets"]) if d in rows]
        means = [rows[cfg["datasets"][i]]["accuracy_mean_pct"] for i in positions]
        errors = [rows[cfg["datasets"][i]]["accuracy_sample_std_pp"] or 0 for i in positions]
        bars = ax.bar(np.array(positions)+(j-.5)*.35, means, .35, color=["#4778B5", "#D67A32"][j],
               yerr=errors, capsize=4, label=model)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in means], padding=5, fontsize=9)
    ax.set(xticks=x, xticklabels=cfg["datasets"], ylabel="Test accuracy (%)", ylim=(0,110),
           xlim=(-0.6,len(x)-0.4), title="Our models: mean accuracy ± sample standard deviation")
    if missing or any(r["runs"] == 1 for r in summary):
        ax.text(0.01, 0.02, "Missing runs: gaps. One run: standard deviation unavailable.",
                transform=ax.transAxes, fontsize=9)
    ax.legend(); save(fig, "accuracy_comparison")
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = [r for r in summary if r["model"] == "textgcn"]
    positions = np.arange(len(rows))
    paper_bars = ax.bar(positions-.18, [r["paper_accuracy_pct"] for r in rows], .36, label="Paper (10 runs)")
    our_bars = ax.bar(positions+.18, [r["accuracy_mean_pct"] for r in rows], .36,
           yerr=[r["accuracy_sample_std_pp"] or 0 for r in rows], capsize=4, label="Our Text GCN")
    ax.bar_label(paper_bars, labels=[f"{r['paper_accuracy_pct']:.2f}%" for r in rows], padding=5, fontsize=9)
    ax.bar_label(our_bars, labels=[f"{r['accuracy_mean_pct']:.2f}%" for r in rows], padding=5, fontsize=9)
    ax.set(xticks=positions, xticklabels=[r["dataset"] for r in rows], ylabel="Test accuracy (%)", ylim=(0,110),
           title="Text GCN: published mean versus our result")
    if not rows:
        ax.text(0.5, 0.5, "No completed Text GCN runs yet", transform=ax.transAxes, ha="center")
    ax.legend(); save(fig, "textgcn_vs_paper")
    # These plots are additional diagnostics, not claimed replicas of paper figures.
    for p in sorted((root / "runs").glob("*/textgcn/seed_*/history.json")):
        if not (p.parent / "metrics.json").exists():
            continue
        h = read_json(p)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label="Training (dropout on)")
        ax.plot([r["epoch"] for r in h], [r["validation_loss"] for r in h], label="Validation (dropout off)")
        for key, offset, color in [("train_loss",10,"C0"),("validation_loss",12,"C1")]:
            ax.annotate(f"{h[-1][key]:.3f}", (h[-1]["epoch"],h[-1][key]),
                        xytext=((-5 if key=="train_loss" else 6),offset),
                        ha=("right" if key=="train_loss" else "left"),
                        textcoords="offset points",fontsize=9,color=color)
        ax.set_xlim(0,h[-1]["epoch"]*1.18)
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title=f"{p.parents[2].name}, {p.parent.name}")
        ax.legend(); save(fig, f"loss_{p.parents[2].name}_{p.parent.name}")
    print(f"Reports saved to {out}; {len(missing)} base-plan runs remain.")



# ========================================================================
# Audit Completed Runs
# ========================================================================
"""Read-only checks of saved run outputs; writes a separate audit report."""
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


def audit(root=Path('artifacts')):
    summary = read_json(root/'reports/summary.json')
    checks, groups = [], []
    for dataset in EXPECTED:
        _, meta = load_dataset(root,dataset)
        split = read_json(root/'splits'/f'{dataset}.json')
        tr,va,te = [set(split[k]) for k in ('train','validation','test')]
        assert not tr&va and not tr&te and not va&te
        assert tr|va == set(meta['train_indices']) and te == set(meta['test_indices'])
        graph = read_json(root/'graphs'/dataset/'graph.json')
        assert graph['nodes'] == meta['statistics']['nodes']
        for model,seeds in [('tfidf_lr',range(42,45)),('textgcn',range(42,52))]:
            results, reports, epochs, aggregate_confusion = [], [], [], None
            for seed in seeds:
                p = root/'runs'/dataset/model/f'seed_{seed}'
                required = ['metrics.json','manifest.json','predictions.csv','probabilities.npz',
                            'classification_report.json','confusion_matrix.csv']
                required += ['last.pt','best_validation.pt','history.json'] if model=='textgcn' else ['model.joblib']
                assert all((p/name).is_file() and (p/name).stat().st_size for name in required),str(p)
                m = read_json(p/'metrics.json'); manifest = read_json(p/'manifest.json')
                signature = manifest.pop('signature')
                assert signature == fingerprint(manifest) == m['signature']
                assert (m['dataset'],m['model'],m['seed']) == (dataset,model,seed)
                assert all(manifest['split'][k] == split[k] for k in ('train','validation','test'))
                with (p/'predictions.csv').open(newline='',encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                index = [int(r['document_index']) for r in rows]
                assert index == split['test']
                classes = meta['class_names']
                truth = np.array([meta['labels'][i] for i in index])
                pred = np.array([classes.index(r['predicted_label']) for r in rows])
                assert all(r['true_label']==classes[y] and r['document_id']==meta['document_ids'][i]
                           for r,y,i in zip(rows,truth,index))
                labels = list(range(len(classes)))
                recalculated = {'accuracy':accuracy_score(truth,pred),
                    'macro_f1':f1_score(truth,pred,labels=labels,average='macro',zero_division=0),
                    'weighted_f1':f1_score(truth,pred,labels=labels,average='weighted',zero_division=0)}
                assert all(abs(m[k]-v)<1e-10 for k,v in recalculated.items())
                with np.load(p/'probabilities.npz') as saved:
                    probs=saved['probabilities']
                    np.testing.assert_array_equal(saved['document_indices'],index)
                    assert probs.shape==(len(te),len(classes)) and np.isfinite(probs).all()
                    assert (probs>=0).all() and (probs<=1).all()
                    np.testing.assert_allclose(probs.sum(1),1,atol=1e-5)
                    np.testing.assert_array_equal(probs.argmax(1),pred)
                matrix=confusion_matrix(truth,pred,labels=labels)
                with (p/'confusion_matrix.csv').open(newline='',encoding='utf-8') as f:
                    saved=list(csv.reader(f))
                np.testing.assert_array_equal(matrix,np.array([r[1:] for r in saved[1:]],dtype=int))
                report=read_json(p/'classification_report.json')
                calculated=classification_report(truth,pred,labels=labels,target_names=classes,output_dict=True,zero_division=0)
                for c in classes:
                    for k in ('precision','recall','f1-score','support'):
                        assert abs(report[c][k]-calculated[c][k])<1e-10
                reports.append(report)
                aggregate_confusion=matrix.copy() if aggregate_confusion is None else aggregate_confusion+matrix
                if model=='textgcn':
                    history=read_json(p/'history.json')
                    assert len(history)==m['epochs'] and [h['epoch'] for h in history]==list(range(1,len(history)+1))
                    assert all(math.isfinite(h[k]) for h in history for k in ('train_loss','validation_loss','validation_accuracy'))
                    assert abs(history[-1]['validation_accuracy']-m['validation_accuracy'])<1e-10
                    assert m['selected_checkpoint']=='last.pt'
                    assert len(history)==200 or (len(history)>11 and history[-1]['validation_loss']>statistics.mean(h['validation_loss'] for h in history[-11:-1]))
                    epochs.append(len(history))
                    assert all((root/'reports'/f'loss_{dataset}_seed_{seed}.{ext}').exists() for ext in ('png','pdf'))
                results.append(m)
            s=next(r for r in summary if r['dataset']==dataset and r['model']==model)
            acc=[r['accuracy']*100 for r in results]
            assert s['runs']==len(results)
            assert abs(s['accuracy_mean_pct']-statistics.mean(acc))<1e-10
            assert abs(s['accuracy_sample_std_pp']-statistics.stdev(acc))<1e-10
            difficult=sorted([{'class':c,'mean_f1':statistics.mean(r[c]['f1-score'] for r in reports),
                               'test_documents':int(reports[0][c]['support']),
                               'training_documents':sum(meta['labels'][i]==j for i in tr)}
                              for j,c in enumerate(classes)],key=lambda r:r['mean_f1'])[:5]
            aggregate_confusion=aggregate_confusion.astype(float)/len(results)
            np.fill_diagonal(aggregate_confusion,0)
            confusions=[]
            positive = np.flatnonzero(aggregate_confusion.ravel() > 0)
            for flat in positive[np.argsort(aggregate_confusion.ravel()[positive])[-3:][::-1]]:
                i,j=np.unravel_index(flat,aggregate_confusion.shape)
                confusions.append({'true':classes[i],'predicted':classes[j],
                                   'mean_documents_per_run':float(aggregate_confusion[i,j])})
            groups.append({'dataset':dataset,'model':model,'runs':len(results),'min_accuracy':min(acc),
                           'max_accuracy':max(acc),'mean':statistics.mean(acc),'std':statistics.stdev(acc),
                           'epochs_range':[min(epochs),max(epochs)] if epochs else None,
                           'difficult_classes':difficult,'largest_confusions':confusions})
    total=sum(g['runs'] for g in groups)
    assert total==65
    write_json(root/'reports/audit_10_runs.json',{'verified_runs':total,'expected_textgcn_runs':50,
        'expected_baseline_runs':15,'checks':'Predictions, probabilities, metrics, confusion matrices, per-class reports, splits, signatures, histories and summary statistics verified. Checkpoint files present; weights not reloaded or inference rerun.',
        'groups':groups})
    print('PASS: 65 runs checked, including all 50 Text GCN runs.')
    for g in groups:
        if g['model']=='textgcn':
            print(g)



def main():
    parser = argparse.ArgumentParser(description="Text GCN implementation: download, train and evaluate")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Download and verify the five datasets")
    prepare_parser.add_argument("--root", default="artifacts")
    prepare_parser.add_argument("--datasets", nargs="+", choices=list(EXPECTED), default=list(EXPECTED))
    train_parser = commands.add_parser("train", help="Train either model; resume compatible runs")
    train_parser.add_argument("--config", default="config.json")
    train_parser.add_argument("--root", default="artifacts")
    train_parser.add_argument("--datasets", nargs="+", choices=list(EXPECTED))
    train_parser.add_argument("--models", nargs="+", choices=["tfidf_lr", "textgcn"])
    train_parser.add_argument("--seeds", nargs="+", type=int)
    train_parser.add_argument("--device", choices=["cpu", "cuda"])
    train_parser.add_argument("--dry-run", action="store_true")
    results_parser = commands.add_parser("results", help="Create result tables and plots")
    results_parser.add_argument("--root", default="artifacts")
    results_parser.add_argument("--config", default="config.json")
    audit_parser = commands.add_parser("audit", help="Check all 65 completed run outputs")
    audit_parser.add_argument("--root", default="artifacts")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.datasets)
    elif args.command == "train":
        run(args.config, args.root, args.datasets, args.models, args.seeds, args.device, args.dry_run)
    elif args.command == "results":
        summarize(args.root, args.config)
    elif args.command == "audit":
        audit(Path(args.root))


if __name__ == "__main__":
    main()
