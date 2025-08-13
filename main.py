# compare_four_methods_notebook.py — single Jupyter cell
# -*- coding: utf-8 -*-

import re
import math
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# =========================
# Config (set these and run)
# =========================
INPUT_CSV  = "generations_gpt-3.5-turbo_gsm8k_3.5_with_answers.csv"   # <-- your input file
OUTPUT_CSV = "gsm3.5_results.csv"                        # <-- set "" to skip saving
STEP_THR   = 0.55  # HCR-W: step cluster threshold (lower than before)
ANS_THR    = 0.60  # HCR-W: answer cluster threshold (lower than before)

# =========================
# Embeddings & utilities
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

def clean_number(x):
    """Turn strings like '####18', '$70,000', '8.' into float; NaN if it fails."""
    try:
        s = re.sub(r'[^\d\.\-]', '', str(x))
        if s in ('', '.', '-'):
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")

def normalize_answer_text(s):
    if s is None:
        return ""
    return re.sub(r'\s+', ' ', str(s)).strip().lower()

# =========================
# Data structures
# =========================
@dataclass
class ReasoningPath:
    problem_id: int
    generation_id: int
    question: str
    ground_truth: str
    raw_path: str
    steps: List[str]
    final_answer: str
    correct_answer: str
    model: str
    prob_like: Optional[float] = None  # probability / logprob / score if available

@dataclass
class Question:
    problem_id: int
    question: str
    ground_truth: str
    paths: List[ReasoningPath]
    correct_answer: str
    model: str

# =========================
# Loading
# =========================
def decompose_path(path: str, _separator_ignored: str | None = None):
    """
    Split reasoning into steps. Prefer newlines; otherwise split on sentence enders.
    More robust than a raw '.' split.
    """
    if not isinstance(path, str) or not path.strip():
        return []
    parts = re.split(r'(?:\n+|(?<=[\.\?\!])\s+)', path.strip())
    return [p.strip() for p in parts if p.strip()]

def load_generations(csv_path: str) -> List[ReasoningPath]:
    df = pd.read_csv(csv_path)

    # Accept a few column name variants
    def pick_exact_or_alias(aliases):
        for c in df.columns:
            if c and c.lower() in aliases:
                return c
        return None

    col_map = {
        "problem_id":      pick_exact_or_alias({"problem_id"}),
        "generation_id":   pick_exact_or_alias({"generation_id","sample_id","gen_id"}),
        "question":        pick_exact_or_alias({"question"}),
        "ground_truth":    pick_exact_or_alias({"ground_truth","gt","label","answer"}),
        "generated_path":  pick_exact_or_alias({"generated_path","rationale","cot","chain_of_thought"}),
        "model_answer":    pick_exact_or_alias({"model_answer","final_answer","prediction"}),
        "correct_answer":  pick_exact_or_alias({"correct_answer","target","gold_answer","gold"}),
        "model":           pick_exact_or_alias({"model"}),
    }

    req = ["problem_id","generation_id","question","ground_truth",
           "generated_path","model_answer","correct_answer","model"]
    for k in req:
        if col_map[k] is None:
            raise ValueError(f"Missing required column for '{k}'. Found columns: {list(df.columns)}")

    # Prob/score columns (pick best available)
    prob_cols = [c for c in df.columns if c and c.lower() in
                 {"prob","sequence_prob","sequence_logprob","logprob","logp","score"}]

    gens: List[ReasoningPath] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading generations"):
        ans = row[col_map["model_answer"]]
        if ans is None or str(ans).strip().lower() in {"", "nan", "none", "unanswerable"}:
            continue

        prob_like = None
        if prob_cols:
            # Prefer prob > score > logprob-like
            prefer = ["prob","sequence_prob","score","sequence_logprob","logprob","logp"]
            for pname in prefer:
                hit = next((c for c in prob_cols if c.lower() == pname), None)
                if hit is not None:
                    try:
                        prob_like = float(row[hit])
                    except Exception:
                        prob_like = None
                    break

        raw = str(row[col_map["generated_path"]])
        rp = ReasoningPath(
            problem_id   = int(row[col_map["problem_id"]]),
            generation_id= int(row[col_map["generation_id"]]),
            question     = str(row[col_map["question"]]),
            ground_truth = str(row[col_map["ground_truth"]]),
            raw_path     = raw,
            steps        = decompose_path(raw),
            final_answer = str(row[col_map["model_answer"]]),
            correct_answer = str(row[col_map["correct_answer"]]),
            model        = str(row[col_map["model"]]),
            prob_like    = prob_like
        )
        gens.append(rp)
    return gens

def group_by_question(gens: List[ReasoningPath]) -> List[Question]:
    qmap: Dict[int, Question] = {}
    for p in tqdm(gens, desc="Grouping by question"):
        if p.problem_id not in qmap:
            qmap[p.problem_id] = Question(
                problem_id=p.problem_id,
                question=p.question,
                ground_truth=p.ground_truth,
                paths=[],
                correct_answer=p.correct_answer,
                model=p.model
            )
        qmap[p.problem_id].paths.append(p)
    return list(qmap.values())

# =========================
# Method 1: Top-Probability
# =========================
def pick_top_probability(q: Question) -> Optional[ReasoningPath]:
    if not q.paths:
        return None
    # If a true prob column exists, higher is better (for logprob: less negative is higher).
    scored = [p for p in q.paths if p.prob_like is not None]
    if scored:
        return max(scored, key=lambda p: p.prob_like)
    # Fallback: earliest generation_id
    return min(q.paths, key=lambda p: p.generation_id)

# =========================
# Method 2: Self-Consistency (MV)
# =========================
def pick_self_consistency(q: Question) -> Optional[ReasoningPath]:
    if not q.paths:
        return None
    votes = Counter(normalize_answer_text(p.final_answer) for p in q.paths)
    if not votes:
        return None
    best_ans, _ = votes.most_common(1)[0]
    candidates = [p for p in q.paths if normalize_answer_text(p.final_answer) == best_ans]
    # tie-break by prob if available, else earliest generation_id
    scored = [p for p in candidates if p.prob_like is not None]
    if scored:
        return max(scored, key=lambda p: p.prob_like)
    return min(candidates, key=lambda p: p.generation_id)

# ===========================================
# Method 3: Semantic Self-Consistency (SCW)
# (embed WHOLE rationale per sample)
# ===========================================
def pick_scw(q: Question) -> Optional[ReasoningPath]:
    if not q.paths:
        return None
    # Use whole reasoning path string; if empty, fall back to question+answer
    texts = []
    for p in q.paths:
        t = p.raw_path if p.raw_path and p.raw_path.strip() else f"{p.question}\nAnswer: {p.final_answer}"
        texts.append(t)

    # Unit-normalized for cosine via dot
    embs = embedding_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    # Cosine sim matrix (N x N)
    sims = (embs @ embs.T).cpu().numpy()
    # Per-sample score = sum of similarities to others
    per_sample_scores = sims.sum(axis=1)

    # Sum scores per normalized answer
    ans_keys = [normalize_answer_text(p.final_answer) for p in q.paths]
    agg: Dict[str, float] = defaultdict(float)
    for i, key in enumerate(ans_keys):
        agg[key] += float(per_sample_scores[i])

    # Pick answer with largest aggregate
    best_ans = max(agg.items(), key=lambda kv: kv[1])[0]
    cand_idx = [i for i,k in enumerate(ans_keys) if k == best_ans]
    # If multiple, pick the sample with max individual score
    best_i = max(cand_idx, key=lambda i: per_sample_scores[i])
    return q.paths[best_i]

# ==========================================================
# Method 4: HCR-W — step clustering + coherence (normalized)
# ==========================================================
def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n

def _cluster_vectors(vectors: List[torch.Tensor], threshold=0.7):
    """
    Incremental RMS-of-cosine-to-centroid threshold clustering.
    Vectors are pre-normalized to unit length, so cosine = dot.
    Returns (assignments, centroids as unit np arrays).
    """
    # Pre-normalize vectors to unit length for cosine geometry
    unit = [t.detach().cpu().numpy() for t in vectors]
    unit = [_l2norm(u) for u in unit]

    clusters: List[List[np.ndarray]] = []
    centroids: List[np.ndarray] = []
    assignments = np.full(len(unit), -1, dtype=int)

    for i, v in enumerate(unit):
        best = -1
        for cidx, cl in enumerate(clusters):
            cl.append(v)
            new_centroid = _l2norm(np.mean(cl, axis=0))
            # RMS of cosine to centroid (now dot because unit vectors)
            sims_sq = [(np.dot(new_centroid, vec))**2 for vec in cl]
            rms = math.sqrt(float(np.mean(sims_sq))) if sims_sq else 0.0
            cl.pop()
            if rms >= threshold:
                best = cidx
                break
        if best != -1:
            clusters[best].append(v)
            assignments[i] = best
            centroids[best] = _l2norm(np.mean(clusters[best], axis=0))
        else:
            clusters.append([v])
            centroids.append(v)
            assignments[i] = len(clusters) - 1
    return assignments, centroids

def _filter_outlier_paths(paths: List[ReasoningPath], min_len=20, max_len=2000):
    """
    Cheap semantic hygiene: drop obviously degenerate or too-short/too-long rationales.
    Never drop all.
    """
    kept = []
    for p in paths:
        txt = (p.raw_path or "").strip()
        if min_len <= len(txt) <= max_len and not re.search(r'(sorry|cannot|as an ai)', txt.lower()):
            kept.append(p)
    return kept if kept else paths

def pick_hcrw(q: Question, step_thr=0.55, ans_thr=0.60) -> Optional[ReasoningPath]:
    if not q.paths:
        return None

    paths = _filter_outlier_paths(q.paths)

    # Embed all steps per path (normalized embeddings)
    paths_step_embs: List[List[torch.Tensor]] = []
    for p in paths:
        if p.steps:
            embs = embedding_model.encode(p.steps, convert_to_tensor=True, normalize_embeddings=True)
        else:
            embs = embedding_model.encode([p.raw_path or ""], convert_to_tensor=True, normalize_embeddings=True)
        paths_step_embs.append(list(embs))

    # Flatten steps to cluster
    flat_steps = [e for path in paths_step_embs for e in path]
    if len(flat_steps) == 0:
        # fallback to SCW if steps are empty
        return pick_scw(q)

    step_assign, step_centroids = _cluster_vectors(flat_steps, threshold=step_thr)

    # Cluster answer embeddings (normalized)
    answer_texts = [p.final_answer for p in paths]
    ans_embs = embedding_model.encode(
        [f"The answer is {a}" for a in answer_texts],
        convert_to_tensor=True, normalize_embeddings=True
    )
    ans_assign, ans_centroids = _cluster_vectors(list(ans_embs), threshold=ans_thr)

    # Compute path coherence (avg cosine to its step cluster centroid)
    path_scores = np.zeros(len(paths), dtype=float)
    idx = 0
    for pi, emb_list in enumerate(paths_step_embs):
        if len(emb_list) == 0:
            continue
        s = 0.0
        for e in emb_list:
            cl = step_assign[idx]
            c = step_centroids[cl]  # unit
            e_np = e.detach().cpu().numpy()  # unit
            cs = float(np.dot(c, e_np))  # cosine
            s += cs
            idx += 1
        path_scores[pi] = s / max(1, len(emb_list))

    # Aggregate scores per answer cluster
    cluster_scores = np.zeros(len(ans_centroids), dtype=float)
    for pi, cidx in enumerate(ans_assign):
        cluster_scores[cidx] += path_scores[pi]

    best_cluster = int(np.argmax(cluster_scores))
    # Among candidates in the best cluster, pick the one closest to the cluster centroid
    cand = [i for i,c in enumerate(ans_assign) if c == best_cluster]
    if not cand:
        # fallback to best path score globally
        return paths[int(np.argmax(path_scores))]
    best_i = -1
    best_sim = -1.0
    for i in cand:
        e = ans_embs[i].detach().cpu().numpy()  # unit
        c = ans_centroids[best_cluster]         # unit
        sim = float(np.dot(e, c))               # cosine
        if sim > best_sim:
            best_sim = sim
            best_i = i
    return paths[best_i]

# =========================
# Evaluation helpers
# =========================
def evaluate(questions: List[Question], picker_fn, name: str, show_bar: bool = True):
    correct = 0
    total = 0
    selections: List[Tuple[int, str]] = []
    iterator = tqdm(questions, desc=f"Evaluating: {name}") if show_bar else questions
    for q in iterator:
        sel = picker_fn(q)
        if sel is None:
            continue
        gt = clean_number(q.correct_answer)
        pred = clean_number(sel.final_answer)
        if not (np.isnan(gt) or np.isnan(pred)):
            total += 1
            if gt == pred:
                correct += 1
        selections.append((q.problem_id, sel.final_answer))
    acc = 100.0 * correct / total if total > 0 else float("nan")
    return {
        "name": name,
        "accuracy": acc,
        "evaluated": total,
        "selections": selections
    }

# =========================
# Run all (Jupyter-friendly)
# =========================
def run_all(input_csv: str, output_csv: str = "", step_thr: float = STEP_THR, ans_thr: float = ANS_THR):
    gens = load_generations(input_csv)
    questions = group_by_question(gens)

    results = []
    # Each evaluate() shows its own progress bar
    results.append(evaluate(questions, pick_top_probability, "Top-Probability"))
    results.append(evaluate(questions, pick_self_consistency, "Self-Consistency (MV)"))
    results.append(evaluate(questions, pick_scw, "Semantic Self-Consistency (SCW)"))
    results.append(evaluate(questions, lambda q: pick_hcrw(q, step_thr, ans_thr), "HCR-W (yours)"))

    # Print summary table
    print("\n=== Comparison ===")
    width = 30
    print(f"{'Method':{width}} | {'Accuracy':>9} | {'Evaluated':>9}")
    print("-" * (width + 1 + 12 + 12))
    for r in results:
        acc = f"{r['accuracy']:.2f}%" if not np.isnan(r['accuracy']) else "NaN"
        print(f"{r['name']:{width}} | {acc:>9} | {r['evaluated']:>9}")

    # Optional: save per-problem selections to CSV
    if output_csv:
        # Build a dataframe with columns: problem_id, top_probability, self_consistency, scw, hcrw
        pid_set = sorted({q.problem_id for q in questions})
        rows: Dict[int, Dict[str, object]] = {pid: {"problem_id": pid} for pid in pid_set}
        name_to_key = {
            "Top-Probability": "top_probability",
            "Self-Consistency (MV)": "self_consistency",
            "Semantic Self-Consistency (SCW)": "scw",
            "HCR-W (yours)": "hcrw"
        }
        for r in results:
            key = name_to_key[r["name"]]
            for pid, ans in r["selections"]:
                rows[pid][key] = ans
        df_out = pd.DataFrame([rows[pid] for pid in pid_set]).sort_values("problem_id")
        df_out.to_csv(output_csv, index=False)
        print(f"\nSaved selections to: {output_csv}")

# =========================
# Execute in notebook
# =========================
if __name__ == "__main__" or True:
    run_all(INPUT_CSV, OUTPUT_CSV, STEP_THR, ANS_THR)
