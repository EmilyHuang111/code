

import re
import math
from dataclasses import dataclass
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
INPUT_CSV  = "generations_gpt-4o_gsm8k_with_answers.csv"   
OUTPUT_CSV = "gsmgpt_4o_results.csv"                   


CLUSTER_MIN_LEN = 2          # ignore paths with <2 steps inside cluster scoring
CLUSTER_MAX_LEN = 120        # ignore paths with >120 steps inside cluster scoring
PENALIZE_NO_MAJ  = True      # penalize clusters with no majority-supported bins
PENALTY_WEIGHT   = -1e9      # weight assigned if PENALIZE_NO_MAJ and no centroids
TRIM_Q           = 0.20      # trimmed-mean fraction (drop worst 20% per bin)
K_MIN, K_MAX     = 3, 32     # bounds for normalized step bins (global per question)

# Contrastive & priors
TAU_CONTRAST     = 0.35      # penalty strength for matching non-cluster centroids
ALPHA_REL        = 1.0       # exponent on avg reliability in cluster quality
BETA_SIZE        = 0.35      # exponent on cluster size prior (soft)
POS_WEIGHT_MODE  = "quadratic"  # positional weights: "uniform" | "linear" | "quadratic"

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

def normalize_answer_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r'\s+', ' ', str(s)).strip().lower()

def extract_final_answer_like(s: str) -> str:
    """
    Normalize final answer for clustering: prefer trailing numeric or yes/no.
    Falls back to trimmed+lowercased text.
    """
    if not isinstance(s, str):
        return ""
    txt = s.strip()
    # yes/no
    m = re.search(r'\b(yes|no)\b[\.\s]*$', txt, flags=re.I)
    if m:
        return m.group(1).lower()
    # trailing number (allow commas, currency, sign)
    m = re.search(r'([\$]?\s*[-+]?\d[\d,]*(?:\.\d+)?)[^\d]*$', txt)
    if m:
        return re.sub(r'[,\s]', '', m.group(1)).lstrip('$')
    return normalize_answer_text(txt)

def decompose_path(path: str) -> List[str]:
    """
    §2.2: Split primarily on newlines (natural CoT separators); fallback to sentence ends.
    """
    if not isinstance(path, str) or not path.strip():
        return []
    parts = re.split(r'\n+', path.strip())
    if len(parts) <= 1:
        parts = re.split(r'(?<=[\.\?\!])\s+', path.strip())
    return [p.strip() for p in parts if p.strip()]

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
def load_generations(csv_path: str) -> List[ReasoningPath]:
    df = pd.read_csv(csv_path)

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
        steps = decompose_path(raw)

        rp = ReasoningPath(
            problem_id     = int(row[col_map["problem_id"]]),
            generation_id  = int(row[col_map["generation_id"]]),
            question       = str(row[col_map["question"]]),
            ground_truth   = str(row[col_map["ground_truth"]]),
            raw_path       = raw,
            steps          = steps,
            final_answer   = str(row[col_map["model_answer"]]),
            correct_answer = str(row[col_map["correct_answer"]]),
            model          = str(row[col_map["model"]]),
            prob_like      = prob_like
        )
        gens.append(rp)
    return gens

def group_by_question(gens: List[ReasoningPath]) -> List[Question]:
    qmap: Dict[int, Question] = {}
    for p in tqdm(gens, desc="Grouping by question"):
        if p.problem_id not in qmap:
            qmap[p.problem_id] = Question(
                problem_id     = p.problem_id,
                question       = p.question,
                ground_truth   = p.ground_truth,
                paths          = [],
                correct_answer = p.correct_answer,
                model          = p.model
            )
        qmap[p.problem_id].paths.append(p)
    return list(qmap.values())

# =========================
# Baselines
# =========================
def pick_top_probability(q: Question) -> Optional[ReasoningPath]:
    if not q.paths:
        return None
    scored = [p for p in q.paths if p.prob_like is not None]
    if scored:
        return max(scored, key=lambda p: p.prob_like)
    return min(q.paths, key=lambda p: p.generation_id)

def pick_self_consistency(q: Question) -> Optional[ReasoningPath]:
    if not q.paths:
        return None
    votes = Counter(normalize_answer_text(extract_final_answer_like(p.final_answer)) for p in q.paths)
    if not votes:
        return None
    best_ans, _ = votes.most_common(1)[0]
    candidates = [p for p in q.paths if normalize_answer_text(extract_final_answer_like(p.final_answer)) == best_ans]
    scored = [p for p in candidates if p.prob_like is not None]
    if scored:
        return max(scored, key=lambda p: p.prob_like)
    return min(candidates, key=lambda p: p.generation_id)

def pick_scw(q: Question) -> Optional[ReasoningPath]:
    if not q.paths:
        return None
    texts = []
    for p in q.paths:
        t = p.raw_path if p.raw_path and p.raw_path.strip() else f"{p.question}\nAnswer: {p.final_answer}"
        texts.append(t)
    embs = embedding_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = (embs @ embs.T).cpu().numpy()
    per_sample_scores = sims.sum(axis=1)
    ans_keys = [normalize_answer_text(extract_final_answer_like(p.final_answer)) for p in q.paths]
    agg: Dict[str, float] = defaultdict(float)
    for i, key in enumerate(ans_keys):
        agg[key] += float(per_sample_scores[i])
    best_ans = max(agg.items(), key=lambda kv: kv[1])[0]
    cand_idx = [i for i,k in enumerate(ans_keys) if k == best_ans]
    best_i = max(cand_idx, key=lambda i: per_sample_scores[i])
    return q.paths[best_i]

# =========================
# HCR-W+++ (global K, contrastive bins, trimmed centroids, pos & reliability weights)
# =========================
def _unit_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n

def _encode_steps(paths: List[List[str]]) -> List[List[np.ndarray]]:
    """Encode list-of-steps -> list-of-unit-vectors per path."""
    all_texts: List[str] = [step for path in paths for step in path]
    if len(all_texts) == 0:
        return [[] for _ in paths]
    with torch.inference_mode():
        embs = embedding_model.encode(all_texts, convert_to_tensor=True, normalize_embeddings=True)
    embs = embs.detach().cpu().numpy()
    out: List[List[np.ndarray]] = []
    idx = 0
    for path in paths:
        k = len(path)
        out.append([embs[idx + t] for t in range(k)])
        idx += k
    return out

def _majority_threshold(n: int) -> int:
    return (n // 2) + 1  # strictly more than half

def _choose_K_global(lengths: List[int]) -> int:
    if not lengths:
        return K_MIN
    K = int(np.median(lengths))
    return max(K_MIN, min(K_MAX, K))

def _bin_index(j: int, L: int, K: int) -> int:
    if L <= 0:
        return 0
    idx = int(np.floor((j * K) / L))
    return min(max(idx, 0), K-1)

def _robust_centroid(vecs: np.ndarray, trim_q: float) -> Tuple[np.ndarray, float]:
    """
    1) mean -> mu0
    2) cosine sims to mu0 -> drop worst q-fraction
    3) recompute mean -> mu; reliability = mean cosine to mu (on kept set)
    """
    if vecs.shape[0] == 0:
        return np.zeros(vecs.shape[1], dtype=np.float32), 0.0
    mu0 = _unit_np(vecs.mean(axis=0))
    sims0 = vecs @ mu0
    if trim_q > 0.0 and vecs.shape[0] > 1:
        k = int(np.floor((1.0 - trim_q) * vecs.shape[0]))
        k = max(1, k)
        idx = np.argsort(sims0)[-k:]  # keep top k by cosine
        vecs_kept = vecs[idx]
    else:
        vecs_kept = vecs
    mu = _unit_np(vecs_kept.mean(axis=0))
    reliability = float((vecs_kept @ mu).mean())
    return mu, reliability

def _pos_weights(K: int, mode: str = "quadratic") -> np.ndarray:
    if K <= 0:
        return np.zeros(0, dtype=np.float32)
    if mode == "uniform":
        w = np.ones(K, dtype=np.float32)
    elif mode == "linear":
        w = np.linspace(0.5, 1.5, K, dtype=np.float32)
    else:  # "quadratic": emphasize later steps
        x = np.linspace(1.0 / K, 1.0, K, dtype=np.float32)
        w = x * x
    # normalize to mean 1.0 (so weights don’t change scale)
    return (w / max(1e-8, w.mean())).astype(np.float32)

def pick_hcrw(q: Question) -> Optional[ReasoningPath]:
    if not q.paths:
        return None

    # 1) Cluster by normalized final answer
    key_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(q.paths):
        key = normalize_answer_text(extract_final_answer_like(p.final_answer))
        key_to_indices[key].append(i)

    # 2) Pre-encode all steps, pick a GLOBAL K for this question
    all_steps = [p.steps if p.steps else decompose_path(p.raw_path) for p in q.paths]
    all_step_embs = _encode_steps(all_steps)
    all_lengths = [len(e) for e in all_step_embs]
    K = _choose_K_global(all_lengths)
    pos_w = _pos_weights(K, POS_WEIGHT_MODE)

    # Precompute per-path K-bin vectors (unit means per bin)
    per_path_bins: List[List[Optional[np.ndarray]]] = []
    for embs in all_step_embs:
        L = len(embs)
        if L == 0:
            per_path_bins.append([None] * K)
            continue
        buckets: Dict[int, List[np.ndarray]] = defaultdict(list)
        for j in range(L):
            b = _bin_index(j, L, K)
            buckets[b].append(embs[j])
        vecs = [None] * K
        for b, arr in buckets.items():
            v = _unit_np(np.mean(np.stack(arr, axis=0), axis=0))
            vecs[b] = v
        per_path_bins.append(vecs)

    # Build global (all-paths) pools per bin for later "negative" centroids
    global_pools: List[List[np.ndarray]] = [[] for _ in range(K)]
    for vecs in per_path_bins:
        for b in range(K):
            v = vecs[b]
            if v is not None:
                global_pools[b].append(v)

    # 3) Score clusters using contrastive, reliability & positional weighting
    cluster_weight: Dict[str, float] = {}
    cluster_best_idx: Dict[str, int] = {}
    cluster_meta: Dict[str, Tuple[int, float, float, int]] = {}  # (num_bins, avg_rel, best_path_score, cluster_size)

    for ans_key, idxs_orig in key_to_indices.items():
        if not idxs_orig:
            continue

        # length sanity filter (don’t drop all)
        idxs = []
        for i in idxs_orig:
            L = all_lengths[i]
            if CLUSTER_MIN_LEN <= L <= CLUSTER_MAX_LEN:
                idxs.append(i)
        if not idxs:
            idxs = idxs_orig

        cl_num = len(idxs)
        maj = _majority_threshold(cl_num)

        # Build cluster-only pools per bin & corresponding negative pools (others)
        pools_pos: List[List[np.ndarray]] = [[] for _ in range(K)]
        pools_neg: List[List[np.ndarray]] = [[] for _ in range(K)]
        in_cluster_mask = set(idxs)

        for i, vecs in enumerate(per_path_bins):
            for b in range(K):
                v = vecs[b]
                if v is None:
                    continue
                if i in in_cluster_mask:
                    pools_pos[b].append(v)
                else:
                    pools_neg[b].append(v)

        # Centroids & reliabilities (positive), and "negative" centroids
        centroids_pos: Dict[int, Tuple[np.ndarray, float]] = {}
        centroids_neg: Dict[int, np.ndarray] = {}
        for b in range(K):
            if len(pools_pos[b]) >= maj:
                vecs = np.stack(pools_pos[b], axis=0)
                mu_pos, rel = _robust_centroid(vecs, TRIM_Q)
                centroids_pos[b] = (mu_pos, rel)
            # For negatives, require at least a modest pool (≥2) to define a distractor centroid
            if len(pools_neg[b]) >= 2:
                mu_neg, _ = _robust_centroid(np.stack(pools_neg[b], axis=0), TRIM_Q)
                centroids_neg[b] = mu_neg

        if not centroids_pos:
            if PENALIZE_NO_MAJ:
                cluster_weight[ans_key] = PENALTY_WEIGHT
                cluster_best_idx[ans_key] = idxs[0]
                cluster_meta[ans_key] = (0, 0.0, 0.0, cl_num)
                continue

        num_bins = len(centroids_pos)
        avg_rel = float(np.mean([rel for (_, rel) in centroids_pos.values()])) if centroids_pos else 0.0

        # Score each path in the cluster
        path_scores: List[float] = []
        for i in idxs:
            vecs = per_path_bins[i]
            num, den, cover = 0.0, 0.0, 0
            for b, (mu_pos, rel) in centroids_pos.items():
                v = vecs[b]
                if v is None:
                    continue
                sim_pos = float(np.dot(v, mu_pos))
                # contrastive margin against available negative centroid at same bin
                if b in centroids_neg:
                    sim_neg = float(np.dot(v, centroids_neg[b]))
                    margin = sim_pos - TAU_CONTRAST * max(0.0, sim_neg)
                else:
                    margin = sim_pos
                w = rel * float(pos_w[b])   # reliability × positional weight
                num += w * margin
                den += w
                cover += 1
            base = (num / den) if den > 0 else 0.0
            coverage_ratio = cover / max(1, num_bins)
            # light length normalization (discourage rambling): scale by sqrt(K / (K + gaps))
            gaps = num_bins - cover
            length_norm = math.sqrt(max(1, cover) / max(1, num_bins))
            score = base * coverage_ratio * length_norm
            path_scores.append(score)

        # Cluster quality multiplier: (avg_rel^alpha) * (size^beta)
        quality = (max(1e-6, avg_rel) ** ALPHA_REL) * (cl_num ** BETA_SIZE)
        total_weight = (float(np.sum(path_scores)) if path_scores else 0.0) * quality
        cluster_weight[ans_key] = total_weight

        if path_scores:
            best_local = int(np.argmax(path_scores))
            best_score = float(path_scores[best_local])
            cluster_best_idx[ans_key] = idxs[best_local]
            cluster_meta[ans_key] = (num_bins, avg_rel, best_score, cl_num)
        else:
            cluster_best_idx[ans_key] = idxs[0]
            cluster_meta[ans_key] = (num_bins, avg_rel, 0.0, cl_num)

    if not cluster_weight:
        return None

    # 4) Select cluster w/ max weight; tie-breaks by hierarchy quality
    items = list(cluster_weight.items())
    max_w = max(w for _, w in items)
    eps = 1e-12
    tied = [k for k, w in items if abs(w - max_w) <= eps]

    if len(tied) > 1:
        # tie-breaks: more bins -> higher avg reliability -> higher best path score -> larger cluster -> MV size
        def tb_key(k):
            nbin, avgrel, bestp, csize = cluster_meta.get(k, (0, 0.0, 0.0, 0))
            mv_size = len(key_to_indices.get(k, []))
            return (nbin, avgrel, bestp, csize, mv_size)
        tied.sort(key=tb_key, reverse=True)
        winner_key = tied[0]
    else:
        winner_key = tied[0]

    return q.paths[cluster_best_idx[winner_key]]

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
    return {"name": name, "accuracy": acc, "evaluated": total, "selections": selections}

# =========================
# Run all (Jupyter-friendly)
# =========================
def run_all(input_csv: str, output_csv: str = ""):
    gens = load_generations(input_csv)
    questions = group_by_question(gens)

    results = []
    results.append(evaluate(questions, pick_top_probability, "Top-Probability"))
    results.append(evaluate(questions, pick_self_consistency, "Self-Consistency (MV)"))
    results.append(evaluate(questions, pick_scw, "Semantic Self-Consistency (SCW)"))
    results.append(evaluate(questions, pick_hcrw, "HCR-W (global-K, contrastive, weighted)"))

    print("\n=== Comparison ===")
    width = 48
    print(f"{'Method':{width}} | {'Accuracy':>9} | {'Evaluated':>9}")
    print("-" * (width + 1 + 12 + 12))
    for r in results:
        acc = f"{r['accuracy']:.2f}%" if not np.isnan(r['accuracy']) else "NaN"
        print(f"{r['name']:{width}} | {acc:>9} | {r['evaluated']:>9}")

    if output_csv:
        pid_set = sorted({q.problem_id for q in questions})
        rows: Dict[int, Dict[str, object]] = {pid: {"problem_id": pid} for pid in pid_set}
        name_to_key = {
            "Top-Probability": "top_probability",
            "Self-Consistency (MV)": "self_consistency",
            "Semantic Self-Consistency (SCW)": "scw",
            "HCR-W (global-K, contrastive, weighted)": "hcrw"
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
    run_all(INPUT_CSV, OUTPUT_CSV)
