#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aegis_dl.normalization import extract_structural_skeleton, normalize_text

DEVOPS_TRIGGER_RE = re.compile(
    r"\b(?:kill|sudo|chmod|chown|rm\s+-rf|drop\s+table|truncate|select\s+.+from|"
    r"nginx|postgres|redis|docker|kubectl|k8s|systemd|daemon|socket|firewall|"
    r"iptables|override|bypass|payload|ssh|jwt|bearer|token|cron|bash|python|"
    r"sql|injection|xss|csrf|privilege|root|admin|execute|dump|env|config)\b",
    re.IGNORECASE,
)

JAILBREAK_MARKER_RE = re.compile(
    r"\b(?:dan\b|do\s+anything\s+now|developer\s+mode|jailbreak|unrestricted|"
    r"no\s+ethical|without\s+moral|hypothetical\s+scenario|fictional\s+world|"
    r"evil\s+confidant|opposite\s+mode|stan\b|dude\b|aim\b|chaosgpt|roleplay\s+as)\b",
    re.IGNORECASE,
)


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


def compute_5gram_minhash(text: str, shingle_size: int = 5, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    if not text:
        m.update(b"_empty_")
        return m
    if len(text) < shingle_size:
        m.update(text.encode("utf-8"))
        return m
    shingles = {text[i : i + shingle_size].encode("utf-8") for i in range(len(text) - shingle_size + 1)}
    m.update_batch(list(shingles))
    return m


def _classify_benign_subcat(text: str, raw_cat: str = "", raw_source: str = "") -> str:
    if raw_cat in ("edge_case", "hard_negative") or raw_source == "harmbench_benign":
        return "hard_negative_devops"
    if DEVOPS_TRIGGER_RE.search(text):
        return "hard_negative_devops"
    return "benign_general"


def load_real_ood_suites(ood_dir: Path) -> dict[str, pd.DataFrame]:
    ood_dir.mkdir(parents=True, exist_ok=True)
    suites: dict[str, pd.DataFrame] = {}

    print("[*] Downloading real OOD holdout suite: Paul/XSTest...")
    xs_ds = load_dataset("Paul/XSTest", "default", split="train")
    xs_rows = []
    for row in xs_ds:
        text = normalize_text(str(row.get("prompt", "")))
        if not text:
            continue
        is_safe = str(row.get("label", "")).strip().lower() == "safe"
        xs_rows.append(
            {
                "text": text,
                "label": 0 if is_safe else 1,
                "sub_category": f"xstest_{row.get('type', 'general')}_{'safe' if is_safe else 'unsafe'}",
                "source": "Paul/XSTest",
            }
        )
    xstest_df = pd.DataFrame(xs_rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    xstest_df.to_parquet(ood_dir / "xstest.parquet", index=False)
    suites["xstest"] = xstest_df

    print("[*] Downloading real OOD holdout suite: leolee99/NotInject...")
    ni_rows = []
    for sp in ("NotInject_one", "NotInject_two", "NotInject_three"):
        ni_ds = load_dataset("leolee99/NotInject", "default", split=sp)
        for row in ni_ds:
            text = normalize_text(str(row.get("prompt", "")))
            if not text:
                continue
            ni_rows.append(
                {
                    "text": text,
                    "label": 0,
                    "sub_category": f"notinject_{row.get('category', sp)}",
                    "source": f"leolee99/NotInject:{sp}",
                }
            )
    notinject_df = pd.DataFrame(ni_rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    notinject_df.to_parquet(ood_dir / "notinject.parquet", index=False)
    suites["notinject"] = notinject_df

    print("[*] Downloading real OOD holdout suite: JailbreakBench/JBB-Behaviors...")
    jbb_rows = []
    for sp, lbl in (("harmful", 2), ("benign", 0)):
        jbb_beh = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=sp)
        for row in jbb_beh:
            text = normalize_text(str(row.get("Goal", "")))
            if not text:
                continue
            jbb_rows.append(
                {
                    "text": text,
                    "label": lbl,
                    "sub_category": f"jbb_behavior_{sp}_{row.get('Category', 'general')}",
                    "source": f"JailbreakBench/behaviors:{sp}",
                }
            )
    jbb_judge = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", split="test")
    for row in jbb_judge:
        text = normalize_text(str(row.get("prompt", "")))
        if not text or len(text) < 12:
            continue
        jbb_rows.append(
            {
                "text": text,
                "label": 2,
                "sub_category": "jbb_adversarial_prompt",
                "source": "JailbreakBench/judge_comparison",
            }
        )
    jbb_df = pd.DataFrame(jbb_rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    jbb_df.to_parquet(ood_dir / "jbb.parquet", index=False)
    suites["jbb"] = jbb_df

    print(
        f"[+] Saved real OOD suites: xstest={len(xstest_df)}, notinject={len(notinject_df)}, jbb={len(jbb_df)}"
    )
    return suites


def load_and_harmonize_in_domain(cfg: dict, seed: int = 42) -> pd.DataFrame:
    records: list[dict] = []

    print("[*] Ingesting neuralchemy/Prompt-injection-dataset (full)...")
    for sp in ("train", "validation", "test"):
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", "full", split=sp)
        for row in ds:
            text = normalize_text(str(row.get("text", "")))
            if len(text) < 8:
                continue
            raw_label = int(row.get("label", 0))
            raw_cat = str(row.get("category", "")).strip().lower()
            raw_src = str(row.get("source", "neuralchemy")).strip().lower()

            if raw_label == 0 or raw_cat == "benign":
                label = 0
                subcat = _classify_benign_subcat(text, raw_cat, raw_src)
            elif raw_cat in ("jailbreak", "adversarial") or raw_src in ("harmbench", "wildguard_judgecomp"):
                label = 2
                subcat = f"jailbreak_{raw_cat or 'adversarial'}"
            elif raw_cat in ("rag_poisoning", "token_smuggling", "encoding"):
                label = 1
                subcat = "indirect_rag_injection" if raw_cat == "rag_poisoning" else f"injection_{raw_cat}"
            else:
                label = 1
                subcat = "direct_delimiter_override"

            records.append(
                {
                    "text": text,
                    "label": label,
                    "sub_category": subcat,
                    "source": f"neuralchemy:{raw_src}",
                    "raw_group_id": str(row.get("group_id") or ""),
                }
            )

    print("[*] Ingesting hendzh/PromptShield...")
    ps_max = cfg.get("dataset", {}).get("in_domain_sources", {}).get("prompt_shield", {}).get("max_samples", 10000)
    ps_rows = []
    for sp in ("train", "validation"):
        ds = load_dataset("hendzh/PromptShield", "default", split=sp)
        for row in ds:
            text = normalize_text(str(row.get("prompt", "")))
            if len(text) < 10 or len(text) > 8000:
                continue
            raw_label = int(row.get("label", 0))
            if raw_label == 0:
                label = 0
                subcat = _classify_benign_subcat(text)
            else:
                if JAILBREAK_MARKER_RE.search(text):
                    label = 2
                    subcat = "jailbreak_persona"
                else:
                    label = 1
                    subcat = "indirect_rag_injection" if len(text) > 350 else "direct_delimiter_override"
            ps_rows.append(
                {
                    "text": text,
                    "label": label,
                    "sub_category": subcat,
                    "source": "hendzh/PromptShield",
                    "raw_group_id": "",
                }
            )
    ps_df = pd.DataFrame(ps_rows).drop_duplicates(subset=["text"])
    if len(ps_df) > ps_max:
        ps_df = ps_df.sample(n=ps_max, random_state=seed)
    records.extend(ps_df.to_dict("records"))

    print("[*] Ingesting xTRam1/safe-guard-prompt-injection...")
    for sp in ("train", "test"):
        ds = load_dataset("xTRam1/safe-guard-prompt-injection", "default", split=sp)
        for row in ds:
            text = normalize_text(str(row.get("text", "")))
            if len(text) < 10 or len(text) > 8000:
                continue
            raw_label = int(row.get("label", 0))
            if raw_label == 0:
                label = 0
                subcat = _classify_benign_subcat(text)
            else:
                if JAILBREAK_MARKER_RE.search(text):
                    label = 2
                    subcat = "jailbreak_persona"
                else:
                    label = 1
                    subcat = "direct_delimiter_override"
            records.append(
                {
                    "text": text,
                    "label": label,
                    "sub_category": subcat,
                    "source": "xTRam1/safe-guard",
                    "raw_group_id": "",
                }
            )

    print("[*] Ingesting jackhhao/jailbreak-classification...")
    for sp in ("train", "test"):
        ds = load_dataset("jackhhao/jailbreak-classification", "default", split=sp)
        for row in ds:
            text = normalize_text(str(row.get("prompt", "")))
            if len(text) < 10 or len(text) > 8000:
                continue
            is_jb = str(row.get("type", "")).strip().lower() == "jailbreak"
            records.append(
                {
                    "text": text,
                    "label": 2 if is_jb else 0,
                    "sub_category": "jailbreak_persona" if is_jb else _classify_benign_subcat(text),
                    "source": "jackhhao/jailbreak-classification",
                    "raw_group_id": "",
                }
            )

    print("[*] Ingesting rubend18/ChatGPT-Jailbreak-Prompts & deepset/prompt-injections...")
    jb_ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", "default", split="train")
    for row in jb_ds:
        text = normalize_text(str(row.get("Prompt", "")))
        if len(text) >= 15:
            records.append(
                {
                    "text": text[:6000],
                    "label": 2,
                    "sub_category": "jailbreak_dan_persona",
                    "source": "rubend18/ChatGPT-Jailbreak-Prompts",
                    "raw_group_id": "",
                }
            )

    for sp in ("train", "test"):
        ds = load_dataset("deepset/prompt-injections", "default", split=sp)
        for row in ds:
            text = normalize_text(str(row.get("text", "")))
            if len(text) < 8:
                continue
            raw_label = int(row.get("label", 0))
            records.append(
                {
                    "text": text,
                    "label": 1 if raw_label == 1 else 0,
                    "sub_category": "direct_delimiter_override" if raw_label == 1 else _classify_benign_subcat(text),
                    "source": "deepset/prompt-injections",
                    "raw_group_id": "",
                }
            )

    df = pd.DataFrame(records)
    df["char_len"] = df["text"].str.len()
    print(f"[+] Raw ingested in-domain corpus: {len(df):,} rows across {df['source'].nunique()} sources.")
    return df


def deduplicate_and_cluster_minhash(
    df: pd.DataFrame,
    ood_texts: list[str],
    shingle_size: int = 5,
    num_perm: int = 128,
    lsh_threshold: float = 0.75,
) -> pd.DataFrame:
    df = df.copy()
    df["norm_key"] = df["text"].str.strip().str.lower()
    df["skeleton"] = df["text"].map(extract_structural_skeleton)
    df = df[df["skeleton"].str.len() >= 5].reset_index(drop=True)

    label_nunique = df.groupby("norm_key")["label"].nunique()
    conflicting_keys = set(label_nunique[label_nunique > 1].index)
    if conflicting_keys:
        print(f"[*] Dropping {len(conflicting_keys)} exact texts with conflicting cross-dataset labels.")
        df = df[~df["norm_key"].isin(conflicting_keys)]

    df = df.drop_duplicates(subset=["norm_key"], keep="first").reset_index(drop=True)
    print(f"[+] After exact normalized deduplication: {len(df):,} unique prompts.")

    ood_norm_set = {t.strip().lower() for t in ood_texts if t}
    ood_skeletons = {extract_structural_skeleton(t) for t in ood_texts if t}
    ood_skeletons.discard("")

    ood_lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    for idx, skel in enumerate(ood_skeletons):
        if len(skel) >= shingle_size:
            ood_lsh.insert(f"ood_{idx}", compute_5gram_minhash(skel, shingle_size, num_perm))

    keep_mask = np.ones(len(df), dtype=bool)
    purged_exact = 0
    purged_lsh = 0
    in_domain_minhashes: list[MinHash] = []

    for i, row in enumerate(df.itertuples(index=False)):
        mh = compute_5gram_minhash(row.skeleton, shingle_size, num_perm)
        in_domain_minhashes.append(mh)
        if row.norm_key in ood_norm_set or row.skeleton in ood_skeletons:
            keep_mask[i] = False
            purged_exact += 1
        elif ood_lsh.query(mh):
            keep_mask[i] = False
            purged_lsh += 1

    if purged_exact or purged_lsh:
        print(
            f"[*] OOD Cross-Contamination Purge: removed {purged_exact} exact and {purged_lsh} near-duplicate (Jaccard>={lsh_threshold}) prompts."
        )
    df = df[keep_mask].reset_index(drop=True)
    kept_minhashes = [mh for mh, k in zip(in_domain_minhashes, keep_mask, strict=True) if k]

    n = len(df)
    dsu = DisjointSet(n)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)

    skel_to_first_idx: dict[str, int] = {}
    raw_gid_to_first_idx: dict[str, int] = {}

    for i, (row, mh) in enumerate(zip(df.itertuples(index=False), kept_minhashes, strict=True)):
        if row.raw_group_id:
            if row.raw_group_id in raw_gid_to_first_idx:
                dsu.union(i, raw_gid_to_first_idx[row.raw_group_id])
            else:
                raw_gid_to_first_idx[row.raw_group_id] = i

        if row.skeleton in skel_to_first_idx:
            dsu.union(i, skel_to_first_idx[row.skeleton])
        else:
            skel_to_first_idx[row.skeleton] = i

        candidates = lsh.query(mh)
        for cand_idx in candidates:
            dsu.union(i, cand_idx)
        lsh.insert(i, mh)

    cluster_ids = [f"grp_{dsu.find(i):06d}" for i in range(n)]
    df["group_id"] = cluster_ids

    cluster_majority_label = df.groupby("group_id")["label"].agg(lambda s: int(s.mode().iloc[0]))
    df["label"] = df["group_id"].map(cluster_majority_label)

    df = df.drop(columns=["norm_key", "skeleton", "raw_group_id"]).reset_index(drop=True)
    print(
        f"[+] MinHash LSH Clustering complete: {len(df):,} unique prompts grouped into {df['group_id'].nunique():,} disjoint clusters."
    )
    return df


def partition_and_save_splits(
    df: pd.DataFrame,
    output_dir: Path,
    n_splits: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(df), dtype=int)

    for fold_idx, (_, test_indices) in enumerate(sgkf.split(df, df["label"], groups=df["group_id"])):
        fold_assignments[test_indices] = fold_idx

    df["fold"] = fold_assignments

    train_df = df[df["fold"] < 14].drop(columns=["fold"]).reset_index(drop=True)
    val_df = df[(df["fold"] >= 14) & (df["fold"] < 17)].drop(columns=["fold"]).reset_index(drop=True)
    test_df = df[df["fold"] >= 17].drop(columns=["fold"]).reset_index(drop=True)

    train_gids, val_gids, test_gids = set(train_df["group_id"]), set(val_df["group_id"]), set(test_df["group_id"])
    assert len(train_gids & val_gids) == 0, "LEAKAGE: Train & Val share group_ids"
    assert len(train_gids & test_gids) == 0, "LEAKAGE: Train & Test share group_ids"
    assert len(val_gids & test_gids) == 0, "LEAKAGE: Val & Test share group_ids"

    train_txts = set(train_df["text"].str.strip().str.lower())
    val_txts = set(val_df["text"].str.strip().str.lower())
    test_txts = set(test_df["text"].str.strip().str.lower())
    assert len(train_txts & val_txts) == 0, "LEAKAGE: Train & Val share exact texts"
    assert len(train_txts & test_txts) == 0, "LEAKAGE: Train & Test share exact texts"
    assert len(val_txts & test_txts) == 0, "LEAKAGE: Val & Test share exact texts"

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        present = set(split["label"].unique())
        assert present == {0, 1, 2}, f"FATAL: {name} split missing classes! Found: {present}"

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"[+] Saved leak-free splits to {output_dir}:")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        ratios = split["label"].value_counts(normalize=True).sort_index().round(3).to_dict()
        counts = split["label"].value_counts().sort_index().to_dict()
        print(
            f"    {name:5s}: {len(split):6,d} rows ({split['group_id'].nunique():5,d} clusters) | counts={counts} | ratios={ratios}"
        )

    return train_df, val_df, test_df


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset_manifest(
    splits: dict[str, pd.DataFrame],
    ood: dict[str, pd.DataFrame],
    processed_dir: Path,
    ood_dir: Path,
    out_path: Path,
) -> None:
    manifest: dict = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "in_domain_splits": {},
        "ood_suites": {},
    }
    for name, df in splits.items():
        p = processed_dir / f"{name}.parquet"
        q = df["text"].str.len().quantile([0.0, 0.5, 0.9, 0.95, 0.99, 1.0]).round(1).to_dict()
        manifest["in_domain_splits"][name] = {
            "path": str(p.relative_to(out_path.parent.parent)),
            "sha256": _file_sha256(p),
            "num_rows": int(len(df)),
            "unique_texts": int(df["text"].nunique()),
            "unique_clusters": int(df["group_id"].nunique()),
            "label_counts": {int(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
            "sub_category_counts": {str(k): int(v) for k, v in df["sub_category"].value_counts().to_dict().items()},
            "source_counts": {str(k): int(v) for k, v in df["source"].value_counts().to_dict().items()},
            "char_length_quantiles": {str(k): float(v) for k, v in q.items()},
        }

    for name, df in ood.items():
        p = ood_dir / f"{name}.parquet"
        manifest["ood_suites"][name] = {
            "path": str(p.relative_to(out_path.parent.parent)),
            "sha256": _file_sha256(p),
            "num_rows": int(len(df)),
            "unique_texts": int(df["text"].nunique()),
            "label_counts": {int(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Wrote dataset manifest to {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_config = repo_root / "configs" / "milestone2_deberta_v3_small.yaml"

    parser = argparse.ArgumentParser(description="AegisLLM Real Data Ingestion & MinHash Zero-Leakage Splitter")
    parser.add_argument("--config", type=str, default=str(default_config))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("project", {}).get("default_seed", 42))
    clust_cfg = cfg.get("dataset", {}).get("clustering", {})
    shingle_size = int(clust_cfg.get("shingle_size", 5))
    num_perm = int(clust_cfg.get("num_permutations", 128))
    lsh_threshold = float(clust_cfg.get("lsh_threshold", 0.75))

    processed_dir = repo_root / "data" / "processed"
    ood_dir = repo_root / "data" / "ood"

    ood_suites = load_real_ood_suites(ood_dir)
    all_ood_texts: list[str] = []
    for ood_df in ood_suites.values():
        all_ood_texts.extend(ood_df["text"].tolist())

    raw_df = load_and_harmonize_in_domain(cfg, seed=seed)
    clean_df = deduplicate_and_cluster_minhash(
        raw_df,
        ood_texts=all_ood_texts,
        shingle_size=shingle_size,
        num_perm=num_perm,
        lsh_threshold=lsh_threshold,
    )

    train_df, val_df, test_df = partition_and_save_splits(
        clean_df,
        output_dir=processed_dir,
        n_splits=int(cfg.get("dataset", {}).get("splits", {}).get("n_folds", 20)),
        seed=seed,
    )

    write_dataset_manifest(
        splits={"train": train_df, "val": val_df, "test": test_df},
        ood=ood_suites,
        processed_dir=processed_dir,
        ood_dir=ood_dir,
        out_path=repo_root / "data" / "manifest.json",
    )
    print("[OK] Real dataset ingestion, MinHash LSH clustering, and zero-leakage verification completed.")


if __name__ == "__main__":
    main()
