import os
from pathlib import Path
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OOD_DIR = REPO_ROOT / "data" / "ood"


@pytest.fixture(scope="module")
def dataset_splits():
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Processed dataset parquet files not found. Run scripts/prepare_dataset.py first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    return train_df, val_df, test_df


@pytest.fixture(scope="module")
def ood_suites():
    xstest_path = OOD_DIR / "xstest.parquet"
    notinject_path = OOD_DIR / "notinject.parquet"
    jbb_path = OOD_DIR / "jbb.parquet"

    if not (xstest_path.exists() and notinject_path.exists() and jbb_path.exists()):
        pytest.skip("OOD dataset parquet files not found. Run scripts/prepare_dataset.py first.")

    return {
        "xstest": pd.read_parquet(xstest_path),
        "notinject": pd.read_parquet(notinject_path),
        "jbb": pd.read_parquet(jbb_path),
    }


def test_zero_group_leakage_across_splits(dataset_splits):
    train_df, val_df, test_df = dataset_splits

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    assert len(train_groups & val_groups) == 0, f"Group overlap Train/Val: {len(train_groups & val_groups)}"
    assert len(train_groups & test_groups) == 0, f"Group overlap Train/Test: {len(train_groups & test_groups)}"
    assert len(val_groups & test_groups) == 0, f"Group overlap Val/Test: {len(val_groups & test_groups)}"


def test_zero_exact_text_leakage(dataset_splits):
    train_df, val_df, test_df = dataset_splits

    train_texts = set(train_df["text"].str.strip().str.lower())
    val_texts = set(val_df["text"].str.strip().str.lower())
    test_texts = set(test_df["text"].str.strip().str.lower())

    assert len(train_texts & val_texts) == 0, f"Exact text overlap Train/Val: {len(train_texts & val_texts)}"
    assert len(train_texts & test_texts) == 0, f"Exact text overlap Train/Test: {len(train_texts & test_texts)}"
    assert len(val_texts & test_texts) == 0, f"Exact text overlap Val/Test: {len(val_texts & test_texts)}"


def test_zero_ood_contamination(dataset_splits, ood_suites):
    train_df, val_df, test_df = dataset_splits
    in_domain_texts = (
        set(train_df["text"].str.strip().str.lower())
        | set(val_df["text"].str.strip().str.lower())
        | set(test_df["text"].str.strip().str.lower())
    )

    for suite_name, ood_df in ood_suites.items():
        ood_texts = set(ood_df["text"].str.strip().str.lower())
        overlap = in_domain_texts & ood_texts
        assert len(overlap) == 0, f"Contamination detected between in-domain splits and OOD '{suite_name}': {len(overlap)} overlapping prompts"


def test_ood_cardinality_real(ood_suites):
    assert ood_suites["xstest"]["text"].nunique() >= 400, "XSTest suite lacks real cardinality (expected >= 400 unique prompts)"
    assert ood_suites["notinject"]["text"].nunique() >= 300, "NotInject suite lacks real cardinality (expected >= 300 unique prompts)"
    assert ood_suites["jbb"]["text"].nunique() >= 180, "JailbreakBench suite lacks real cardinality (expected >= 180 unique prompts)"


def test_length_diversity(dataset_splits):
    train_df, _, _ = dataset_splits
    p95_len = float(train_df["text"].str.len().quantile(0.95))
    assert p95_len >= 400.0, f"Training set lacks long-context prompt diversity: P95 char length is {p95_len:.1f} (< 400)"


def test_normalization_hygiene(dataset_splits):
    train_df, val_df, test_df = dataset_splits
    zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]

    for df, split_name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        for char in zero_width_chars:
            matches = df["text"].str.contains(char, regex=False)
            assert not matches.any(), f"Found unstripped zero-width codepoint '{repr(char)}' in {split_name} split"


def test_label_validity(dataset_splits):
    train_df, val_df, test_df = dataset_splits
    expected_labels = {0, 1, 2}

    for df, split_name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        present_labels = set(df["label"].unique())
        assert present_labels == expected_labels, f"Split {split_name} must contain all 3 classes {expected_labels}, found {present_labels}"
