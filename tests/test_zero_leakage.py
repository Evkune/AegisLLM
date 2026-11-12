"""
Unit tests validating zero-leakage experimental hygiene and dataset integrity.
Software Engineering for Deep Learning - ETSIINF (UPM)
"""

import os
import pandas as pd
import pytest


@pytest.fixture
def dataset_splits():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data", "processed")

    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        pytest.skip("Processed dataset parquet files not yet generated. Run scripts/prepare_dataset.py first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    return train_df, val_df, test_df


def test_zero_group_leakage_across_splits(dataset_splits):
    """Asserts that Train, Validation, and Test partitions share zero group_ids."""
    train_df, val_df, test_df = dataset_splits

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    assert len(train_groups & val_groups) == 0, f"Found {len(train_groups & val_groups)} overlapping groups between Train and Val!"
    assert len(train_groups & test_groups) == 0, f"Found {len(train_groups & test_groups)} overlapping groups between Train and Test!"
    assert len(val_groups & test_groups) == 0, f"Found {len(val_groups & test_groups)} overlapping groups between Val and Test!"


def test_normalization_hygiene(dataset_splits):
    """Asserts that all texts in the dataset are free from zero-width non-printing characters."""
    train_df, val_df, test_df = dataset_splits
    zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]

    for df, split_name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        for char in zero_width_chars:
            matches = df["text"].str.contains(char, regex=False)
            assert not matches.any(), f"Found unstripped zero-width codepoint '{repr(char)}' in {split_name} split!"


def test_label_validity(dataset_splits):
    """Asserts all labels are strictly in {0: BENIGN, 1: PROMPT_INJECTION, 2: JAILBREAK}."""
    train_df, val_df, test_df = dataset_splits
    valid_labels = {0, 1, 2}

    for df, split_name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        present_labels = set(df["label"].unique())
        assert present_labels.issubset(valid_labels), f"Invalid labels found in {split_name}: {present_labels - valid_labels}"
