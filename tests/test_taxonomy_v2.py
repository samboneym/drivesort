"""Tests for taxonomy_v2 — node model, persistence, tree navigation."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from drivesort.taxonomy_v2 import TaxonomyV2, TaxonomyNode


def make_unit(values: list[float]) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


@pytest.fixture
def tmp_path_taxonomy(tmp_path):
    return tmp_path / "taxonomy.json"


@pytest.fixture
def tax(tmp_path_taxonomy):
    return TaxonomyV2(path=tmp_path_taxonomy)


class TestAddNode:
    def test_add_root_node(self, tax):
        embs = np.stack([make_unit([1, 0, 0]), make_unit([0.9, 0.1, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a", "b"], folder_id="fid1")
        assert "books" in tax.nodes
        assert tax.nodes["books"].parent is None
        assert tax.nodes["books"].name == "Books"
        assert tax.nodes["books"].member_count == 2

    def test_add_child_node(self, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f1")
        tax.add_node("books/fantasy", "Fantasy", parent="books",
                     member_embeddings=embs, member_ids=["a"], folder_id="f2")
        assert tax.nodes["books/fantasy"].parent == "books"

    def test_centroid_is_normalised(self, tax):
        embs = np.stack([make_unit([3, 4, 0]), make_unit([0, 4, 3])])
        tax.add_node("x", "X", parent=None, member_embeddings=embs,
                     member_ids=["a", "b"], folder_id="f1")
        centroid = tax.nodes["x"].centroid_array()
        assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5


class TestTreeNavigation:
    def test_children_of_root(self, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f1")
        tax.add_node("books/fantasy", "Fantasy", parent="books",
                     member_embeddings=embs, member_ids=["a"], folder_id="f2")
        tax.add_node("work", "Work", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f3")
        roots = tax.children_of(None)
        assert {n.path for n in roots} == {"books", "work"}
        children = tax.children_of("books")
        assert len(children) == 1
        assert children[0].path == "books/fantasy"

    def test_children_of_leaf_is_empty(self, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f1")
        assert tax.children_of("books") == []


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path_taxonomy, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="fid1", description="Reading")
        tax.add_node("books/fantasy", "Fantasy", parent="books",
                     member_embeddings=embs, member_ids=["a"], folder_id="fid2")
        tax.save()

        loaded = TaxonomyV2.load(path=tmp_path_taxonomy)
        assert "books" in loaded.nodes
        assert "books/fantasy" in loaded.nodes
        assert loaded.nodes["books"].description == "Reading"
        assert loaded.nodes["books/fantasy"].parent == "books"
        centroid = loaded.nodes["books"].centroid_array()
        assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5

    def test_load_missing_file_returns_empty(self, tmp_path_taxonomy):
        loaded = TaxonomyV2.load(path=tmp_path_taxonomy)
        assert loaded.nodes == {}
