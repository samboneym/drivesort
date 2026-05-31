"""
Tests for drivesort/taxonomy.py

Coverage:
- add_category() normalises centroids correctly
- classify() finds the nearest category
- classify() marks files as novel when distance exceeds threshold
- classify() returns runner_up correctly
- classify() handles empty taxonomy
- confirm() updates centroids incrementally
- confirm() re-normalises after update
- log_novel_file() deduplicates
- save() / load round-trip preserves data
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from drivesort.taxonomy import Taxonomy, CategoryEntry, ClassificationResult


@pytest.fixture
def temp_taxonomy_path():
    """Temporary directory for taxonomy.json during tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "taxonomy.json"


@pytest.fixture
def empty_taxonomy(temp_taxonomy_path):
    """Fresh empty taxonomy."""
    return Taxonomy(path=temp_taxonomy_path, novelty_threshold=0.42)


def make_unit_vector(values):
    """Create a normalised unit vector from values."""
    v = np.array(values, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


class TestAddCategory:
    """Test category creation and centroid calculation."""

    def test_add_category_normalises_centroid(self, empty_taxonomy):
        """Centroid should be L2-normalised after add_category()."""
        # Create three embeddings (not normalised)
        embs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        empty_taxonomy.add_category(
            name="Test",
            description="Test category",
            folder_id="test_folder_id",
            member_embeddings=embs,
            member_ids=["f1", "f2", "f3"],
        )

        entry = empty_taxonomy.categories["Test"]
        centroid = entry.centroid_array()
        norm = np.linalg.norm(centroid)
        assert np.isclose(norm, 1.0), "Centroid should be unit vector"

    def test_add_category_mean_embedding(self, empty_taxonomy):
        """Centroid should be mean of member embeddings (before normalisation)."""
        embs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        empty_taxonomy.add_category(
            name="Test",
            description="Test",
            folder_id="fold1",
            member_embeddings=embs,
            member_ids=["f1", "f2"],
        )

        entry = empty_taxonomy.categories["Test"]
        mean = np.mean(embs, axis=0)
        mean_normalised = mean / (np.linalg.norm(mean) + 1e-8)

        assert np.allclose(entry.centroid_array(), mean_normalised)

    def test_add_category_stores_metadata(self, empty_taxonomy):
        """add_category() should store name, description, folder_id, member_count."""
        embs = np.ones((5, 384), dtype=np.float32)
        ids = [f"f{i}" for i in range(5)]

        empty_taxonomy.add_category(
            name="MyCategory",
            description="My test category",
            folder_id="folder_xyz",
            member_embeddings=embs,
            member_ids=ids,
        )

        entry = empty_taxonomy.categories["MyCategory"]
        assert entry.name == "MyCategory"
        assert entry.description == "My test category"
        assert entry.folder_id == "folder_xyz"
        assert entry.member_count == 5
        assert entry.member_ids == ids


class TestClassify:
    """Test classification logic."""

    def test_classify_empty_taxonomy(self, empty_taxonomy):
        """classify() on empty taxonomy should return is_novel=True, no category."""
        emb = make_unit_vector([1, 0, 0, 0])
        result = empty_taxonomy.classify(emb, "f1", "test.txt")

        assert result.is_novel is True
        assert result.category is None
        assert result.confidence == 0.0
        assert result.distance == 1.0

    def test_classify_finds_nearest(self, empty_taxonomy):
        """classify() should return the nearest category by cosine distance."""
        # Two categories
        cat1_embs = np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]], dtype=np.float32)
        cat2_embs = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.1]], dtype=np.float32)

        empty_taxonomy.add_category("Cat1", "First", "f1", cat1_embs, ["id1", "id2"])
        empty_taxonomy.add_category("Cat2", "Second", "f2", cat2_embs, ["id3", "id4"])

        # Query closer to Cat1
        query = make_unit_vector([0.95, 0.05, 0.0])
        result = empty_taxonomy.classify(query, "new_file", "test.txt")

        assert result.category == "Cat1"
        assert result.distance < 0.5  # Should be close

    def test_classify_marks_novel_when_far(self, empty_taxonomy):
        """classify() should mark is_novel=True when distance > novelty_threshold."""
        cat_embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", cat_embs, ["id1"])

        # Query far away (opposite direction)
        query = make_unit_vector([-1.0, 0.0, 0.0])
        result = empty_taxonomy.classify(query, "far_file", "test.txt")

        # Cosine distance from [1,0,0] to [-1,0,0] = 2.0 (opposite)
        # But normalised it's 1 - dot([1,0,0], [-1,0,0]) = 1 - (-1) = 2.0
        # Actually both are normalised, so distance = 1 - (-1) = 2.0 (clamped to >1 is OOD)
        assert result.is_novel is True

    def test_classify_not_novel_when_close(self, empty_taxonomy):
        """classify() should mark is_novel=False when distance < novelty_threshold."""
        cat_embs = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", cat_embs, ["id1", "id2"])

        # Query close to Cat1
        query = make_unit_vector([0.95, 0.05, 0.0])
        result = empty_taxonomy.classify(query, "close_file", "test.txt")

        assert result.is_novel is False
        assert result.category == "Cat1"

    def test_classify_returns_runner_up(self, empty_taxonomy):
        """classify() should return the second-nearest category as runner_up."""
        cat1_embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        cat2_embs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        empty_taxonomy.add_category("Cat1", "First", "f1", cat1_embs, ["id1"])
        empty_taxonomy.add_category("Cat2", "Second", "f2", cat2_embs, ["id2"])

        # Query between them, closer to Cat1
        query = make_unit_vector([0.7, 0.3, 0.0])
        result = empty_taxonomy.classify(query, "mid_file", "test.txt")

        assert result.category == "Cat1"
        assert result.runner_up == "Cat2"
        assert result.runner_up_confidence > 0.0

    def test_classify_confidence_is_one_minus_distance(self, empty_taxonomy):
        """confidence should equal 1.0 - distance for best match."""
        cat_embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", cat_embs, ["id1"])

        query = make_unit_vector([1.0, 0.0, 0.0])
        result = empty_taxonomy.classify(query, "exact_file", "test.txt")

        expected_confidence = max(0.0, 1.0 - result.distance)
        assert np.isclose(result.confidence, expected_confidence)


class TestConfirm:
    """Test centroid updating via confirm()."""

    def test_confirm_updates_centroid(self, empty_taxonomy):
        """confirm() should update the centroid incrementally."""
        init_embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", init_embs, ["id1"])

        old_centroid = empty_taxonomy.categories["Cat1"].centroid_array().copy()

        new_emb = make_unit_vector([0.0, 1.0, 0.0])
        empty_taxonomy.confirm("Cat1", "id2", new_emb)

        new_centroid = empty_taxonomy.categories["Cat1"].centroid_array()

        # Should be between old centroid and new embedding
        assert not np.allclose(new_centroid, old_centroid)
        assert empty_taxonomy.categories["Cat1"].member_count == 2

    def test_confirm_normalises_centroid(self, empty_taxonomy):
        """confirm() should L2-normalise the centroid after update."""
        init_embs = np.array([[1.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", init_embs, ["id1"])

        new_emb = make_unit_vector([1.0, 0.0])
        empty_taxonomy.confirm("Cat1", "id2", new_emb)

        centroid = empty_taxonomy.categories["Cat1"].centroid_array()
        norm = np.linalg.norm(centroid)
        assert np.isclose(norm, 1.0), "Centroid should remain unit vector after confirm()"

    def test_confirm_deduplicates_member_ids(self, empty_taxonomy):
        """confirm() should not duplicate file IDs in member_ids."""
        init_embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", init_embs, ["id1"])

        new_emb = make_unit_vector([0.0, 1.0, 0.0])
        empty_taxonomy.confirm("Cat1", "id1", new_emb)  # Same ID

        assert empty_taxonomy.categories["Cat1"].member_ids.count("id1") == 1

    def test_confirm_incremental_mean(self, empty_taxonomy):
        """confirm() should compute correct running mean."""
        # Start with 2 files
        init_embs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", init_embs, ["id1", "id2"])

        old_centroid = empty_taxonomy.categories["Cat1"].centroid_array().copy()

        # Add a third file
        third_emb = make_unit_vector([1.0, 1.0])
        empty_taxonomy.confirm("Cat1", "id3", third_emb)

        new_centroid = empty_taxonomy.categories["Cat1"].centroid_array()

        # Expected: (old_centroid * 2 + third_emb) / 3, then normalised
        expected = (old_centroid * 2 + third_emb) / 3
        expected = expected / (np.linalg.norm(expected) + 1e-8)

        assert np.allclose(new_centroid, expected, atol=1e-5)


class TestNovelLog:
    """Test novel file logging and deduplication."""

    def test_log_novel_file_creates_log(self, empty_taxonomy):
        """log_novel_file() should create novel_files.json."""
        from pathlib import Path
        novel_path = Path("data/novel_files.json")

        # Clean up first
        if novel_path.exists():
            novel_path.unlink()

        emb = make_unit_vector([1, 0, 0, 0])
        empty_taxonomy.log_novel_file("novel_f1", "unknown.txt", emb)

        assert novel_path.exists()

        # Cleanup
        if novel_path.exists():
            novel_path.unlink()

    def test_log_novel_file_deduplicates(self, empty_taxonomy):
        """log_novel_file() should not create duplicate entries for same file ID."""
        from pathlib import Path
        novel_path = Path("data/novel_files.json")

        # Clean up first
        if novel_path.exists():
            novel_path.unlink()

        emb1 = make_unit_vector([1, 0, 0, 0])
        emb2 = make_unit_vector([0, 1, 0, 0])

        empty_taxonomy.log_novel_file("f1", "file.txt", emb1)
        empty_taxonomy.log_novel_file("f1", "file.txt", emb2)  # Same ID

        records, _ = empty_taxonomy.load_novel_files()
        assert len(records) == 1

        # Cleanup
        if novel_path.exists():
            novel_path.unlink()

    def test_load_novel_files_returns_empty_when_missing(self, empty_taxonomy):
        """load_novel_files() should return empty structures if log doesn't exist."""
        from pathlib import Path
        novel_path = Path("data/novel_files.json")

        # Clean up first to ensure it doesn't exist
        if novel_path.exists():
            novel_path.unlink()

        records, embeddings = empty_taxonomy.load_novel_files()

        assert records == []
        assert embeddings.shape == (0,)


class TestPersistence:
    """Test save() and load round-trip."""

    def test_save_and_load_preserves_taxonomy(self, empty_taxonomy, temp_taxonomy_path):
        """save() then load should produce identical taxonomy."""
        # Build taxonomy
        cat1_embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        cat2_embs = np.array([[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]], dtype=np.float32)

        empty_taxonomy.add_category("Cat1", "First category", "fold1", cat1_embs, ["id1"])
        empty_taxonomy.add_category("Cat2", "Second category", "fold2", cat2_embs, ["id2", "id3"])

        empty_taxonomy.save()

        # Load from disk
        loaded = Taxonomy(path=temp_taxonomy_path)

        # Should have same categories
        assert set(loaded.category_names) == {"Cat1", "Cat2"}

        # Check metadata
        assert loaded.categories["Cat1"].description == "First category"
        assert loaded.categories["Cat2"].member_count == 2

        # Check centroids are preserved
        assert np.allclose(
            loaded.categories["Cat1"].centroid_array(),
            empty_taxonomy.categories["Cat1"].centroid_array(),
        )

    def test_taxonomy_initialization_with_nonexistent_file(self, temp_taxonomy_path):
        """Taxonomy() should handle missing file gracefully."""
        # File doesn't exist yet
        assert not temp_taxonomy_path.exists()

        tax = Taxonomy(path=temp_taxonomy_path)
        assert tax.is_empty()

    def test_is_empty_true_on_no_categories(self, empty_taxonomy):
        """is_empty() should return True when no categories exist."""
        assert empty_taxonomy.is_empty()

    def test_is_empty_false_after_add(self, empty_taxonomy):
        """is_empty() should return False after add_category()."""
        embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "Test", "f1", embs, ["id1"])
        assert not empty_taxonomy.is_empty()


class TestCategoryManagement:
    """Test category manipulation functions."""

    def test_rename_category(self, empty_taxonomy):
        """rename_category() should update the name."""
        embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("OldName", "Test", "f1", embs, ["id1"])

        empty_taxonomy.rename_category("OldName", "NewName")

        assert "OldName" not in empty_taxonomy.categories
        assert "NewName" in empty_taxonomy.categories
        assert empty_taxonomy.categories["NewName"].name == "NewName"

    def test_remove_category(self, empty_taxonomy):
        """remove_category() should delete the category."""
        embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_taxonomy.add_category("ToDelete", "Test", "f1", embs, ["id1"])

        empty_taxonomy.remove_category("ToDelete")

        assert "ToDelete" not in empty_taxonomy.categories

    def test_category_names_property(self, empty_taxonomy):
        """category_names should return list of all category names."""
        embs = np.ones((2, 3), dtype=np.float32)
        empty_taxonomy.add_category("Cat1", "First", "f1", embs, ["id1"])
        empty_taxonomy.add_category("Cat2", "Second", "f2", embs, ["id2"])

        names = empty_taxonomy.category_names
        assert set(names) == {"Cat1", "Cat2"}
