"""Tests for cluster_cache — hit/miss logic and param-based invalidation."""
import numpy as np
import pytest
from drivesort.cluster_cache import ClusterCache


@pytest.fixture
def cache(tmp_path):
    return ClusterCache(path=tmp_path / "cluster_cache.pkl")


def make_arrays():
    emb_2d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    return emb_2d, labels


class TestClusterCache:
    def test_miss_on_empty(self, cache):
        assert cache.load(["key1", "key2"], {"min_cluster_size": 3}) is None

    def test_save_then_hit(self, cache):
        emb_2d, labels = make_arrays()
        params = {"min_cluster_size": 3, "umap_n_neighbors": 15}
        cache.save(["key1", "key2"], params, emb_2d, labels)
        result = cache.load(["key2", "key1"], params)  # order-independent
        assert result is not None
        np.testing.assert_array_equal(result.embeddings_2d, emb_2d)
        np.testing.assert_array_equal(result.labels, labels)

    def test_miss_on_param_change(self, cache):
        emb_2d, labels = make_arrays()
        params_a = {"min_cluster_size": 3, "umap_n_neighbors": 15}
        params_b = {"min_cluster_size": 5, "umap_n_neighbors": 15}
        cache.save(["key1"], params_a, emb_2d, labels)
        assert cache.load(["key1"], params_b) is None

    def test_miss_on_key_change(self, cache):
        emb_2d, labels = make_arrays()
        params = {"min_cluster_size": 3}
        cache.save(["key1", "key2"], params, emb_2d, labels)
        assert cache.load(["key1", "key3"], params) is None
