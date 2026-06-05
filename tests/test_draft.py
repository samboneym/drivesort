"""Tests for draft persistence — save, load, clear, auto-save."""
import pytest
from drivesort.draft import DraftManager, DraftState, StagedChange, UserDecision


@pytest.fixture
def mgr(tmp_path):
    return DraftManager(path=tmp_path / "draft.json")


def make_state() -> DraftState:
    return DraftState(
        taxonomy_nodes={"books": {"path": "books", "name": "Books"}},
        staged_changes=[
            StagedChange(file_id="f1", file_name="Dune.pdf",
                         current_path=None, proposed_path="books")
        ],
        user_decisions=[
            UserDecision(file_id="f1", action="assign",
                         path="books", timestamp="2026-06-05T10:00:00Z")
        ],
    )


class TestDraftManager:
    def test_exists_false_initially(self, mgr):
        assert mgr.exists() is False

    def test_save_load_roundtrip(self, mgr):
        state = make_state()
        mgr.save(state)
        assert mgr.exists() is True
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.taxonomy_nodes == state.taxonomy_nodes
        assert len(loaded.staged_changes) == 1
        assert loaded.staged_changes[0].file_id == "f1"
        assert loaded.staged_changes[0].proposed_path == "books"
        assert len(loaded.user_decisions) == 1

    def test_clear_removes_file(self, mgr):
        mgr.save(make_state())
        mgr.clear()
        assert mgr.exists() is False
        assert mgr.load() is None

    def test_saved_at_is_populated(self, mgr):
        mgr.save(make_state())
        loaded = mgr.load()
        assert loaded.saved_at != ""
