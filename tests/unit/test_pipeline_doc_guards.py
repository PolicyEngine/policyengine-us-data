from scripts.guards import pipeline_docs, pydoc_completeness


def test_pipeline_docs_guard_passes():
    assert pipeline_docs.check() == []


def test_pydoc_completeness_guard_passes():
    assert pydoc_completeness.check() == []
