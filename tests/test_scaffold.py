"""Scaffold validation and specification tagging tests."""

from eatsched.utils.config import get_project_root, load_yaml


def test_docs_exist_and_tagged():
    """Verify that SPEC.md and ASSUMPTIONS.md exist and contain required tags."""
    root = get_project_root()
    spec_path = root / "docs" / "SPEC.md"
    assumptions_path = root / "docs" / "ASSUMPTIONS.md"
    results_path = root / "docs" / "RESULTS.md"
    audit_path = root / "docs" / "AUDIT.md"

    assert spec_path.exists(), "docs/SPEC.md missing"
    assert assumptions_path.exists(), "docs/ASSUMPTIONS.md missing"
    assert results_path.exists(), "docs/RESULTS.md missing"
    assert audit_path.exists(), "docs/AUDIT.md missing"

    spec_text = spec_path.read_text(encoding="utf-8")
    assert "[PAPER]" in spec_text, "SPEC.md must contain [PAPER] tags"
    assert "[ABSTRACT]" in spec_text, "SPEC.md must contain [ABSTRACT] tags"
    assert "[ASSUMPTION]" in spec_text, "SPEC.md must contain [ASSUMPTION] tags"

    assumptions_text = assumptions_path.read_text(encoding="utf-8")
    assert "[ASSUMPTION]" in assumptions_text, "ASSUMPTIONS.md must contain [ASSUMPTION] tags"
    # Verify core assumption IDs are registered
    for i in range(1, 13):
        asm_id = f"ASM-{i:02d}"
        assert asm_id in assumptions_text, f"Assumption {asm_id} missing in ASSUMPTIONS.md"


def test_cluster_configs():
    """Verify cluster configuration files are valid and well-structured."""
    root = get_project_root()
    cluster_dir = root / "configs" / "clusters"
    expected_clusters = ["edge32.yaml", "edge48.yaml", "edge64.yaml", "cloud_edge.yaml"]
    for filename in expected_clusters:
        filepath = cluster_dir / filename
        assert filepath.exists(), f"Cluster config {filename} missing"
        cfg = load_yaml(filepath)
        assert "name" in cfg
        assert "num_servers" in cfg
        assert "arrival_rate" in cfg
        assert cfg["num_servers"] > 0
        assert cfg["arrival_rate"] > 0


def test_workload_configs():
    """Verify workload profile configuration files."""
    root = get_project_root()
    workload_dir = root / "configs" / "workloads"
    expected_workloads = [
        "aigc_sd.yaml",
        "ml_inference.yaml",
        "video_analytics.yaml",
        "cpu_batch.yaml",
    ]
    for filename in expected_workloads:
        filepath = workload_dir / filename
        assert filepath.exists(), f"Workload config {filename} missing"
        cfg = load_yaml(filepath)
        assert "name" in cfg
        assert "type" in cfg


def test_agent_configs():
    """Verify agent hyperparameters configuration files."""
    root = get_project_root()
    agents_dir = root / "configs" / "agents"
    expected_agents = [
        "eat.yaml",
        "d2sac.yaml",
        "dppo.yaml",
        "sac.yaml",
        "ppo.yaml",
        "td3.yaml",
        "ddpg.yaml",
        "heuristics.yaml",
    ]
    for filename in expected_agents:
        filepath = agents_dir / filename
        assert filepath.exists(), f"Agent config {filename} missing"
        cfg = load_yaml(filepath)
        assert cfg is not None


def test_package_imports():
    """Verify eatsched root and subpackages import correctly."""
    import eatsched
    import eatsched.utils.config

    assert eatsched.__version__ == "0.1.0"
