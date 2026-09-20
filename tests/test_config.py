from edge_ids.config import Config


def test_loads_nested_values_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  seed: 7\nteacher:\n  hidden: [8, 4]\n")
    cfg = Config.load(p)
    assert cfg.data.seed == 7
    assert cfg.teacher.hidden == [8, 4]


def test_missing_keys_fall_back_to_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  seed: 7\n")
    cfg = Config.load(p)
    assert cfg.distill.temperature == 4.0
    assert cfg.prune.min_neurons == 4


def test_default_config_matches_spec_values():
    cfg = Config.default()
    assert cfg.teacher.hidden == [512, 256, 128, 64]
    assert cfg.student.hidden == [32, 16]
    assert cfg.distill.alpha == 0.7
    assert cfg.prune.total_amount == 0.5
