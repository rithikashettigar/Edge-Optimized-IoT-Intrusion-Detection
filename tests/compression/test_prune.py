import torch

from conftest import make_separable_splits as separable_splits
from edge_ids.compression.prune import iterative_prune, neuron_scores, structured_prune
from edge_ids.config import Config
from edge_ids.models import count_parameters
from edge_ids.models.student import StudentMLP


def test_pruning_zero_percent_is_numerically_identical():
    # Guards against row/column transposition in the weight-copy logic: a
    # transposed copy still produces a valid model that quietly computes nonsense.
    torch.manual_seed(0)
    model = StudentMLP(6, hidden=(8, 4)).eval()
    x = torch.randn(5, 6)
    rebuilt = structured_prune(model, amount=0.0).eval()
    assert torch.allclose(model(x), rebuilt(x), atol=1e-6)


def test_pruning_actually_reduces_parameter_count():
    # A masking implementation would leave this unchanged - the whole point of
    # rebuilding rather than zeroing (spec 6).
    model = StudentMLP(6, hidden=(16, 8))
    pruned = structured_prune(model, amount=0.5)
    assert count_parameters(pruned) < count_parameters(model)
    assert pruned.hidden_sizes() == (8, 4)


def test_pruned_model_produces_correct_output_shape():
    pruned = structured_prune(StudentMLP(6, hidden=(16, 8)), amount=0.5).eval()
    assert pruned(torch.randn(3, 6)).shape == (3, 2)


def test_minimum_neurons_floor_prevents_layer_collapse():
    pruned = structured_prune(StudentMLP(6, hidden=(16, 8)), amount=0.99, min_neurons=4)
    assert all(width >= 4 for width in pruned.hidden_sizes())


def test_output_layer_is_never_pruned():
    pruned = structured_prune(StudentMLP(6, hidden=(16, 8)), amount=0.5)
    assert pruned.linear_layers()[-1].out_features == 2


def test_input_width_is_preserved():
    pruned = structured_prune(StudentMLP(6, hidden=(16, 8)), amount=0.5)
    assert pruned.linear_layers()[0].in_features == 6


def test_highest_norm_neurons_are_the_ones_kept():
    torch.manual_seed(0)
    model = StudentMLP(4, hidden=(4, 4))
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].weight[2] = 10.0  # dominant
        model.net[0].weight[1] = 1.0
        model.net[0].weight[3] = 0.5
        # neuron 0 left at zero - genuinely dead
    scores = neuron_scores(model.net[0])
    assert int(scores.argmax()) == 2

    pruned = structured_prune(model, amount=0.5, min_neurons=1)
    assert pruned.hidden_sizes()[0] == 2
    # The two survivors must be neurons 2 and 1, carried over intact.
    kept = pruned.linear_layers()[0].weight.detach()
    assert torch.allclose(kept[1], torch.full((4,), 10.0))
    assert torch.allclose(kept[0], torch.full((4,), 1.0))


def test_surviving_biases_travel_with_their_neurons():
    torch.manual_seed(0)
    model = StudentMLP(4, hidden=(4,))
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].weight[3] = 5.0
        model.net[0].bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    pruned = structured_prune(model, amount=0.75, min_neurons=1)
    assert pruned.linear_layers()[0].bias.detach().tolist() == [4.0]


def test_quantizable_flag_survives_the_rebuild():
    pruned = structured_prune(StudentMLP(6, hidden=(8, 4), quantizable=True), amount=0.5)
    assert pruned.quantizable is True


def _prune_cfg():
    cfg = Config.default()
    cfg.student.hidden, cfg.student.batch_size, cfg.student.lr = [16, 8], 32, 1e-2
    cfg.prune.total_amount, cfg.prune.iterations, cfg.prune.finetune_epochs = 0.5, 2, 8
    return cfg


def test_iterative_pruning_shrinks_the_model_and_keeps_accuracy():
    cfg = _prune_cfg()
    splits = separable_splits()
    model = StudentMLP(6, hidden=(16, 8))
    pruned, history = iterative_prune(model, splits, cfg, teacher=None)
    assert count_parameters(pruned) < count_parameters(model)
    assert history["final_val_f1"] > 0.75


def test_iterative_pruning_approaches_the_configured_total():
    cfg = _prune_cfg()
    model = StudentMLP(6, hidden=(16, 8))
    pruned, _ = iterative_prune(model, separable_splits(), cfg, teacher=None)
    # 50% of hidden neurons removed overall, reached in two ~29% steps.
    assert sum(pruned.hidden_sizes()) <= 0.6 * sum(model.hidden_sizes())


def test_iterative_pruning_leaves_the_original_model_untouched():
    cfg = _prune_cfg()
    model = StudentMLP(6, hidden=(16, 8))
    before = model.net[0].weight.detach().clone()
    iterative_prune(model, separable_splits(), cfg, teacher=None)
    assert torch.allclose(before, model.net[0].weight)
    assert model.hidden_sizes() == (16, 8)
