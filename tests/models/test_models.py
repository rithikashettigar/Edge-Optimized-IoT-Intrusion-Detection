import torch

from edge_ids.models import count_parameters
from edge_ids.models.student import StudentMLP
from edge_ids.models.teacher import TeacherMLP


def test_teacher_forward_shape():
    model = TeacherMLP(10, hidden=(8, 4)).eval()
    assert model(torch.randn(5, 10)).shape == (5, 2)


def test_student_forward_shape():
    model = StudentMLP(10, hidden=(6, 3)).eval()
    assert model(torch.randn(5, 10)).shape == (5, 2)


def test_teacher_is_massively_larger_than_student():
    assert count_parameters(TeacherMLP(70)) > 20 * count_parameters(StudentMLP(70))


def test_student_has_no_batchnorm():
    # BatchNorm would need fusion before static quantization and would have to be
    # sliced in step with the pruning rebuild (spec 4.2).
    assert not any(
        isinstance(m, torch.nn.BatchNorm1d) for m in StudentMLP(70).modules()
    )


def test_teacher_does_use_batchnorm():
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in TeacherMLP(70).modules())


def test_student_reports_hidden_sizes():
    assert StudentMLP(10, hidden=(6, 3)).hidden_sizes() == (6, 3)


def test_student_linear_layers_ordered_input_to_output():
    layers = StudentMLP(10, hidden=(6, 3)).linear_layers()
    assert [layer.out_features for layer in layers] == [6, 3, 2]


def test_quantizable_student_runs_in_float_before_conversion():
    model = StudentMLP(10, hidden=(6, 3), quantizable=True).eval()
    assert model(torch.randn(4, 10)).shape == (4, 2)


def test_student_accepts_batch_of_one():
    # Single-sample inference is the deployment case; it must not need batch statistics.
    assert StudentMLP(10, hidden=(6, 3)).eval()(torch.randn(1, 10)).shape == (1, 2)


def test_default_shapes_match_the_spec():
    assert TeacherMLP(70).hidden == (512, 256, 128, 64)
    assert StudentMLP(70).hidden_sizes() == (32, 16)
