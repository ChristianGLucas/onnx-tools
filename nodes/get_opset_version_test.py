from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_custom_domain_model, build_tiny_mlp_model
from nodes.get_opset_version import get_opset_version
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_opset_17():
    result = get_opset_version(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert_ok(result)
    assert result.opset_version == 17
    assert len(result.all_opsets) == 1
    assert result.all_opsets[0].domain == ""
    assert result.all_opsets[0].version == 17


def test_custom_domain_model_reports_both_opsets():
    result = get_opset_version(ax(), OnnxModel(model_data=build_custom_domain_model()))
    assert_ok(result)
    assert result.opset_version == 17
    domains = {op.domain: op.version for op in result.all_opsets}
    assert domains == {"": 17, "my.custom.domain": 1}


def test_malformed_model_is_malformed_model_error():
    result = get_opset_version(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")
