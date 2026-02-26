"""Unit tests for node_allocation module (standalone, no pytest required)."""
import os
import sys
import warnings
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from node_allocation import (
    NodeAllocation,
    ExternalJudgeConfig,
    parse_node_allocation,
    derive_from_legacy_config,
    validate_cli_matches_config,
)


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run(self, name, test_func):
        try:
            test_func()
            self.passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, traceback.format_exc()))
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    
    def summary(self):
        print(f"\n{'='*50}")
        print(f"Results: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, err in self.errors:
                print(f"  - {name}: {err[:100]}")
        return self.failed == 0


def test_basic_creation():
    alloc = NodeAllocation(total=2, roles={"rollout": 1, "trainer": 1})
    assert alloc.total == 2
    assert alloc.roles["rollout"] == 1
    assert alloc.roles["trainer"] == 1


def test_validate_success():
    alloc = NodeAllocation(total=3, roles={"rollout": 1, "trainer": 2})
    alloc.validate()


def test_validate_sum_mismatch():
    alloc = NodeAllocation(total=3, roles={"rollout": 1, "trainer": 1})
    try:
        alloc.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "sum" in str(e).lower()


def test_validate_missing_rollout():
    alloc = NodeAllocation(total=2, roles={"trainer": 2})
    try:
        alloc.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "rollout" in str(e).lower()


def test_validate_missing_trainer():
    alloc = NodeAllocation(total=2, roles={"rollout": 2})
    try:
        alloc.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "trainer" in str(e).lower()


def test_single_node_roles_tags():
    alloc = NodeAllocation(total=2, roles={"rollout": 1, "trainer": 1})
    tags = alloc.get_node_tags()
    assert tags == ["rollout", "trainer"], f"Got {tags}"


def test_multi_node_roles_tags():
    alloc = NodeAllocation(total=4, roles={"rollout": 2, "trainer": 2})
    tags = alloc.get_node_tags()
    assert tags == ["rollout_0", "rollout_1", "trainer_0", "trainer_1"], f"Got {tags}"


def test_asymmetric_roles_tags():
    alloc = NodeAllocation(total=3, roles={"rollout": 1, "trainer": 2})
    tags = alloc.get_node_tags()
    assert tags == ["rollout", "trainer_0", "trainer_1"], f"Got {tags}"


def test_get_role_for_node_index_simple():
    alloc = NodeAllocation(total=2, roles={"rollout": 1, "trainer": 1})
    assert alloc.get_role_for_node_index(0) == ("rollout", 0)
    assert alloc.get_role_for_node_index(1) == ("trainer", 0)


def test_get_role_for_node_index_multi():
    alloc = NodeAllocation(total=4, roles={"rollout": 2, "trainer": 2})
    assert alloc.get_role_for_node_index(0) == ("rollout", 0)
    assert alloc.get_role_for_node_index(1) == ("rollout", 1)
    assert alloc.get_role_for_node_index(2) == ("trainer", 0)
    assert alloc.get_role_for_node_index(3) == ("trainer", 1)


def test_get_tag_for_node_index():
    alloc = NodeAllocation(total=4, roles={"rollout": 2, "trainer": 2})
    assert alloc.get_tag_for_node_index(0) == "rollout_0"
    assert alloc.get_tag_for_node_index(1) == "rollout_1"
    assert alloc.get_tag_for_node_index(2) == "trainer_0"
    assert alloc.get_tag_for_node_index(3) == "trainer_1"


def test_external_judge_config():
    config = ExternalJudgeConfig()
    assert config.enabled is False
    assert config.experiment_name is None
    
    config = ExternalJudgeConfig(enabled=True, experiment_name="judge-exp")
    assert config.enabled is True
    assert config.experiment_name == "judge-exp"


def test_node_allocation_with_judge_dict():
    alloc = NodeAllocation(
        total=2,
        roles={"rollout": 1, "trainer": 1},
        external_judge={"enabled": True, "experiment_name": "test"}
    )
    assert isinstance(alloc.external_judge, ExternalJudgeConfig)
    assert alloc.external_judge.enabled is True


def test_parse_from_dict_with_nodes():
    config = {
        "nodes": {
            "total": 3,
            "roles": {"rollout": 1, "trainer": 2},
        }
    }
    alloc = parse_node_allocation(config)
    assert alloc.total == 3
    assert alloc.roles["rollout"] == 1
    assert alloc.roles["trainer"] == 2


def test_parse_from_dict_without_nodes():
    config = {"cluster": {"n_nodes": 2}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        alloc = parse_node_allocation(config)
        assert len(w) >= 1, "Should have triggered deprecation warning"
    assert alloc.total == 2


def test_derive_legacy_2_nodes():
    config = {"cluster": {"n_nodes": 2}}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        alloc = derive_from_legacy_config(config)
    assert alloc.total == 2
    assert alloc.roles["rollout"] == 1
    assert alloc.roles["trainer"] == 1


def test_derive_legacy_4_nodes():
    config = {"cluster": {"n_nodes": 4}}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        alloc = derive_from_legacy_config(config)
    assert alloc.total == 4
    assert alloc.roles["rollout"] == 1
    assert alloc.roles["trainer"] == 3


def test_validate_cli_matches():
    alloc = NodeAllocation(total=2, roles={"rollout": 1, "trainer": 1})
    validate_cli_matches_config(2, alloc)


def test_validate_cli_mismatch():
    alloc = NodeAllocation(total=2, roles={"rollout": 1, "trainer": 1})
    try:
        validate_cli_matches_config(3, alloc)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def main():
    print("=" * 50)
    print("Node Allocation Unit Tests")
    print("=" * 50)
    
    runner = TestRunner()
    
    print("\nNodeAllocation dataclass:")
    runner.run("basic_creation", test_basic_creation)
    runner.run("validate_success", test_validate_success)
    runner.run("validate_sum_mismatch", test_validate_sum_mismatch)
    runner.run("validate_missing_rollout", test_validate_missing_rollout)
    runner.run("validate_missing_trainer", test_validate_missing_trainer)
    
    print("\nNode tags:")
    runner.run("single_node_roles_tags", test_single_node_roles_tags)
    runner.run("multi_node_roles_tags", test_multi_node_roles_tags)
    runner.run("asymmetric_roles_tags", test_asymmetric_roles_tags)
    runner.run("get_role_for_node_index_simple", test_get_role_for_node_index_simple)
    runner.run("get_role_for_node_index_multi", test_get_role_for_node_index_multi)
    runner.run("get_tag_for_node_index", test_get_tag_for_node_index)
    
    print("\nExternalJudgeConfig:")
    runner.run("external_judge_config", test_external_judge_config)
    runner.run("node_allocation_with_judge_dict", test_node_allocation_with_judge_dict)
    
    print("\nparse_node_allocation:")
    runner.run("parse_from_dict_with_nodes", test_parse_from_dict_with_nodes)
    runner.run("parse_from_dict_without_nodes", test_parse_from_dict_without_nodes)
    
    print("\nderive_from_legacy_config:")
    runner.run("derive_legacy_2_nodes", test_derive_legacy_2_nodes)
    runner.run("derive_legacy_4_nodes", test_derive_legacy_4_nodes)
    
    print("\nvalidate_cli_matches_config:")
    runner.run("validate_cli_matches", test_validate_cli_matches)
    runner.run("validate_cli_mismatch", test_validate_cli_mismatch)
    
    success = runner.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
