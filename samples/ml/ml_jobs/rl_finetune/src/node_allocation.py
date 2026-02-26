"""
Unified Node Allocation Schema for Multi-Node RL Training.

This module provides:
1. NodeAllocation dataclass for declaring node roles
2. Parsing utilities for YAML config
3. Backward compatibility for legacy configs (without nodes section)
4. Validation to ensure roles sum to total nodes

Example config:
    nodes:
      total: 2
      roles:
        rollout: 1
        trainer: 1
      external_judge:
        enabled: false
        experiment_name: null
        trial_name: null
"""
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExternalJudgeConfig:
    """Configuration for external judge server (runs as separate job)."""
    enabled: bool = False
    experiment_name: Optional[str] = None
    trial_name: Optional[str] = None
    config: Optional[str] = None


@dataclass
class NodeAllocation:
    """Unified node allocation configuration."""
    total: int
    roles: dict[str, int] = field(default_factory=lambda: {"rollout": 1, "trainer": 1})
    external_judge: ExternalJudgeConfig = field(default_factory=ExternalJudgeConfig)
    
    def __post_init__(self):
        if isinstance(self.external_judge, dict):
            self.external_judge = ExternalJudgeConfig(**self.external_judge)
    
    def validate(self):
        """Validate that roles sum to total nodes."""
        roles_sum = sum(self.roles.values())
        if roles_sum != self.total:
            raise ValueError(
                f"Node roles must sum to total. "
                f"Got roles sum={roles_sum} but total={self.total}. "
                f"Roles: {self.roles}"
            )
        
        for role, count in self.roles.items():
            if count < 0:
                raise ValueError(f"Role '{role}' has negative count: {count}")
        
        if "rollout" not in self.roles:
            raise ValueError("Node allocation must include 'rollout' role")
        if "trainer" not in self.roles:
            raise ValueError("Node allocation must include 'trainer' role")
    
    def get_node_tags(self) -> list[str]:
        """Generate node tag names based on role counts.
        
        Returns:
            List of tag names like ['rollout', 'trainer'] for single-node roles
            or ['rollout_0', 'rollout_1', 'trainer_0', 'trainer_1'] for multi-node.
        """
        tags = []
        for role, count in self.roles.items():
            if count == 1:
                tags.append(role)
            else:
                for i in range(count):
                    tags.append(f"{role}_{i}")
        return tags
    
    def get_role_for_node_index(self, node_idx: int) -> tuple[str, int]:
        """Get the role and sub-index for a given node index.
        
        Args:
            node_idx: Global node index (0-based, sorted by IP)
            
        Returns:
            Tuple of (role_name, sub_index within role)
        """
        current_idx = 0
        for role, count in self.roles.items():
            if node_idx < current_idx + count:
                sub_idx = node_idx - current_idx
                return role, sub_idx
            current_idx += count
        raise ValueError(f"Node index {node_idx} exceeds total nodes {self.total}")
    
    def get_tag_for_node_index(self, node_idx: int) -> str:
        """Get the node tag for a given node index."""
        role, sub_idx = self.get_role_for_node_index(node_idx)
        count = self.roles[role]
        if count == 1:
            return role
        return f"{role}_{sub_idx}"


def parse_node_allocation(config) -> NodeAllocation:
    """Parse NodeAllocation from config object or dict.
    
    Args:
        config: Config object with optional 'nodes' attribute, or dict
        
    Returns:
        NodeAllocation instance
    """
    if hasattr(config, 'nodes') and config.nodes is not None:
        nodes_dict = config.nodes if isinstance(config.nodes, dict) else vars(config.nodes)
        return NodeAllocation(
            total=nodes_dict.get('total', 2),
            roles=nodes_dict.get('roles', {"rollout": 1, "trainer": 1}),
            external_judge=nodes_dict.get('external_judge', {}),
        )
    
    if isinstance(config, dict) and 'nodes' in config:
        nodes_dict = config['nodes']
        return NodeAllocation(
            total=nodes_dict.get('total', 2),
            roles=nodes_dict.get('roles', {"rollout": 1, "trainer": 1}),
            external_judge=nodes_dict.get('external_judge', {}),
        )
    
    return derive_from_legacy_config(config)


def derive_from_legacy_config(config) -> NodeAllocation:
    """Derive node allocation from legacy config (without nodes section).
    
    Uses allocation_mode and cluster.n_nodes to infer role distribution.
    Emits deprecation warning.
    
    Args:
        config: Legacy config object
        
    Returns:
        NodeAllocation with inferred roles
    """
    warnings.warn(
        "Config missing 'nodes' section. Deriving node allocation from "
        "allocation_mode and cluster.n_nodes. Please update your config to use "
        "the new 'nodes' section for explicit control.",
        DeprecationWarning,
        stacklevel=3
    )
    
    n_nodes = 2
    if hasattr(config, 'cluster'):
        cluster = config.cluster if isinstance(config.cluster, dict) else vars(config.cluster)
        n_nodes = cluster.get('n_nodes', 2)
    elif isinstance(config, dict) and 'cluster' in config:
        n_nodes = config['cluster'].get('n_nodes', 2)
    
    use_3node_env = os.environ.get("AREAL_3NODE_MODE") == "1"
    if use_3node_env:
        warnings.warn(
            "AREAL_3NODE_MODE environment variable is deprecated. "
            "Use 'nodes.external_judge.enabled: true' in config instead.",
            DeprecationWarning,
            stacklevel=3
        )
    
    allocation_mode_str = ""
    if hasattr(config, 'allocation_mode'):
        allocation_mode_str = config.allocation_mode
    elif isinstance(config, dict):
        allocation_mode_str = config.get('allocation_mode', '')
    
    n_rollout_nodes = 1
    n_trainer_nodes = max(1, n_nodes - n_rollout_nodes)
    
    if '+' in allocation_mode_str:
        n_trainer_nodes = n_nodes - n_rollout_nodes
    
    if n_rollout_nodes + n_trainer_nodes != n_nodes:
        n_trainer_nodes = n_nodes - n_rollout_nodes
    
    return NodeAllocation(
        total=n_nodes,
        roles={
            "rollout": n_rollout_nodes,
            "trainer": n_trainer_nodes,
        },
        external_judge=ExternalJudgeConfig(enabled=use_3node_env),
    )


def validate_cli_matches_config(cli_num_nodes: int, node_allocation: NodeAllocation):
    """Validate that CLI --num-nodes matches config.nodes.total.
    
    Args:
        cli_num_nodes: Value from --num-nodes CLI argument
        node_allocation: Parsed NodeAllocation from config
        
    Raises:
        ValueError if mismatch
    """
    if cli_num_nodes != node_allocation.total:
        raise ValueError(
            f"CLI --num-nodes={cli_num_nodes} does not match "
            f"config nodes.total={node_allocation.total}. "
            f"Please ensure they are consistent."
        )
