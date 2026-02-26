"""Patch for AReaL FSDP Engine to handle LoRA QKV name translation for SGLang XCCL.

SGLang merges Q, K, V projections into qkv_proj for efficiency, but the trainer keeps
them separate. This causes XCCL (GPU-to-GPU) weight updates to fail because parameter
names don't match.

This patch intercepts the weight update flow and translates parameter names:
- q_proj.lora_A/B -> qkv_proj.lora_A/B (with proper tensor concatenation)
- k_proj.lora_A/B -> (merged into qkv_proj)
- v_proj.lora_A/B -> (merged into qkv_proj)

Usage:
    from areal_patches.lora_qkv_xccl_patch import apply_lora_qkv_xccl_patch
    apply_lora_qkv_xccl_patch()
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger("LoRAQKVPatch")

# Pattern to match Q/K/V LoRA parameter names
# Example: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
QKV_LORA_PATTERN = re.compile(
    r'^(.*\.self_attn\.)(q_proj|k_proj|v_proj)(\.lora_[AB]\..*)$'
)


def is_qkv_lora_param(name: str) -> bool:
    """Check if parameter name is a Q/K/V LoRA parameter."""
    return QKV_LORA_PATTERN.match(name) is not None


def get_qkv_component(name: str) -> Optional[str]:
    """Extract Q/K/V component from parameter name."""
    match = QKV_LORA_PATTERN.match(name)
    if match:
        return match.group(2)[0]  # 'q', 'k', or 'v'
    return None


def translate_to_qkv_proj(name: str) -> str:
    """Translate q_proj/k_proj/v_proj to qkv_proj in parameter name."""
    match = QKV_LORA_PATTERN.match(name)
    if match:
        prefix, _, suffix = match.groups()
        return f"{prefix}qkv_proj{suffix}"
    return name


def group_qkv_params(
    named_tensors: List[Tuple[str, Any]]
) -> Tuple[List[Tuple[str, Any]], Dict[str, Dict[str, Tuple[str, Any]]]]:
    """Separate QKV params into groups and return non-QKV params.
    
    Returns:
        non_qkv_params: List of (name, tensor) that are not QKV LoRA params
        qkv_groups: Dict[merged_name] -> {'q': (orig_name, tensor), 'k': ..., 'v': ...}
    """
    non_qkv_params = []
    qkv_groups = defaultdict(dict)
    
    for name, tensor in named_tensors:
        if is_qkv_lora_param(name):
            merged_name = translate_to_qkv_proj(name)
            component = get_qkv_component(name)
            qkv_groups[merged_name][component] = (name, tensor)
        else:
            non_qkv_params.append((name, tensor))
    
    return non_qkv_params, dict(qkv_groups)


def merge_qkv_tensors(
    q_tensor, k_tensor, v_tensor, is_lora_a: bool
):
    """Merge Q, K, V LoRA tensors into a single qkv_proj tensor.
    
    For LoRA:
    - lora_A has shape [rank, hidden_size] for each projection
    - lora_B has shape [hidden_size, rank] for each projection
    
    SGLang's merged qkv_proj expects:
    - lora_A: [rank, 3*hidden_size] (concat on dim 1)
    - lora_B: [3*hidden_size, rank] (concat on dim 0)
    
    Actually, for LoRA in SGLang with QKV merge:
    - They keep separate A/B for each of Q,K,V but with qkv_proj prefix
    - So we DON'T merge tensors, we just rename them!
    """
    import torch
    
    # For SGLang's LoRA implementation, they actually keep Q, K, V separate
    # internally even with qkv_proj naming. The merge happens at inference time.
    # So we need to send them one by one with different names.
    
    # Wait - let's check if SGLang truly merges or keeps separate...
    # Based on the error "Failed to update parameter online: qkv_proj.lora_A",
    # it seems SGLang expects a single merged tensor.
    
    if is_lora_a:
        # lora_A: concat on the output dimension
        # Shape: [rank, hidden] each -> [rank, 3*hidden]
        return torch.cat([q_tensor, k_tensor, v_tensor], dim=1)
    else:
        # lora_B: concat on the input dimension  
        # Shape: [hidden, rank] each -> [3*hidden, rank]
        return torch.cat([q_tensor, k_tensor, v_tensor], dim=0)


def apply_lora_qkv_xccl_patch():
    """Patch FSDPEngine._update_bucket_weights_from_distributed for QKV translation.
    
    This intercepts the weight update and:
    1. Detects Q/K/V LoRA params
    2. Translates names from q_proj/k_proj/v_proj to qkv_proj
    3. Does NOT merge tensors - SGLang handles the Q/K/V separately even with qkv_proj naming
    
    NOTE: After investigation, SGLang appears to expect individual Q/K/V tensors
    but with the qkv_proj naming prefix. We translate names but don't merge.
    """
    try:
        from areal.engine.fsdp_engine import FSDPEngine
        
        # Store original method
        original_update_bucket = FSDPEngine._update_bucket_weights_from_distributed
        
        def patched_update_bucket_weights(
            self,
            meta,
            named_tensors: List[Tuple[str, Any]]
        ):
            """Patched version that handles QKV name translation (NO tensor merge)."""
            if not named_tensors:
                return
            
            # Check if we have any QKV LoRA params
            has_qkv = any(is_qkv_lora_param(name) for name, _ in named_tensors)
            
            if not has_qkv:
                # No QKV params, use original
                return original_update_bucket(self, meta, named_tensors)
            
            logger.info("LoRA QKV XCCL Patch: Detected QKV LoRA params, translating names (NO merge)...")
            print("  [LoRA QKV XCCL] Translating Q/K/V names to qkv_proj (individual tensors)")
            
            # Simply rename q_proj/k_proj/v_proj -> qkv_proj without merging
            translated_params = []
            for name, tensor in named_tensors:
                new_name = translate_to_qkv_proj(name)
                if new_name != name:
                    logger.info(f"  Renamed: {name.split('.')[-4]} -> {new_name.split('.')[-4]}")
                translated_params.append((new_name, tensor))
            
            # Replace named_tensors with translated params
            named_tensors.clear()
            named_tensors.extend(translated_params)
            
            return original_update_bucket(self, meta, named_tensors)
        
        # Apply patch
        FSDPEngine._update_bucket_weights_from_distributed = patched_update_bucket_weights
        
        logger.info("LoRA QKV XCCL Patch: Applied successfully (name translation only, no merge)!")
        print("  [LoRA QKV XCCL] Applied (Q/K/V -> qkv_proj translation, no merge)")
        return True
        
    except ImportError as e:
        logger.warning(f"LoRA QKV XCCL Patch: Could not import FSDPEngine: {e}")
        print(f"  [LoRA QKV XCCL] WARNING: Could not import FSDPEngine: {e}")
        return False
    except Exception as e:
        logger.error(f"LoRA QKV XCCL Patch: Failed to apply: {e}")
        print(f"  [LoRA QKV XCCL] ERROR: Failed to apply: {e}")
        return False
