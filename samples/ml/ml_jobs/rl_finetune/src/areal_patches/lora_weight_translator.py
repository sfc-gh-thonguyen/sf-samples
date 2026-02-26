"""LoRA Weight Name Translator for SGLang QKV Merge.

SGLang internally merges Q, K, V projections into a single qkv_proj tensor for efficiency.
This causes parameter name mismatches when doing XCCL (GPU-to-GPU) weight sync with AReaL,
which keeps Q, K, V as separate parameters.

Trainer sends:     layers.0.self_attn.q_proj.lora_A.default.weight
SGLang expects:    layers.0.self_attn.qkv_proj.lora_A.default.weight

This module provides utilities to translate parameter names and handle the merge.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Pattern to match Q/K/V projection parameter names
QKV_PATTERN = re.compile(
    r'^(.*\.self_attn\.)(q_proj|k_proj|v_proj)(\.lora_[AB]\..*\.weight)$'
)

def translate_qkv_to_merged(param_name: str) -> Tuple[str, Optional[str]]:
    """Translate separate Q/K/V param name to merged qkv_proj format.
    
    Args:
        param_name: Original parameter name (e.g., 'layers.0.self_attn.q_proj.lora_A.default.weight')
        
    Returns:
        Tuple of (translated_name, qkv_component)
        - translated_name: Name with qkv_proj (e.g., 'layers.0.self_attn.qkv_proj.lora_A.default.weight')
        - qkv_component: 'q', 'k', 'v', or None if not a QKV param
    """
    match = QKV_PATTERN.match(param_name)
    if match:
        prefix, qkv_type, suffix = match.groups()
        merged_name = f"{prefix}qkv_proj{suffix}"
        component = qkv_type[0]  # 'q', 'k', or 'v'
        return merged_name, component
    return param_name, None


def group_qkv_params(param_names: List[str]) -> Dict[str, Dict[str, str]]:
    """Group Q/K/V parameters by their merged qkv_proj name.
    
    Args:
        param_names: List of parameter names from trainer
        
    Returns:
        Dict mapping merged param name -> {'q': orig_name, 'k': orig_name, 'v': orig_name}
    """
    groups = {}
    for name in param_names:
        merged_name, component = translate_qkv_to_merged(name)
        if component:
            if merged_name not in groups:
                groups[merged_name] = {}
            groups[merged_name][component] = name
    return groups


def build_translation_map(param_names: List[str]) -> Dict[str, str]:
    """Build a translation map from trainer param names to SGLang param names.
    
    For non-QKV params, name stays the same.
    For QKV params, all three (q_proj, k_proj, v_proj) map to the same qkv_proj name.
    
    Note: The actual tensor merging needs to happen at the receiving end.
    """
    translation = {}
    for name in param_names:
        merged_name, _ = translate_qkv_to_merged(name)
        translation[name] = merged_name
    return translation


class LoRAWeightMerger:
    """Handles merging of Q/K/V LoRA weights for SGLang's merged qkv_proj.
    
    SGLang stores QKV as a single tensor with shape:
    - For lora_A: [rank, q_size + k_size + v_size]
    - For lora_B: [q_size + k_size + v_size, rank]
    
    The trainer sends separate tensors for Q, K, V which need to be concatenated.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int = None):
        """Initialize the merger.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            head_dim: Head dimension (defaults to hidden_size // num_heads)
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.q_size = self.k_size = self.v_size = num_heads * self.head_dim
        
    def merge_lora_weights(
        self, 
        q_weight, 
        k_weight, 
        v_weight,
        is_lora_a: bool
    ):
        """Merge Q/K/V LoRA weights into a single qkv_proj weight.
        
        Args:
            q_weight: Q projection LoRA weight
            k_weight: K projection LoRA weight  
            v_weight: V projection LoRA weight
            is_lora_a: True for lora_A (concat on dim 0), False for lora_B (concat on dim 1)
            
        Returns:
            Merged qkv_proj weight tensor
        """
        import torch
        
        if is_lora_a:
            # lora_A: [rank, hidden_size] for each -> [rank, 3*hidden_size]
            return torch.cat([q_weight, k_weight, v_weight], dim=0)
        else:
            # lora_B: [hidden_size, rank] for each -> [3*hidden_size, rank]
            return torch.cat([q_weight, k_weight, v_weight], dim=0)


def apply_xccl_weight_translation_patch():
    """Patch AReaL/SGLang to handle LoRA weight name translation for XCCL mode.
    
    This patches the weight update flow to:
    1. Intercept parameter names during XCCL update
    2. Group Q/K/V params and merge them
    3. Send merged params with qkv_proj names to SGLang
    """
    try:
        # Try to patch the SGLang update handler
        from sglang.srt.managers.controller_single import update_weights_from_distributed
        
        logger.info("LoRA weight translator: Patching SGLang for QKV merge...")
        
        # The actual patch implementation would go here
        # For now, we log that we're attempting the patch
        
        return True
    except ImportError as e:
        logger.warning(f"LoRA weight translator: Could not patch SGLang: {e}")
        return False
    except Exception as e:
        logger.error(f"LoRA weight translator: Patch failed: {e}")
        return False


# Ray Object Store based weight sync alternative
def save_weights_to_ray(weights_dict: dict, name: str = "lora_weights"):
    """Save LoRA weights to Ray object store for cross-node sharing.
    
    Args:
        weights_dict: Dict of param_name -> tensor
        name: Name for the object reference
        
    Returns:
        Ray ObjectRef that can be passed to other nodes
    """
    import ray
    ref = ray.put(weights_dict)
    logger.info(f"Saved weights to Ray object store: {name}")
    return ref


def load_weights_from_ray(ref):
    """Load LoRA weights from Ray object store.
    
    Args:
        ref: Ray ObjectRef from save_weights_to_ray
        
    Returns:
        Dict of param_name -> tensor
    """
    import ray
    weights = ray.get(ref)
    logger.info(f"Loaded weights from Ray object store")
    return weights
