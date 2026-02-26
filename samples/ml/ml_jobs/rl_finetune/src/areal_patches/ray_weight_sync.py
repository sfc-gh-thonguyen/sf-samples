"""
Ray Object Store based LoRA Weight Sync for SPCS.

PROBLEM: SPCS has no shared filesystem across nodes, so disk-based weight sync fails.
SOLUTION: Use Ray's object store as a cross-node memory transport.

Flow:
1. Trainer saves LoRA checkpoint to Ray object store (ray.put)
2. Remote function on SGLang node retrieves weights (ray.get)
3. Saves to local /tmp/ on SGLang node
4. Calls /load_lora_adapter with local path
5. SGLang's disk loading handles Q/K/V -> qkv_proj name translation

This patches AReaL's FSDPEngine._update_weights_from_disk method.
"""
import os
import json
import tempfile
import requests
from typing import Dict, Any, Optional

import ray
import torch


LORA_ADAPTER_LOCAL_PATH = "/tmp/ray_lora_adapter"


def save_lora_weights_to_ray(weights: Dict[str, torch.Tensor], adapter_config: dict) -> ray.ObjectRef:
    """Save LoRA weights and config to Ray object store.
    
    Args:
        weights: Dict of parameter name -> tensor
        adapter_config: PEFT adapter config dict
        
    Returns:
        Ray ObjectRef that can be passed to other nodes
    """
    # CRITICAL: Convert tensors to numpy to avoid Ray's zero-copy tensor issues
    # Ray's zero-copy mechanism can leave tensors with invalid storage after transfer
    # Using numpy ensures proper serialization/deserialization
    import numpy as np
    
    # Convert to numpy arrays (these serialize cleanly through Ray)
    numpy_weights = {}
    for k, v in weights.items():
        # Handle DTensors (distributed tensors from FSDP)
        if hasattr(v, 'to_local'):
            tensor = v.to_local().cpu().detach()
        elif hasattr(v, 'full_tensor'):
            tensor = v.full_tensor().cpu().detach()
        else:
            tensor = v.cpu().detach()
        
        # Ensure we have a regular tensor, not a subclass
        if tensor.__class__.__name__ != 'Tensor':
            tensor = tensor.data
        
        # Store original dtype for reconstruction
        original_dtype = str(tensor.dtype)
        
        # Convert BFloat16 to Float32 for numpy (numpy doesn't support bf16)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        
        numpy_weights[k] = {
            "data": tensor.numpy().tobytes(),  # Raw bytes
            "shape": list(tensor.shape),
            "dtype": original_dtype,  # Store original dtype for reconstruction
        }
    
    data = {
        "numpy_weights": numpy_weights,
        "adapter_config": adapter_config,
    }
    return ray.put(data)


@ray.remote(resources={"node_tag:head": 0.001})  # Schedule on SGLang/head node
def load_lora_on_sglang_node(
    weights_data: dict,  # Ray auto-dereferences ObjectRefs, so this is the actual data
    sglang_addr: str,
    lora_name: str = "lora-adapter",
) -> dict:
    """Remote function to load LoRA weights on SGLang node.
    
    This runs on the SGLang node and:
    1. Receives weights data (Ray auto-dereferences ObjectRefs)
    2. Saves to local filesystem
    3. Calls SGLang /load_lora_adapter endpoint
    
    Args:
        weights_data: Dict containing weights and adapter_config (auto-dereferenced by Ray)
        sglang_addr: SGLang server address (e.g., "10.0.0.1:34000")
        lora_name: Name for the LoRA adapter
        
    Returns:
        Dict with success status and any error messages
    """
    try:
        import numpy as np
        import safetensors.torch as st
        
        numpy_weights = weights_data["numpy_weights"]
        adapter_config = weights_data["adapter_config"]
        
        # Reconstruct tensors from numpy bytes
        dtype_map = {
            "torch.float32": (torch.float32, np.float32),
            "torch.float16": (torch.float16, np.float16),
            "torch.bfloat16": (torch.bfloat16, np.float32),  # numpy doesn't support bfloat16, use float32
            "torch.int64": (torch.int64, np.int64),
            "torch.int32": (torch.int32, np.int32),
        }
        
        weights = {}
        for k, v in numpy_weights.items():
            shape = tuple(v["shape"])
            dtype_str = v["dtype"]
            torch_dtype, np_dtype = dtype_map.get(dtype_str, (torch.float32, np.float32))
            
            # Convert bytes to numpy array then to tensor
            np_array = np.frombuffer(v["data"], dtype=np_dtype).reshape(shape)
            weights[k] = torch.from_numpy(np_array.copy()).to(torch_dtype)  # .copy() ensures we own the memory
        
        local_path = os.path.join(LORA_ADAPTER_LOCAL_PATH, lora_name)
        os.makedirs(local_path, exist_ok=True)
        
        weights_path = os.path.join(local_path, "adapter_model.safetensors")
        config_path = os.path.join(local_path, "adapter_config.json")
        
        print(f"[RayWeightSync] Saving {len(weights)} weights to {weights_path}")
        st.save_file(weights, weights_path)
        
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        print(f"[RayWeightSync] Saved LoRA adapter to {local_path}")
        print(f"[RayWeightSync] Weights: {list(weights.keys())[:5]}... ({len(weights)} total)")
        
        url = f"http://{sglang_addr}/load_lora_adapter"
        payload = {
            "lora_name": lora_name,
            "lora_path": local_path,
        }
        
        print(f"[RayWeightSync] Calling {url}")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            print(f"[RayWeightSync] Successfully loaded LoRA adapter")
            return {"success": True, "message": "LoRA loaded successfully"}
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"[RayWeightSync] Failed to load LoRA: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[RayWeightSync] Exception: {error_msg}")
        return {"success": False, "error": error_msg}


def create_adapter_config(
    base_model_name: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    target_modules: list = None,
) -> dict:
    """Create a PEFT-compatible adapter config."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    return {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }


def extract_lora_weights(model) -> Dict[str, torch.Tensor]:
    """Extract LoRA weights from a PEFT model.
    
    Args:
        model: A model with LoRA adapters (PEFT model)
        
    Returns:
        Dict of LoRA parameter names to tensors
    """
    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            clean_name = name
            if clean_name.startswith("base_model.model."):
                clean_name = clean_name[len("base_model.model."):]
            lora_weights[clean_name] = param.detach().clone()
    return lora_weights


class RayWeightSyncManager:
    """Manager for Ray-based LoRA weight synchronization."""
    
    def __init__(
        self,
        sglang_addrs: list,
        base_model_name: str,
        lora_name: str = "lora-adapter",
        lora_rank: int = 16,
        lora_alpha: int = 16,
    ):
        """Initialize the weight sync manager.
        
        Args:
            sglang_addrs: List of SGLang server addresses
            base_model_name: Base model name/path for adapter config
            lora_name: Name of the LoRA adapter
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
        """
        self.sglang_addrs = sglang_addrs
        self.base_model_name = base_model_name
        self.lora_name = lora_name
        self.adapter_config = create_adapter_config(
            base_model_name=base_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self._update_count = 0
        
    def sync_weights(self, model) -> dict:
        """Sync LoRA weights from trainer to all SGLang servers.
        
        Args:
            model: The PEFT model with LoRA adapters
            
        Returns:
            Dict with sync results
        """
        self._update_count += 1
        
        lora_weights = extract_lora_weights(model)
        if not lora_weights:
            return {"success": False, "error": "No LoRA weights found in model"}
        
        print(f"[RayWeightSync] Update #{self._update_count}: Syncing {len(lora_weights)} LoRA params")
        
        weights_ref = save_lora_weights_to_ray(lora_weights, self.adapter_config)
        
        futures = []
        for addr in self.sglang_addrs:
            future = load_lora_on_sglang_node.remote(
                weights_data=weights_ref,  # Ray auto-dereferences ObjectRefs
                sglang_addr=addr,
                lora_name=self.lora_name,
            )
            futures.append((addr, future))
        
        results = {}
        for addr, future in futures:
            try:
                result = ray.get(future, timeout=120)
                results[addr] = result
            except Exception as e:
                results[addr] = {"success": False, "error": str(e)}
        
        all_success = all(r.get("success", False) for r in results.values())
        return {
            "success": all_success,
            "update_count": self._update_count,
            "results": results,
        }


_weight_sync_manager: Optional[RayWeightSyncManager] = None


def get_weight_sync_manager() -> Optional[RayWeightSyncManager]:
    """Get the global weight sync manager."""
    return _weight_sync_manager


def init_weight_sync_manager(
    sglang_addrs: list,
    base_model_name: str,
    lora_name: str = "lora-adapter",
    lora_rank: int = 16,
    lora_alpha: int = 16,
) -> RayWeightSyncManager:
    """Initialize the global weight sync manager."""
    global _weight_sync_manager
    _weight_sync_manager = RayWeightSyncManager(
        sglang_addrs=sglang_addrs,
        base_model_name=base_model_name,
        lora_name=lora_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    return _weight_sync_manager


def auto_init_weight_sync_from_env() -> Optional[RayWeightSyncManager]:
    """Auto-initialize weight sync manager from environment variables.
    
    Uses:
    - AREAL_LLM_SERVER_ADDRS: Comma-separated list of SGLang addresses
    - Model config is extracted from the FSDPEngine when first weight update is called
    
    Returns:
        RayWeightSyncManager if initialized, None otherwise
    """
    global _weight_sync_manager
    
    sglang_addrs_str = os.environ.get("AREAL_LLM_SERVER_ADDRS", "")
    if not sglang_addrs_str:
        print("[RayWeightSync] AREAL_LLM_SERVER_ADDRS not set - cannot auto-initialize")
        return None
    
    sglang_addrs = sglang_addrs_str.split(",")
    print(f"[RayWeightSync] Auto-initializing with {len(sglang_addrs)} servers: {sglang_addrs}")
    
    _weight_sync_manager = RayWeightSyncManager(
        sglang_addrs=sglang_addrs,
        base_model_name="auto",  # Will be updated when we get the model
        lora_name="lora-gsm8k",  # Default from config
        lora_rank=16,
        lora_alpha=16,
    )
    return _weight_sync_manager


def apply_ray_weight_sync_patch():
    """Patch AReaL's FSDPEngine to use Ray-based weight sync.
    
    This replaces _update_weights_from_disk with our Ray object store implementation.
    Automatically initializes from AREAL_LLM_SERVER_ADDRS environment variable.
    """
    try:
        from areal.engine.fsdp_engine import FSDPEngine
    except ImportError:
        print("[RayWeightSync] Could not import FSDPEngine - patch not applied")
        return False
    
    original_update_from_disk = getattr(FSDPEngine, '_update_weights_from_disk', None)
    
    def patched_update_weights_from_disk(self, meta):
        """Patched method that uses Ray object store instead of disk.
        
        This intercepts the disk-based weight update and redirects it
        through Ray object store for cross-node communication.
        """
        global _weight_sync_manager
        
        manager = get_weight_sync_manager()
        
        if manager is None:
            print("[RayWeightSync] Auto-initializing weight sync manager...")
            manager = auto_init_weight_sync_from_env()
            
            if manager is None:
                print("[RayWeightSync] WARNING: Could not initialize, falling back to original")
                if original_update_from_disk:
                    return original_update_from_disk(self, meta)
                return
        
        model = getattr(self, 'model', None) or getattr(self, '_model', None)
        if model is None:
            print("[RayWeightSync] ERROR: Could not find model in FSDPEngine")
            if original_update_from_disk:
                return original_update_from_disk(self, meta)
            return
        
        if manager.base_model_name == "auto":
            try:
                model_config = getattr(model, 'config', None)
                if model_config:
                    base_name = getattr(model_config, 'name_or_path', None) or getattr(model_config, '_name_or_path', None)
                    if base_name:
                        manager.base_model_name = base_name
                        manager.adapter_config["base_model_name_or_path"] = base_name
                        print(f"[RayWeightSync] Detected base model: {base_name}")
            except Exception as e:
                print(f"[RayWeightSync] Could not detect base model name: {e}")
        
        print(f"[RayWeightSync] Intercepted _update_weights_from_disk")
        result = manager.sync_weights(model)
        
        if result["success"]:
            print(f"[RayWeightSync] Successfully synced weights (update #{result['update_count']})")
        else:
            # Log detailed error info
            print(f"[RayWeightSync] ===== WEIGHT SYNC FAILED =====")
            print(f"[RayWeightSync] Result: {result}")
            if "results" in result:
                for addr, res in result["results"].items():
                    print(f"[RayWeightSync]   {addr}: {res}")
            
            # DO NOT fall back to disk method - it won't work in SPCS
            # Instead raise a clear error so we know what happened
            raise RuntimeError(
                f"[RayWeightSync] Cross-node weight sync failed! "
                f"Result: {result}. "
                f"This is expected if AREAL_LLM_SERVER_ADDRS is not set correctly."
            )
    
    FSDPEngine._update_weights_from_disk = patched_update_weights_from_disk
    print("[RayWeightSync] Patched FSDPEngine._update_weights_from_disk")
    
    return True
