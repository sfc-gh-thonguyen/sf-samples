#!/usr/bin/env python3
"""
Wrapper that applies LoRA QKV XCCL patch BEFORE importing AReaL modules.

This script must be run INSTEAD of `python -m areal.launcher.ray` to ensure
the patch is applied before any AReaL code is imported in the trainer process.
"""
import os
import sys
import importlib.util

def apply_patches():
    """Apply all necessary patches before importing AReaL."""
    # Get the patches directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    patches_dir = os.path.join(script_dir, 'areal_patches')
    
    print("=" * 60)
    print("TRAINER WRAPPER: Applying patches before AReaL import")
    print("=" * 60)
    
    # Apply LoRA QKV XCCL patch
    try:
        # Add patches dir to path
        sys.path.insert(0, script_dir)
        sys.path.insert(0, patches_dir)
        
        from lora_qkv_xccl_patch import apply_lora_qkv_xccl_patch
        if apply_lora_qkv_xccl_patch():
            print("  [TRAINER] LoRA QKV XCCL patch applied!")
        else:
            print("  [TRAINER] WARNING: LoRA QKV XCCL patch failed!")
    except Exception as e:
        print(f"  [TRAINER] ERROR applying patch: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Apply patches first
    apply_patches()
    
    # Now import and run AReaL launcher
    print("\n" + "=" * 60)
    print("TRAINER WRAPPER: Starting AReaL launcher")
    print("=" * 60)
    
    # Import and run the main entry point
    from areal.launcher.ray import main as ray_main
    ray_main()

if __name__ == "__main__":
    main()
