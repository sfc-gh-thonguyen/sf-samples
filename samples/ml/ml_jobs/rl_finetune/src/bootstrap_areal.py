#!/usr/bin/env python3
"""Bootstrap script to install AReaL using uv (much faster than pip)."""
import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Install AReaL with --no-deps to avoid dependency hell
# The base image should have most deps (torch, transformers, etc.)
run("uv pip install 'sglang[srt]>=0.4.9.post2' --system")
run("uv pip install 'areal @ git+https://github.com/inclusionAI/AReaL.git' --no-deps --system")

# Install minimal missing deps that AReaL actually needs for GRPO
run("uv pip install hydra-core omegaconf wandb --system")

print("Bootstrap complete!")
