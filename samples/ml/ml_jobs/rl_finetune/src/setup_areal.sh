#!/bin/bash
set -e

echo "=========================================="
echo "Environment Check"
echo "=========================================="
echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"
echo "Ray version: $(python3 -c 'import ray; print(ray.__version__)')"

echo "=========================================="
echo "Cloning AReaL"
echo "=========================================="
cd /tmp
rm -rf AReaL
git clone https://github.com/inclusionAI/AReaL.git
cd AReaL

echo "=========================================="
echo "Installing AReaL into SPCS environment"
echo "=========================================="
pip install -e . --no-deps
pip install sglang[srt] transformers datasets accelerate peft bitsandbytes --no-build-isolation 2>/dev/null || \
    pip install sglang transformers datasets accelerate peft bitsandbytes

echo "=========================================="
echo "Verifying installation"
echo "=========================================="
python3 -c "import areal; print('AReaL version:', getattr(areal, '__version__', 'unknown'))"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import ray; print(f'Ray: {ray.__version__}')"
python3 -c "import sglang; print('SGLang available')" || echo "SGLang not available"

echo "=========================================="
echo "AReaL installation complete!"
echo "=========================================="
