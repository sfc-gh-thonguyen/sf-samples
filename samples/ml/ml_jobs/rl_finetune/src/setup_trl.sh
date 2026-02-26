#!/bin/bash
set -e

echo "=========================================="
echo "Installing TRL and dependencies"
echo "=========================================="

pip install trl peft accelerate bitsandbytes datasets

python -c "import trl; print('TRL version:', trl.__version__)"

echo "=========================================="
echo "TRL installation complete!"
echo "=========================================="
