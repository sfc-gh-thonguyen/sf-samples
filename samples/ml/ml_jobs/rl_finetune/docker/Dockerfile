FROM ghcr.io/inclusionai/areal-runtime:v0.5.3

# Install cron and sysvinit-utils (for service command) required by ML Jobs bootstrap
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    sysvinit-utils \
    && rm -rf /var/lib/apt/lists/*

# Keep Ray at the version from the base image (2.53.0)
# Ray 2.53.0 supports --dashboard-port but removed --dashboard-grpc-port
# The ML Jobs bootstrap should handle this correctly

# Create a wrapper script to intercept ray commands and fix deprecated flags
# This is needed because ML Jobs bootstrap uses --dashboard-grpc-port which was
# renamed to --dashboard-agent-grpc-port in Ray 2.53.0
RUN mv /AReaL/.venv/bin/ray /AReaL/.venv/bin/ray-original
COPY <<'WRAPPER_EOF' /AReaL/.venv/bin/ray
#!/bin/bash
# Ray 2.53.0 compatibility wrapper for ML Jobs bootstrap
# 
# Handles:
# 1. --dashboard-grpc-port -> --dashboard-agent-grpc-port (deprecated in Ray 2.53.0)
# 2. Sets default values for empty --dashboard-port and --dashboard-host
#
# Reference: https://docs.ray.io/en/latest/ray-core/configure.html
# Defaults: --dashboard-port=8265, --dashboard-host=127.0.0.1

cmd=()
i=0
args=("$@")
while [ $i -lt ${#args[@]} ]; do
    arg="${args[$i]}"
    next_arg="${args[$((i+1))]:-}"
    
    # Convert deprecated --dashboard-grpc-port to --dashboard-agent-grpc-port
    if [[ "$arg" == "--dashboard-grpc-port" ]]; then
        cmd+=("--dashboard-agent-grpc-port")
    elif [[ "$arg" == --dashboard-grpc-port=* ]]; then
        value="${arg#--dashboard-grpc-port=}"
        cmd+=("--dashboard-agent-grpc-port=$value")
    
    # Handle --dashboard-port with empty value -> use default 8265
    elif [[ "$arg" == "--dashboard-port=" ]] || [[ "$arg" == "--dashboard-port=''" ]] || [[ "$arg" == '--dashboard-port=""' ]]; then
        cmd+=("--dashboard-port=8265")
    elif [[ "$arg" == "--dashboard-port" ]] && [[ -z "$next_arg" || "$next_arg" == --* ]]; then
        cmd+=("--dashboard-port=8265")
    
    # Handle --dashboard-host with empty value -> use default 127.0.0.1
    elif [[ "$arg" == "--dashboard-host=" ]] || [[ "$arg" == "--dashboard-host=''" ]] || [[ "$arg" == '--dashboard-host=""' ]]; then
        cmd+=("--dashboard-host=127.0.0.1")
    elif [[ "$arg" == "--dashboard-host" ]] && [[ -z "$next_arg" || "$next_arg" == --* ]]; then
        cmd+=("--dashboard-host=127.0.0.1")
    
    else
        cmd+=("$arg")
    fi
    
    ((i++))
done

exec /AReaL/.venv/bin/ray-original "${cmd[@]}"
WRAPPER_EOF
RUN chmod +x /AReaL/.venv/bin/ray && \
    ln -sf /AReaL/.venv/bin/ray /usr/local/bin/ray

# Install AReaL
RUN git clone https://github.com/inclusionAI/AReaL /opt/AReaL && \
    cd /opt/AReaL && \
    pip install -e . --no-deps

# Install Snowflake runtime dependencies for ML Jobs
# Install to BOTH venv python (default python3) and system python
# The venv python is what ML Jobs uses when it calls "python3"
RUN uv pip install --python /AReaL/.venv/bin/python --no-deps snowflake-ml-python==1.26.0 && \
    uv pip install --python /AReaL/.venv/bin/python --no-deps snowflake-connector-python==3.17.0 && \
    uv pip install --python /AReaL/.venv/bin/python --no-deps snowflake-snowpark-python && \
    uv pip install --python /AReaL/.venv/bin/python --no-deps snowflake.core && \
    uv pip install --python /AReaL/.venv/bin/python asn1crypto boto3 sortedcontainers tomlkit cachetools retrying pytimeparse sqlparse importlib_resources cryptography pyopenssl

# Also install to system python as backup
RUN /usr/bin/python3 -m pip install --no-deps snowflake-ml-python==1.26.0 && \
    /usr/bin/python3 -m pip install --no-deps snowflake-connector-python==3.17.0 && \
    /usr/bin/python3 -m pip install --no-deps snowflake-snowpark-python && \
    /usr/bin/python3 -m pip install --no-deps snowflake.core && \
    /usr/bin/python3 -m pip install --ignore-installed asn1crypto boto3 sortedcontainers tomlkit cachetools retrying pytimeparse sqlparse importlib_resources cryptography pyopenssl

# Copy and install mlruntimes_service wheel (provides snowflake.runtime)
# Use --no-deps to avoid dependency resolution slowdown - deps already installed above
COPY mlruntimes_service-2.2.26-py3-none-any.whl /tmp/
RUN uv pip install --python /AReaL/.venv/bin/python --no-deps /tmp/mlruntimes_service-2.2.26-py3-none-any.whl && \
    /usr/bin/python3 -m pip install --no-deps /tmp/mlruntimes_service-2.2.26-py3-none-any.whl

ENV PYTHONPATH="/opt/AReaL:$PYTHONPATH"

# Disable HuggingFace XET storage to use traditional LFS download
# XET requires additional network endpoints (transfer.xethub.hf.co, cas-bridge.xethub.hf.co)
# that may not be whitelisted in enterprise environments
ENV HF_HUB_DISABLE_XET=1

# Add ML Jobs entrypoint script (required by Snowflake ML Jobs)
RUN echo '#!/bin/bash\nexec "$@"' > /usr/local/bin/_entrypoint.sh && \
    chmod +x /usr/local/bin/_entrypoint.sh
