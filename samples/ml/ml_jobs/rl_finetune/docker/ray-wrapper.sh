#!/bin/bash
# Ray compatibility wrapper for ML Jobs bootstrap
# Ray 2.53.0+ renamed --dashboard-grpc-port to --dashboard-agent-grpc-port

cmd=()
i=0
args=("$@")
while [ $i -lt ${#args[@]} ]; do
    arg="${args[$i]}"
    next_arg="${args[$((i+1))]:-}"
    
    if [[ "$arg" == "--dashboard-grpc-port" ]]; then
        cmd+=("--dashboard-agent-grpc-port")
    elif [[ "$arg" == --dashboard-grpc-port=* ]]; then
        value="${arg#--dashboard-grpc-port=}"
        cmd+=("--dashboard-agent-grpc-port=$value")
    elif [[ "$arg" == "--dashboard-port=" ]] || [[ "$arg" == "--dashboard-port=''" ]] || [[ "$arg" == '--dashboard-port=""' ]]; then
        cmd+=("--dashboard-port=8265")
    elif [[ "$arg" == "--dashboard-port" ]] && [[ -z "$next_arg" || "$next_arg" == --* ]]; then
        cmd+=("--dashboard-port=8265")
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
