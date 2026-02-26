#!/usr/bin/env python3
"""
Log wrapper for SPCS jobs.

This wrapper captures all stdout/stderr to a log file and tees to console,
allowing us to retrieve full logs from the stage after the job completes.

Usage: python log_wrapper.py <actual_script.py> [args...]

The log file is written to /mnt/job_stage/logs/<timestamp>_<script>_<hostname>.log
"""
import os
import sys
import subprocess
import socket
import datetime
import threading
import queue


def main():
    if len(sys.argv) < 2:
        print("Usage: python log_wrapper.py <script.py> [args...]")
        sys.exit(1)
    
    script = sys.argv[1]
    args = sys.argv[2:]
    
    hostname = socket.gethostname()[:20]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(script).replace(".py", "")
    
    log_dir = "/mnt/job_stage/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{timestamp}_{script_name}_{hostname}.log")
    
    print(f"[LOG_WRAPPER] Starting {script} with args: {args}")
    print(f"[LOG_WRAPPER] Log file: {log_file}")
    print(f"[LOG_WRAPPER] Hostname: {hostname}")
    print("=" * 80)
    sys.stdout.flush()
    
    cmd = [sys.executable, script] + args
    
    with open(log_file, 'w', buffering=1) as f:
        f.write(f"[LOG_WRAPPER] Script: {script}\n")
        f.write(f"[LOG_WRAPPER] Args: {args}\n")
        f.write(f"[LOG_WRAPPER] Hostname: {hostname}\n")
        f.write(f"[LOG_WRAPPER] Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n")
        f.flush()
        
        def stream_output(stream, name):
            lines_since_print = 0
            last_print = None
            while True:
                line = stream.readline()
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='replace')
                f.write(line_str)
                f.flush()
                
                if "Waiting for instances" in line_str:
                    lines_since_print += 1
                    if lines_since_print == 1 or lines_since_print % 30 == 0:
                        print(line_str, end='')
                        sys.stdout.flush()
                else:
                    print(line_str, end='')
                    sys.stdout.flush()
                    lines_since_print = 0
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0
        )
        
        stream_output(proc.stdout, "stdout")
        
        ret = proc.wait()
        
        f.write(f"\n[LOG_WRAPPER] Process exited with code: {ret}\n")
        f.flush()
    
    print("=" * 80)
    print(f"[LOG_WRAPPER] Process exited with code: {ret}")
    print(f"[LOG_WRAPPER] Full logs at: {log_file}")
    
    if ret != 0:
        print("\n[LOG_WRAPPER] === LAST 200 LINES OF LOG ===")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-200:]:
                    print(line, end='')
        except Exception as e:
            print(f"[LOG_WRAPPER] Error reading log: {e}")
    
    sys.exit(ret)


if __name__ == "__main__":
    main()
