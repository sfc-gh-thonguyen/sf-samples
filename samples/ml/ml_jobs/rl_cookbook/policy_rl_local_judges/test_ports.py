#!/usr/bin/env python3
"""Quick SPCS port test — checks which ports can bind and accept connections.

Submits a lightweight ML Job that:
1. Tries to bind a TCP server on candidate ports
2. Connects back to itself on each bound port
3. Prints results to logs

Usage:
    SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python test_ports.py
"""
import argparse
import os

from snowflake.snowpark import Session
from snowflake.ml import jobs


TEST_SCRIPT = '''#!/usr/bin/env python3
import socket
import sys
import time
import threading

sys.stdout.reconfigure(line_buffering=True)

# Candidate ports to test
PORTS = [
    8080, 8443, 8899,          # common app ports
    12031, 12032,               # SPCS range start
    30000, 32768, 35000,        # mid range
    38899,                      # our judge port
    40000, 45000,               # upper range
    49999, 50000,               # SPCS range end
    50001, 50080,               # above SPCS range
]

print("=" * 60)
print("SPCS Port Connectivity Test")
print("=" * 60)

results = {}

for port in PORTS:
    # Step 1: Try to bind
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        srv.settimeout(5)
        bind_ok = True
    except OSError as e:
        results[port] = f"BIND FAILED: {e}"
        print(f"  Port {port:>5}: BIND FAILED — {e}")
        continue

    # Step 2: Try to connect to ourselves
    connect_ok = False
    try:
        def accept_one(s):
            try:
                conn, _ = s.accept()
                conn.sendall(b"ok")
                conn.close()
            except Exception:
                pass

        t = threading.Thread(target=accept_one, args=(srv,), daemon=True)
        t.start()

        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.settimeout(3)
        cli.connect(("127.0.0.1", port))
        data = cli.recv(16)
        cli.close()
        connect_ok = (data == b"ok")
        t.join(timeout=3)
    except Exception as e:
        results[port] = f"BIND OK, CONNECT FAILED: {e}"
        print(f"  Port {port:>5}: BIND OK, CONNECT FAILED — {e}")
        srv.close()
        continue

    srv.close()

    if connect_ok:
        results[port] = "OK"
        print(f"  Port {port:>5}: OK (bind + loopback connect)")
    else:
        results[port] = "BIND OK, LOOPBACK FAILED"
        print(f"  Port {port:>5}: BIND OK, LOOPBACK FAILED")

# Also test HTTP on 38899 (simulate vLLM health endpoint)
print("\\n--- HTTP Server Test on port 38899 ---")
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.request

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"healthy")
        def log_message(self, *args):
            pass

    httpd = HTTPServer(("0.0.0.0", 38899), Handler)
    t = threading.Thread(target=httpd.handle_request, daemon=True)
    t.start()
    time.sleep(0.5)

    resp = urllib.request.urlopen("http://127.0.0.1:38899/health", timeout=5)
    body = resp.read()
    print(f"  HTTP GET http://127.0.0.1:38899/health -> {resp.status} {body}")
    httpd.server_close()
except Exception as e:
    print(f"  HTTP test FAILED: {e}")

print("\\n" + "=" * 60)
print("Summary:")
for port in PORTS:
    status = results.get(port, "NOT TESTED")
    marker = "✓" if status == "OK" else "✗"
    print(f"  {marker} {port:>5}: {status}")
print("=" * 60)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--compute-pool", default="RL_LOCAL_JUDGE_POOL")
    parser.add_argument("--database", default="RL_TRAINING_DB")
    parser.add_argument("--schema", default="RL_SCHEMA")
    args = parser.parse_args()

    connection_name = os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME")
    builder = Session.builder
    if connection_name:
        builder = builder.config("connection_name", connection_name)
    builder = builder.config("database", args.database)
    builder = builder.config("schema", args.schema)
    builder = builder.config("role", "SYSADMIN")
    session = builder.create()
    session.sql(f"USE DATABASE {args.database}").collect()
    session.sql(f"USE SCHEMA {args.schema}").collect()

    # Write test script to temp dir
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="port_test_")
    script_path = os.path.join(tmpdir, "test_ports_spcs.py")
    with open(script_path, "w") as f:
        f.write(TEST_SCRIPT)

    print(f"Submitting port test job on {args.compute_pool}...")
    job = jobs.submit_directory(
        tmpdir,
        entrypoint="test_ports_spcs.py",
        compute_pool=args.compute_pool,
        stage_name=f"{args.database}.{args.schema}.rl_payload_stage",
        database=args.database,
        schema=args.schema,
        session=session,
    )
    print(f"Job: {job.id}")
    print("Waiting for completion...")
    status = job.wait()
    print(f"Status: {status}")
    print(f"\nLogs:\n{job.get_logs()}")


if __name__ == "__main__":
    main()
