import os, sys, subprocess, webbrowser, time, urllib.request, json
from pathlib import Path

def main():
    os.chdir(Path(__file__).parent)

    venv_python = Path("venv/Scripts/python.exe")
    python_exe = str(venv_python.resolve()) if venv_python.exists() else sys.executable

    backend_reqs = Path("backend/requirements.txt")
    installed_flag = Path(".deps_installed")
    if backend_reqs.exists() and not installed_flag.exists():
        print("  Installing dependencies (one-time)...", flush=True)
        subprocess.run([python_exe, "-m", "pip", "install", "-q", "-r", str(backend_reqs)], shell=True)
        installed_flag.write_text("done")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"  Starting server on {host}:{port} ...", flush=True)
    proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.main:app", "--host", host, "--port", str(port)],
        cwd=os.getcwd(),
    )

    url = f"http://127.0.0.1:{port}"
    print("  Waiting for server", end="", flush=True)
    for _ in range(60):
        try:
            r = urllib.request.urlopen(f"{url}/api/health", timeout=2)
            if r.status == 200:
                print(" ready!", flush=True)
                break
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    else:
        print("\n  Server failed to start — check console output.", flush=True)
        proc.terminate()
        return

    webbrowser.open(f"http://localhost:{port}")
    print(f"\n  Dashboard: http://localhost:{port}", flush=True)
    print(f"  API docs:  http://localhost:{port}/docs", flush=True)
    print("  Press Ctrl+C to stop.\n", flush=True)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
