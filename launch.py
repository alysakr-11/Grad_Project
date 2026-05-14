import os, sys, subprocess, webbrowser, time
from pathlib import Path

def main():
    os.chdir(Path(__file__).parent)

    # Install deps first so dotenv is available
    backend_reqs = Path("backend/requirements.txt")
    if backend_reqs.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(backend_reqs)], shell=True)

    # Load .env after install
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"  Starting backend on {host}:{port} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", host, "--port", str(port), "--reload"],
        cwd=os.getcwd(),
    )

    time.sleep(2)
    url = f"http://localhost:{port}"
    webbrowser.open(url)

    print(f"  Dashboard: {url}")
    print(f"  API docs:  {url}/docs")
    print("  Press Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
