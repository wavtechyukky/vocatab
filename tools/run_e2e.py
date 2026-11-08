"""全体の照らし合わせ（ui/test/e2e.ts）をヘッドレスの Chrome で走らせ、結果を出す。

ui/test/e2e.ts を esbuild で ui/test/e2e.js にまとめ、リポジトリの頭を http で配り、e2e.html を開く。
使い方: python tools/run_e2e.py [--cases sakura,mixed,params,dips]
  環境変数 GOLDEN_EN_VB（手元の英語のボイスバンク。tools/golden/dump_golden.py と同じ）があれば、既定のケースに
  english・english_count も足す。
"""
import argparse
import asyncio
import functools
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="sakura,mixed,params,dips" + (",english,english_count" if os.environ.get("GOLDEN_EN_VB") else ""))
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--model-only", default="", help="音響モデルだけを同じ入力で動かして比べる楽譜の名前")
args = ap.parse_args()

r = subprocess.run(["npx", "esbuild@0.25.0", str(ROOT / "ui/test/e2e.ts"), "--bundle", "--format=iife", "--target=es2022",
                    f"--outfile={ROOT / 'ui/test/e2e.js'}"])
if r.returncode:
    sys.exit(r.returncode)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(ROOT)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = 9341
prof = tempfile.mkdtemp(prefix="cdp_")
proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main():
    tabs = []
    for _ in range(50):
        try:
            tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
            break
        except Exception:
            await asyncio.sleep(0.2)
    ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
    async with aiohttp.ClientSession() as s, s.ws_connect(ws_url, max_msg_size=0) as ws:
        n = 0

        async def call(method, params=None):
            nonlocal n
            n += 1
            await ws.send_json({"id": n, "method": method, "params": params or {}})
            while True:
                m = await ws.receive_json()
                if m.get("id") == n:
                    return m

        q = f"model_only={args.model_only}" if args.model_only else f"cases={args.cases}"
        await call("Page.navigate", {"url": f"http://127.0.0.1:{args.port}/ui/test/e2e.html?{q}"})
        t0 = time.time()
        res = None
        while time.time() - t0 < 900:
            rr = await call("Runtime.evaluate", {"expression": "window.__result", "returnByValue": True})
            res = rr["result"].get("result", {}).get("value")
            if res:
                break
            await asyncio.sleep(1.0)
    if not res:
        print("時間内に終わらなかった")
        return 1
    print(res["log"])
    return 1 if res["failures"] else 0


try:
    code = asyncio.run(main())
finally:
    proc.terminate()
    srv.shutdown()
sys.exit(code)
