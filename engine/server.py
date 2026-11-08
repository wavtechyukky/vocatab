"""Python のエンジンのサーバー: 画面（ui/）を配り、エンジンの API を出す。

    python engine/server.py         → http://127.0.0.1:8090/（--port で変える）

配る index.html には window.SVS_ENGINE = "server" を差し込むので、開くだけで画面はこのエンジンを使う
（?engine=web を付けるとブラウザの中のエンジン。そのために web_index.json・voicebanks/・vocoders/ も配る）。

API（配列は float32 を base64 にしたもの。mel は bin ごとに並べた [128 × T]）:
    GET  /api/info        ボイスバンクとボコーダの一覧
    POST /api/analyze     トラックの音符 → フレーズごとの音素・長さ・F0 の部品（plain・vib・uv）
    POST /api/synthesize  フレーズの鍵 ＋ 画面で組み立てた最後の F0 → 波形・mel（① ボコーダの入力、③ 出た音）
    GET  /api/breaths     ボイスバンクのブレスのパターンの一覧（?voicebank=）
    POST /api/breath      ブレス 1 つ（voicebank・index・vocoder）→ 波形（画面がフレーズの波形に足す。声の調整は音量だけ）

フレーズごとの解析の結果は取っておき（中身が同じフレーズは計算し直さない）、synthesize はその鍵で呼ぶ。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from engine.melspec import wav_to_mel  # noqa: E402
from engine.pipeline import PhraseAnalysis, PrettyPitchLeapSinger  # noqa: E402
from engine.voicebank import scan  # noqa: E402

VBS, VOCS = scan(ROOT)
_engines: dict[str, PrettyPitchLeapSinger] = {}
_engines_lock = threading.Lock()
_cache: "OrderedDict[str, tuple[str, PhraseAnalysis]]" = OrderedDict()   # 鍵 → (ボイスバンク, 解析)
_CACHE_MAX = 512

app = FastAPI(title="VocaTab engine")


def engine(key: str) -> PrettyPitchLeapSinger:
    if key not in VBS:
        raise HTTPException(404, f"ボイスバンクがありません: {key}")
    with _engines_lock:
        if key not in _engines:
            # 最初の解析のときにモデルを読む（ふだんは 2 秒ほど。Mac を起動し直した後などの初回は 30 秒ほど）
            print(f"モデルを読み込み中: {VBS[key].name} …", flush=True)
            t0 = time.time()
            try:
                _engines[key] = PrettyPitchLeapSinger(VBS[key])
            except Exception as e:
                # PrettyPitch が無い・言語パックが無い（例: 韓国語は PrettyPitch の対応ブランチが要る）など。
                # 起動は止めず、このボイスバンクを使おうとしたときだけ知らせる
                pp = VBS[key].code.get("prettypitch")
                raise HTTPException(400, f"ボイスバンク「{VBS[key].name}」を読めません: {e}"
                                    f"（PrettyPitch の場所: {pp}。voicebank.yaml の code.prettypitch か"
                                    "環境変数 PRETTYPITCH_ROOT で変えられます）") from e
            print(f"モデルを読み込みました（{time.time() - t0:.1f} 秒）", flush=True)
        return _engines[key]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode("ascii")


def unb64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


@app.get("/api/info")
def info():
    return {"voicebanks": [v.to_json() for v in VBS.values()],
            "vocoders": [v.to_json() for v in VOCS.values()],
            "frame_rate": 44100 / 256, "sample_rate": 44100, "hop": 256}


class AnalyzeReq(BaseModel):
    voicebank: str
    singer: int = 0
    bpm: float
    notes: list[dict]          # {id, onset, dur（刻み、4 分音符 = 480）, pitch, lyric, retake_f0?（種、0 = 元のテイク）,
                               #  ph_shift?（{音符の中の音素の番号: 秒}。手で動かした音素の頭）}
    seed: int = 0


@app.post("/api/analyze")
def analyze(r: AnalyzeReq):
    eng = engine(r.voicebank)
    fr = eng.frame_rate
    try:
        phrases = eng.phrases_from_notes(r.notes, r.bpm)
    except KeyError as e:          # モーラ表に無い歌詞など
        raise HTTPException(400, str(e)) from e
    retake = {n["id"]: int(n.get("retake_f0") or 0) for n in r.notes}
    # 手で動かした音素の頭: 音符の id → {音符の中の音素の番号: 秒}
    shift = {n["id"]: {int(k): float(v) for k, v in (n.get("ph_shift") or {}).items() if v} for n in r.notes}
    out = []
    for ph, ids in phrases:
        f0_rt = {i: retake.get(nid, 0) for i, nid in enumerate(ids) if retake.get(nid, 0)}
        sh = {i: shift[nid] for i, nid in enumerate(ids) if shift.get(nid)}
        sig = [r.voicebank, r.singer, r.seed, round(ph.start_sec, 6),
               [(n.phonemes, n.pitch_midi, round(n.dur_sec, 6), round(n.gap_after_sec, 6)) for n in ph.notes],
               sorted(f0_rt.items()), sorted((i, sorted((k, round(v, 5)) for k, v in d.items())) for i, d in sh.items())]
        key = hashlib.sha1(json.dumps(sig, ensure_ascii=False).encode()).hexdigest()[:16]
        hit = _cache.get(key)
        if hit is None:
            an = eng.analyze(ph, r.singer, r.seed, f0_retakes=f0_rt, ph_shifts=sh)
            _cache[key] = (r.voicebank, an, list(ids))
            if len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
        else:
            an = hit[1]
            _cache.move_to_end(key)
        phs = []
        for j, (p, d, ni) in enumerate(zip(an.phonemes, an.ph_durs, an.note_index)):
            e = {"ph": p, "frames": int(d), "note": ids[min(ni, len(ids) - 1)] if ni >= 0 and ids else None,
                 "shift": float(an.shifts[j]) if an.shifts else 0.0,
                 "nucleus": bool(eng.ps.is_nucleus(p))}   # 画面の色分け（母音など）
            dip = an.dips[j] if an.dips is not None else None
            if dip is not None:                           # 子音の既定の凹み（深さ dB・広さ）。画面が描く
                e["dip"] = {"db": float(20.0 * np.log10(max(dip["amp"], 1e-4))), "w": float(dip["w"])}
            phs.append(e)
        out.append({"key": key, "note_ids": ids, "start_frame": int(round(an.start_sec * fr)),
                    "start_sec": float(an.start_sec), "seed": int(an.seed),   # ブラウザ版（protocol.ts）と同じ項目
                    "n_frames": int(an.ph_durs.sum()), "phonemes": phs,
                    "plain_log2": b64(an.plain_log2), "vib_log2": b64(an.vib_log2), "uv": b64(an.uv),
                    "base_log2": b64(an.base_log2 if an.base_log2 is not None else an.plain_log2)})
    return {"frame_rate": fr, "phrases": out}


class SynthReq(BaseModel):
    voicebank: str
    singer: int = 0
    key: str
    f0_log2: str               # base64 float32 [T]（log2 Hz）
    vocoder: str
    params: dict[str, str] = {}  # ボイスパラメータ（名前 → base64 float32 [T]。無いものは既定値）
    mel_out: bool = True       # 出た音の mel（③）も返すか
    mel_plain: bool = False    # ボイスパラメータを掛けない音の mel（②）も返すか
    retake_mel: dict[str, int] = {}  # 音符の id → 種（AI リテイクの声。0 = 元のテイク）
    # 動かした子音の凹み: 音符の id → {音符の中の音素の番号: {"db": 深さ, "w": 広さ}}（どちらか片方でもよい）
    ph_dip: dict[str, dict[str, dict[str, float]]] = {}


@app.post("/api/synthesize")
def synthesize(r: SynthReq):
    hit = _cache.get(r.key)
    if hit is None:
        raise HTTPException(404, "解析の結果がありません（analyze からやり直してください）")
    if r.vocoder not in VOCS:
        raise HTTPException(404, f"ボコーダがありません: {r.vocoder}")
    eng = engine(hit[0])
    an, ids = hit[1], hit[2]
    f0_hz = np.power(2.0, unb64(r.f0_log2).astype(np.float64)).astype(np.float32)
    params = {k: unb64(v) for k, v in r.params.items()}
    pos = {str(nid): i for i, nid in enumerate(ids)}
    mel_rt = {pos[k]: int(s) for k, s in r.retake_mel.items() if k in pos and int(s)}
    dips = {pos[k]: {int(i): v for i, v in d.items()} for k, d in r.ph_dip.items() if k in pos and d}
    res = eng.synthesize(an, f0_hz, r.singer, VOCS[r.vocoder], params=params, want_plain=r.mel_plain,
                         mel_retakes=mel_rt, ph_dips=dips)
    T = int(an.ph_durs.sum())

    def mel_of(w):
        m = wav_to_mel(w)
        if m.shape[1] < T:
            m = np.pad(m, ((0, 0), (0, T - m.shape[1])), constant_values=np.log(1e-5))
        return b64(m[:, :T])

    out = {"n_frames": T, "sample_rate": 44100, "wav": b64(res["wav"]), "mel_in": b64(res["mel_in"][:, :T])}
    if r.mel_out:
        out["mel_out"] = mel_of(res["wav"])
    if r.mel_plain and res["wav_plain"] is not None:
        out["mel_plain"] = mel_of(res["wav_plain"])
    return out


@app.get("/api/breaths")
def breaths(voicebank: str):
    return engine(voicebank).breath_index()


class BreathReq(BaseModel):
    voicebank: str
    index: int
    vocoder: str


@app.post("/api/breath")
def breath(r: BreathReq):
    eng = engine(r.voicebank)
    if r.vocoder not in VOCS:
        raise HTTPException(404, f"ボコーダがありません: {r.vocoder}")
    idx, _ = eng.load_breaths()
    if not (0 <= r.index < len(idx["clips"])):
        raise HTTPException(404, f"ブレスがありません: {r.index}")
    wav = eng.breath(r.index, VOCS[r.vocoder])
    return {"frames": int(idx["clips"][r.index]["frames"]), "sample_rate": 44100, "wav": b64(wav)}


@app.middleware("http")
async def no_cache(request, call_next):
    """画面のファイル（index.html・dist/app.js など）は毎回新しいか確かめさせる。
    作り直した app.js をブラウザが古いまま使い回さないように（中身が同じなら 304 で済む）。"""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


_ENGINE_TAG = '<script>window.SVS_ENGINE = "server";</script>\n'


@app.get("/")
@app.get("/index.html")
def index_page():
    """画面の index.html に「このページは Python のエンジンを使う」印を差し込んで返す（client.ts の makeEngine が見る）。
    毎回読むので、index.html を直してもサーバーを立ち上げ直さなくてよい。"""
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    anchor = '<script type="module" src="dist/app.js">'
    return HTMLResponse(html.replace(anchor, _ENGINE_TAG + anchor, 1) if anchor in html else html)


# ブラウザの中のエンジン（?engine=web）をこのサーバーで試すときに要るもの。重み（.onnx）も配るので、
# --host を外に開けるときは注意（既定は 127.0.0.1）
@app.get("/web_index.json")
def web_index():
    return FileResponse(ROOT / "web_index.json")


for _sub in ("voicebanks", "vocoders"):
    if (ROOT / _sub).is_dir():
        app.mount(f"/{_sub}", StaticFiles(directory=str(ROOT / _sub)), name=_sub)
app.mount("/", StaticFiles(directory=str(ROOT / "ui"), html=True), name="ui")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    # 動いていても何も出ないと分からないので、開く所と止め方を出す（uvicorn の細かいログは出さない）
    print(f"VocaTab: http://{a.host}:{a.port}/ をブラウザで開いてください（止めるには Ctrl+C）", flush=True)
    print("ボイスバンク: " + ("、".join(v.name for v in VBS.values()) or "なし（voicebanks/ を確かめてください）"), flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
