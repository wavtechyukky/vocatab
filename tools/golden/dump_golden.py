"""ブラウザ版（TypeScript）の照らし合わせ用に、Python のエンジンで試験用の楽譜を合成し、各段の入力と出力を書き出す。

エンジンのコードは変えず、段ごとの関数を外から包んで値を横取りする。
出力: ui/test/golden/<楽譜の名前>/case.json（小さい値はそのまま、大きい配列は .bin を指す）と *.bin
  配列は {"$dtype": "float32", "$shape": [...], "$bin": "x.bin"} か {"$dtype": ..., "$shape": ..., "$data": [...]}。
  int64 は int32 に縮めて書く（$dtype は元の型のまま）。bool は uint8。

使い方: python tools/golden/dump_golden.py [ケースの名前 ...]   （リポジトリのどこからでも。名前を省くと全部）
  韓国語のケース（korean）と英語のケース（english・english_count）は、手元のボイスバンク（同梱していない）の名前を
  環境変数 GOLDEN_KO_VB・GOLDEN_EN_VB で渡したときだけ書き出す（node の試験も同じ変数で走らせる）。
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import pipeline as P  # noqa: E402
from engine import voiceparams as VP  # noqa: E402
from engine.melspec import wav_to_mel  # noqa: E402
from engine.voicebank import scan  # noqa: E402

OUT = ROOT / "ui" / "test" / "golden"
INLINE_MAX = 4096


class Saver:
    def __init__(self, d: Path):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def conv(self, v, hint: str = "a"):
        if isinstance(v, np.ndarray):
            if v.dtype.kind in "US":
                return {"$dtype": "str", "$shape": list(v.shape), "$data": v.tolist()}
            dt = str(v.dtype)
            a = v
            if v.dtype == np.int64:
                a = v.astype(np.int32)
            elif v.dtype == np.bool_:
                a = v.astype(np.uint8)
            elif v.dtype == np.float16:
                a = v.astype(np.float32)
                dt = "float16"
            a = np.ascontiguousarray(a)
            if a.size <= INLINE_MAX:
                # JSON に NaN・無限大は書けないので null にする（読む側で NaN に戻す）
                data = [None if isinstance(x, float) and not np.isfinite(x) else x for x in a.reshape(-1).tolist()]
                return {"$dtype": dt, "$shape": list(v.shape), "$data": data}
            self.n += 1
            name = f"{self.n:03d}_{hint}.bin"
            (self.d / name).write_bytes(a.tobytes())
            return {"$dtype": dt, "$shape": list(v.shape), "$bin": name}
        if isinstance(v, (np.floating, float)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if dataclasses.is_dataclass(v):
            return {f.name: self.conv(getattr(v, f.name), f.name) for f in dataclasses.fields(v)}
        if isinstance(v, dict):
            return {str(k): self.conv(x, f"{hint}_{k}") for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self.conv(x, hint) for x in v]
        if isinstance(v, (set, frozenset)):
            return sorted(v)
        return v


def snap(x):
    """後で書き換えられても困らないよう、その時点の値を写す。"""
    return copy.deepcopy(x)


def main():
    vbs, vocs = scan(ROOT)
    vb = next(iter(vbs.values()))       # 既定のボイスバンク（日本語）。ケースの "vb" で替える
    eng = P.PrettyPitchLeapSinger(vb)
    nhv = next(v for v in vocs.values() if v.kind == "nhvsing")
    nsf = next((v for v in vocs.values() if v.kind == "nsf_hifigan"), None)

    rec: dict = {}   # 今の句・合成の横取りの置き場

    def wrap_attr(obj, name, fn):
        orig = getattr(obj, name)
        setattr(obj, name, fn(orig))

    def capture(key, keep_args=None):
        def deco(orig):
            def f(*a, **k):
                out = orig(*a, **k)
                rec[key] = {"args": snap(keep_args(*a, **k)) if keep_args else None, "out": snap(out)}
                return out
            return f
        return deco

    def run_capture(key):
        def deco(orig):
            def f(output_names, feeds, *a, **k):
                out = orig(output_names, feeds, *a, **k)
                names = list(output_names) if output_names else [f"out{i}" for i in range(len(out))]
                rec[key] = {"feeds": snap(dict(feeds)), "outputs": {n: np.asarray(o).copy() for n, o in zip(names, out)}}
                return out
            return f
        return deco

    dp, R, pp = eng.dp, eng.R, eng.pp
    wrap_attr(dp, "split_phrase_syllables", capture("split"))
    wrap_attr(dp, "postprocess_durations", capture("postprocess", lambda ph2, prof, on, cd, **k: {"onset_nat": on, "coda_nat": cd, "core_nat": k.get("core_nat"), "spk_id": k.get("spk_id")}))
    wrap_attr(R, "phrase_start_sec", capture("start_sec"))
    wrap_attr(dp, "to_frames", capture("to_frames", lambda d, fr: {"dur_sec": d, "frame_rate": fr}))
    wrap_attr(R, "_align_durs", capture("align", lambda d, T: {"ph_durs": d, "T": T}))
    wrap_attr(pp, "build_per_frame_arrays", capture("per_frame", lambda a, b, c, n: {"ph_strs": a, "ph_durs": b, "values": c, "n": n}))
    wrap_attr(pp, "correct_f0", capture("correct_f0", lambda out, **k: {"in": out}))
    wrap_attr(R, "mel_postprocess", capture("mel_post", lambda ctx, mel, uv: {"mel": mel, "uv": uv, "frame_rate": ctx.frame_rate}))
    wrap_attr(VP, "spectral", capture("vp_spectral", lambda h, z, eh, en, params, T: {"harm": h, "noise": z, "env_h": eh, "env_n": en, "params": params, "T": T}))
    wrap_attr(VP, "branch_mix", capture("vp_branch", lambda h, z, uv, params, cons: {"h": h, "z": z, "uv_hard": uv, "params": params, "cons": cons}))
    wrap_attr(VP, "finish", capture("vp_finish", lambda wav, params, T: {"wav": wav, "params": params, "T": T}))
    orig_apply = P._apply_shifts

    def apply_wrap(seq, note_index, shifts, pau_min):
        before = np.asarray(seq.dur_sec).copy()
        out = orig_apply(seq, note_index, shifts, pau_min)
        rec["apply_shifts"] = {"dur_before": before, "note_index": list(note_index), "shifts": {str(k): {str(i): s for i, s in v.items()} for k, v in shifts.items()},
                               "pau_min": pau_min, "applied": list(out), "dur_after": np.asarray(seq.dur_sec).copy(),
                               "phonemes": list(seq.phonemes)}
        return out
    P._apply_shifts = apply_wrap

    def wrap_sessions(e):
        wrap_attr(e.s_cons, "run", run_capture("cons_run"))
        wrap_attr(e.s_f0, "run", run_capture("f0_run"))
        wrap_attr(e.s_ac, "run", run_capture("ac_run"))
        wrap_attr(e.vocoder(nhv), "run", run_capture("voc_run"))
        if nsf:
            wrap_attr(e.vocoder(nsf), "run", run_capture("voc_run"))
        return e

    engines = {vb.key: wrap_sessions(eng)}   # ボイスバンクごとのエンジン（ホスト側の処理のモジュールは共通）

    lyr = ["さ", "く", "ら", "さ", "く", "ら", "や", "よ", "い", "の", "そ", "ら"]
    pit = [69, 69, 71, 69, 69, 71, 69, 71, 72, 71, 69, 65]
    sakura = [{"id": i + 1, "onset": 480 * i, "dur": 480, "pitch": p, "lyric": l} for i, (l, p) in enumerate(zip(lyr, pit))]
    mixed = [
        {"id": 1, "onset": 0, "dur": 480, "pitch": 67, "lyric": "あ"},
        {"id": 2, "onset": 480, "dur": 480, "pitch": 69, "lyric": "ー"},
        {"id": 3, "onset": 960, "dur": 480, "pitch": 71, "lyric": "k r a cl sh u"},
        {"id": 4, "onset": 1440 + 120, "dur": 360, "pitch": 72, "lyric": "s t"},       # 前に 0.0625 秒の休符（句は切れない）
        {"id": 5, "onset": 1920, "dur": 720, "pitch": 74, "lyric": "kakiku"},
        {"id": 6, "onset": 2640, "dur": 240, "pitch": 72, "lyric": "ん"},
        {"id": 7, "onset": 2880 + 1920, "dur": 480, "pitch": 64, "lyric": "が"},       # 長い休符の後（次の句）
        {"id": 8, "onset": 5280, "dur": 960, "pitch": 66, "lyric": "っ"},
        {"id": 9, "onset": 6240, "dur": 480, "pitch": 68, "lyric": "しょ"},
        {"id": 10, "onset": 6720, "dur": 480, "pitch": 69, "lyric": "sh"},
    ]
    # 韓国語（G2P・句末の終声）: 音符をまたぐ連音（음 악）、伸ばしへの終声の移動（밤 - -）、鼻音化（국 물）、
    # 濃音化（학 교）、音素の直書き、短い休符で G2P の並びが切れる所（좋 | 다）、1 音符に 2 音節（사랑）、
    # 長い休符の後の次の句の流音化（신 라）と句末の二重パッチム（닭）
    ko_lyr = [(0, 480, 55, "음"), (480, 480, 57, "악"), (960, 480, 55, "밤"), (1440, 480, 57, "-"), (1920, 480, 59, "-"),
              (2400, 480, 57, "국"), (2880, 480, 55, "물"), (3360, 480, 52, "학"), (3840, 480, 55, "교"),
              (4320, 480, 57, "s a L"), (4800, 480, 59, "좋"), (5520, 480, 57, "다"), (6000, 960, 55, "사랑"),
              (8880, 480, 52, "신"), (9360, 480, 55, "라"), (9840, 720, 57, "닭")]
    korean = [{"id": i + 1, "onset": o, "dur": d, "pitch": p, "lyric": l} for i, (o, d, p, l) in enumerate(ko_lyr)]
    # 子音の凹み: 破裂（か ぱ た）・有声の摩擦（ざ）・鳴り子音（な りょ ま）・促音（っ）・破擦（ちゃ）・鼻音の後の有声破裂（ん が）・
    # 無声の摩擦（し）。手で動かした凹みは、深さと広さ・深さだけ・広さだけ・範囲の外（−5 dB と 1 に切る）・浅い側の外（0 dB に切る）・
    # 既定の凹みの無い所（ん の後の が、母音。効かない）。滑舌は句の中で 0〜2 に動かす
    dp_lyr = ["か", "ざ", "な", "っ", "ぱ", "ちゃ", "りょ", "ん", "が", "た", "ま", "し"]
    dp_pit = [67, 69, 71, 72, 72, 74, 72, 71, 69, 67, 69, 67]
    dips = [{"id": i + 1, "onset": 360 * i, "dur": 360, "pitch": p, "lyric": l} for i, (l, p) in enumerate(zip(dp_lyr, dp_pit))]
    # 英語: 単語の音節を続く記号の音符へ配る（+・++・+3）、伸ばし（- と +~。前の母音が続き、伸ばしの記号で学習した F0 には
    # 記号で、acoustic_merge_melisma の音響モデルには前の音素とまとめて渡す）、音素の書き方、子音だけの音符、長い休符の後の次の句。
    # 伸ばしの母音（id 5 の ay）の手で動かした頭は、まとめるボイスバンクでは当てない
    en_lyr = [(0, 480, 64, "I"), (480, 480, 67, "love"), (960, 480, 69, "the"), (1440, 480, 71, "skies"), (1920, 480, 72, "-"),
              (2400, 480, 71, "beautiful"), (2880, 480, 69, "+"), (3360, 480, 67, "+"), (3840, 960, 64, "night"),
              (4800, 480, 62, "+~"), (6240, 480, 64, "[hh eh l ow]"), (6720, 480, 67, "+"), (7200, 480, 69, "imagination"),
              (7680, 480, 71, "++"), (8160, 480, 69, "+"), (8640, 480, 67, "together"), (9120, 480, 65, "+3"),
              (9600, 960, 64, "dreams"), (10560, 480, 62, "s")]
    english = [{"id": i + 1, "onset": o, "dur": d, "pitch": p, "lyric": l} for i, (o, d, p, l) in enumerate(en_lyr)]
    en_vb = os.environ.get("GOLDEN_EN_VB")
    cases = [
        {"name": "sakura", "bpm": 100, "notes": sakura, "singer": 0, "seed": 0},
        {"name": "mixed", "bpm": 120, "notes": mixed, "singer": 1, "seed": 7,
         "retake_f0": {3: 123456789}, "retake_mel": {5: 987654321},
         "ph_shift": {1: {0: -0.08}, 3: {1: 0.02}, 7: {0: -0.2}, 9: {1: -0.03}}},
        {"name": "params", "bpm": 100, "notes": sakura[:6], "singer": 2, "seed": 3, "params": True, "nsf": True},
        {"name": "dips", "bpm": 110, "notes": dips, "singer": 1, "seed": 11, "articulation": True,
         "ph_dip": {1: {0: {"db": -4.0, "w": 0.8}}, 3: {0: {"db": -2.5}}, 5: {0: {"w": 0.3}}, 7: {0: {"db": -6.0, "w": 1.5}},
                    9: {0: {"db": -3.0}}, 10: {0: {"db": 0.5}}, 2: {1: {"db": -3.0}},
                    4: {0: {"db": -4.5}}, 11: {0: {"db": -2.0, "w": 0.5}}}},   # っ は台形のまま（閉鎖）、ま は ν
    ]
    if os.environ.get("GOLDEN_KO_VB"):
        cases.append({"name": "korean", "vb": os.environ["GOLDEN_KO_VB"], "bpm": 120, "notes": korean, "singer": 0, "seed": 5,
                      "ph_shift": {3: {0: -0.05}}})
    if en_vb:
        for name, style, seed in (("english", "sequential", 13), ("english_count", "count", 14)):
            cases.append({"name": name, "vb": en_vb, "bpm": 100, "notes": english, "singer": 0, "seed": seed,
                          "lyric_style": style, "ph_shift": {5: {0: -0.05}, 9: {0: -0.03}}})
    only = set(sys.argv[1:])   # ケースの名前を渡すと、それだけ書き出す

    for c in cases:
        if only and c["name"] not in only:
            continue
        key = c.get("vb", vb.key)
        if key not in vbs:   # 同梱していないボイスバンク（韓国語など）のケースは飛ばす
            print(f"{c['name']} は飛ばす: ボイスバンク {key} が手元に無い")
            continue
        if key not in engines:
            engines[key] = wrap_sessions(P.PrettyPitchLeapSinger(vbs[key]))
        eng = engines[key]
        out_dir = OUT / c["name"]
        sv = Saver(out_dir)
        phrases = eng.phrases_from_notes(c["notes"], c["bpm"], lyric_style=c.get("lyric_style", "sequential"))
        res = {"input": {k: v for k, v in c.items() if k != "name"}, "frame_rate": eng.frame_rate, "phrases": []}
        for pi, (ph, ids) in enumerate(phrases):
            f0_rt = {i: c.get("retake_f0", {}).get(nid, 0) for i, nid in enumerate(ids) if c.get("retake_f0", {}).get(nid, 0)}
            sh = {i: c["ph_shift"][nid] for i, nid in enumerate(ids) if c.get("ph_shift", {}).get(nid)}
            mel_rt = {i: c.get("retake_mel", {}).get(nid, 0) for i, nid in enumerate(ids) if c.get("retake_mel", {}).get(nid, 0)}
            dip = {i: c["ph_dip"][nid] for i, nid in enumerate(ids) if c.get("ph_dip", {}).get(nid)}
            rec.clear()
            an = eng.analyze(ph, c["singer"], c["seed"], f0_retakes=f0_rt, ph_shifts=sh)
            ana = {"phrase": snap(ph), "ids": ids, "f0_retakes": f0_rt, "ph_shifts": sh, **{k: v for k, v in rec.items()}, "result": an}
            # 合成: エディタと同じく、最後の F0 = 抑揚 + ビブラート（画面の曲線は既定のまま）
            T = int(an.ph_durs.sum())
            f0_hz = np.power(2.0, an.plain_log2 + an.vib_log2).astype(np.float32)
            params = {}
            if c.get("params"):
                t = np.linspace(0, 1, T, dtype=np.float32)
                params = {"tension": (0.6 * np.sin(2 * np.pi * t)).astype(np.float32),
                          "breathiness": (0.5 * t).astype(np.float32),
                          "gender": (-0.4 + 0.8 * t).astype(np.float32),
                          "voicing": (1.0 - 0.3 * t).astype(np.float32),
                          "consonant": np.full(T, 0.4, np.float32),
                          "resonance": (0.5 * np.cos(2 * np.pi * t)).astype(np.float32),
                          "growl": (0.6 * t).astype(np.float32),
                          "loudness": (-6 + 6 * t).astype(np.float32)}
            if c.get("articulation"):   # 滑舌: 0（既定の凹み無し）〜 2（2 倍）を行き来する
                t = np.linspace(0, 1, T, dtype=np.float32)
                params["articulation"] = (1.0 + np.sin(2 * np.pi * 1.5 * t)).astype(np.float32)
            syn = []
            for voc in [nhv] + ([nsf] if c.get("nsf") and nsf else []):
                rec.clear()
                eng._ac_cache = P._LRU(64)
                eng._voc_cache = P._LRU(48)
                r = eng.synthesize(an, f0_hz, c["singer"], voc, params=params, want_plain=True, mel_retakes=mel_rt,
                                   ph_dips=dip)
                syn.append({"vocoder": voc.key, "kind": voc.kind, "gain_db": voc.gain_db, "voice_params": voc.voice_params,
                            "f0_hz": f0_hz, "params": params, "mel_retakes": mel_rt, "ph_dips": dip,
                            **{k: v for k, v in rec.items()}, "result": r, "mel_out": wav_to_mel(r["wav"])})
            ana["synth"] = syn
            res["phrases"].append(sv.conv(ana, f"p{pi}"))
            print(f"{c['name']} 句 {pi}: {T} フレーム、音素 {len(an.phonemes)}")
        (out_dir / "case.json").write_text(json.dumps(res, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"書き出し: {OUT}")


if __name__ == "__main__":
    main()
