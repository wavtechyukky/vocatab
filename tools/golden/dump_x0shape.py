"""子音の凹み（ui/src/engine-web/x0shape.ts）の照らし合わせ用に、engine/x0shape.py の結果を書き出す。

日本語・英語の音素からランダムな句（長さもランダム。同じ種なら毎回同じ）を作り、レシピの変種（日本語のボイスバンクの
x0_shape・既定・台形・閉鎖の上限・音素ごとの上書き・整形しない）、手で動かした凹み（overrides）、滑舌の倍率（scales）
を組み合わせて、dips と loggain の結果を ui/test/golden/x0shape.json に書く。

使い方: python tools/golden/dump_x0shape.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import x0shape  # noqa: E402

OUT = ROOT / "ui" / "test" / "golden" / "x0shape.json"
NUCLEUS = {"ja": {"a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N"},
           "en": {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw", "ax"}}
CONS = {"ja": ["k", "ky", "t", "ty", "p", "py", "b", "by", "d", "dy", "g", "gy", "cl", "ch", "ts", "z", "j", "n", "ny",
               "m", "my", "y", "r", "ry", "w", "v", "s", "sh", "h", "hy", "f", "fy"],
        "en": ["b", "d", "dx", "g", "k", "p", "q", "t", "cl", "ch", "tr", "z", "zh", "v", "dh", "jh", "dr", "m", "n", "ng",
               "l", "r", "w", "y", "s", "sh", "f", "th", "hh"]}


def main():
    rng = random.Random(20261108)
    vb_cfg = yaml.safe_load((ROOT / "voicebanks/leapsinger_ja_3singer/voicebank.yaml").read_text(encoding="utf-8"))["x0_shape"]
    variants = [vb_cfg, {}, {"burst_shape": "trap"}, {"burst_shape": "trap", "closure_max": 4, "floor": 0.5},
                {"per_ph": {"k": {"shape": "trap"}, "r": {"depth": 0.7}, "z": {"off": True}, "cl": {"shape": "nu"}},
                 "fric": False, "W": 4, "nasal_exc": False},
                {"son": False, "burst": False, "fric_depth": 0.6},
                {"enabled": False}]
    cases = []
    for n in range(600):
        lang = "ja" if n % 3 else "en"
        vowels = sorted(NUCLEUS[lang])
        phs, durs = ["pau"], [rng.randint(5, 40)]
        for _ in range(rng.randint(1, 12)):
            for _ in range(rng.choice([0, 1, 1, 1, 2])):
                phs.append(rng.choice(CONS[lang]))
                durs.append(rng.randint(0, 25))          # 長さ 0 の音素も混ぜる
            phs.append(rng.choice(vowels))
            durs.append(rng.randint(1, 80))
            if rng.random() < 0.15:
                phs.append("pau")
                durs.append(rng.randint(1, 30))
        phs.append("pau")
        durs.append(rng.randint(5, 40))
        cfg = variants[n % len(variants)]
        is_cons = lambda p: p != "pau" and p not in NUCLEUS[lang]  # noqa: E731
        d = x0shape.dips(phs, durs, cfg, lang, is_cons)
        cons_idx = [j for j, p in enumerate(phs) if is_cons(p)]
        ov = {}
        for j in rng.sample(cons_idx, k=min(len(cons_idx), rng.randint(0, 3))):
            ov[j] = {"amp": float(10 ** (rng.uniform(-5.0, 0.0) / 20)), "w": rng.uniform(0.0, 1.0)}
        sc = {j: rng.choice([0.0, 0.5, 1.0, 1.7, 2.0]) for j in cons_idx if j not in ov and rng.random() < 0.6}
        g = x0shape.loggain(phs, durs, cfg, lang, ov or None, sc or None)
        cases.append({"lang": lang, "cfg": cfg, "phonemes": phs, "durs": durs,
                      "overrides": {str(k): v for k, v in ov.items()}, "scales": {str(k): v for k, v in sc.items()},
                      "dips": d, "loggain": None if g is None else [float(x) for x in np.asarray(g, np.float32)]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"nucleus": {k: sorted(v) for k, v in NUCLEUS.items()}, "cases": cases}), encoding="utf-8")
    print(f"{OUT}: {len(cases)} 句、{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
