"""韓国語の G2P（ui/src/engine-web/g2p_ko.ts）の照らし合わせ用に、PrettyPitch の svs/g2p_ko.py の結果を書き出す。

ハングルの全音節（11,172）から、1〜4 音節の並びを種を決めてランダムに作り（同じ並びは毎回同じ）、Python の
syllables_to_phonemes の結果と一緒に ui/test/golden/g2p_ko.json に書く。2 音節の並びは、次の音節の頭子音
19 種 × 終声 28 種を全部の組み合わせで入れる（音節をまたぐ規則はここで決まる）。

使い方: python tools/golden/dump_g2p_ko.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.voicebank import scan  # noqa: E402

OUT = ROOT / "ui" / "test" / "golden" / "g2p_ko.json"


def main():
    vbs, _ = scan(ROOT)
    vb = next(iter(vbs.values()))   # PrettyPitch の場所はどのボイスバンクでも同じ（韓国語のボイスバンクが無くても動く）
    sys.path.insert(0, str(vb.code["prettypitch"]))
    from svs.g2p_ko import syllables_to_phonemes

    rng = random.Random(20261022)
    syl = lambda cho, jung, jong: chr(0xAC00 + (cho * 21 + jung) * 28 + jong)  # noqa: E731
    seqs = []
    # 終声 28 種 × 次の音節の頭子音 19 種 × 次の母音（ㅏ・ㅣ・ㅑ。ㅣ は口蓋音化、ㅑ は y の落ち方）× 前の母音 2 種
    for jong in range(28):
        for cho in range(19):
            for njung in (0, 20, 2):
                for jung in (0, 19):
                    seqs.append([syl(rng.randrange(19), jung, jong), syl(cho, njung, rng.randrange(28))])
    for _ in range(20000):
        n = rng.randint(1, 4)
        seqs.append([chr(0xAC00 + rng.randrange(11172)) for _ in range(n)])
    cases = [{"in": s, "out": syllables_to_phonemes(s)} for s in seqs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT}: {len(cases)} 並び")


if __name__ == "__main__":
    main()
