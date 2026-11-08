"""英語の G2P（ui/src/engine-web/g2p_en.ts）の照らし合わせ用に、PrettyPitch の svs/g2p_en.py と score_io._apply_g2p_en の
結果を書き出す。

歌詞 1 つずつの読み（単語・句読点・ハイフン・アポストロフィ・大文字・音素の書き方・辞書に無い語・書き方の誤り）と、
単語に続く記号の音符への音節の配り方（+・++・+~・+*・-・+3 など。2 つの読み方 sequential・count）を、決まった例と、
種を決めたランダムな並びで作り、ui/test/golden/g2p_en.json に書く。試験は英語のボイスバンクが無くても動くよう、
en_full の音素表と、使う語だけの発音辞書も一緒に書く。PrettyPitch は手元のボイスバンクの code.prettypitch
（環境変数 PRETTYPITCH_ROOT があればそちら）を使う。

使い方: python tools/golden/dump_g2p_en.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from engine.voicebank import scan  # noqa: E402
from export_web_assets import phoneset_json  # noqa: E402

OUT = ROOT / "ui" / "test" / "golden" / "g2p_en.json"

WORDS = ["a", "I", "the", "love", "again", "night", "skies", "beautiful", "together", "everything", "dreams", "heart",
         "forever", "remember", "yesterday", "tomorrow", "imagination", "strength", "rhythm", "fire", "hour", "world",
         "little", "butterfly", "don't", "can't", "I'm", "you're", "'cause", "rock-and-roll", "well-known", "Hello,",
         "\"Stop!\"", "WORLD", "o'clock", "…yeah", "(oh)", "it’s", "smile", "stars", "into", "away"]
LYRICS = WORDS + [
    "I love you", "hello world", "naïve", "xyzzyq", "love xyzzyq", "--", "...", "'",
    "[hh eh l ow]", "hello[hh eh l ow]", "[HH AH0 L OW1]", " [hh eh l ow] ", "hh eh l ow", "s", "t", "dx", "s t",
    "oh", "ah", "AH0", "[zz qq]", "[]", "[hh EH0 l ow]", "hh zz", "S",
]
MARKS = ["+", "++", "+++", "+~", "+*", "-", "", "ー", "ｰ", "+2", "+3", "+5", "+x", "+-"]


def candidates(lyric: str, d: dict, punct: str) -> set:
    """歌詞の語が辞書で引かれうる見出し（g2p_en._norm_word が試す 2 つ）のうち、辞書にあるもの。"""
    out = set()
    for tok in lyric.split():
        for part in tok.split("-"):
            t = part.replace("’", "'").replace("‘", "'").lower().strip(punct)
            for k in (t, t.strip("'")):
                if k in d:
                    out.add(k)
    return out


def main():
    vbs, _ = scan(ROOT)
    vb = next(iter(vbs.values()))       # PrettyPitch の場所はどのボイスバンクでも同じ
    sys.path.insert(0, str(vb.code["prettypitch"]))
    from duration_model import phoneset as ps_mod
    from svs import g2p_en, score_io

    ps = ps_mod.load("en_full")
    d = g2p_en.load_dict()

    def read(lyric):
        try:
            return {"lyric": lyric, "out": g2p_en.lyric_phonemes(lyric, ps), "err": None}
        except KeyError as e:
            return {"lyric": lyric, "out": None, "err": str(e.args[0])}

    SN = score_io.ScoreNote
    rng = random.Random(20261108)
    seqs = []
    fixed = [["again", "+"], ["night", "+~"], ["skies", "-"], ["beautiful", "+", "+"], ["beautiful", "++"],
             ["beautiful", "+3"], ["beautiful", "+", "+", "+", "+"], ["together", "+", "-", "+"], ["I", "+"],
             ["hello", "+*"], ["[hh eh l ow]", "+"], ["love", None, "+"], ["love", "-", "-"], ["dreams", "+"],
             ["+", "love"], ["imagination", "++", "+"], ["imagination", "+4", "+2"], ["strength", "+"],
             ["fire", "+", "-"], ["everything", "+3", "+"], ["s", "+"], ["xyzzyq", "+"]]
    for f in fixed:
        seqs.append(f)
    for _ in range(3000):
        seq = []
        for _ in range(rng.randint(1, 4)):
            seq.append(rng.choice(WORDS + ["[hh eh l ow]", "[s t aa r]", "s"]))
            for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 4])):
                seq.append(rng.choice(MARKS))
            if rng.random() < 0.15:
                seq.append(None)          # 休符
        seqs.append(seq)

    score_cases = []
    used = set()
    for seq in seqs:
        notes = []
        for x in seq:
            if x is None:
                notes.append(SN(dur_beats=1.0, midi=0, lyric=None, is_rest=True))
                continue
            # 音素の書き方の歌詞は、エンジン（_lyric_phonemes）と同じく音素を先に決めておく
            ph = None
            if x.startswith("[") or x == "s":
                ph = g2p_en.explicit_phonemes(x, ps)
            notes.append(SN(dur_beats=1.0, midi=60, lyric=x, is_rest=False, phonemes=ph))
            used |= candidates(x, d, g2p_en._PUNCT)
        for style in ("sequential", "count"):
            c = {"style": style, "notes": [{"lyric": n.lyric, "is_rest": n.is_rest, "phonemes": n.phonemes} for n in notes]}
            try:
                out = score_io._apply_g2p_en(list(notes), ps, style)
                c["out"] = [{"lyric": n.lyric, "phonemes": n.phonemes} for n in out]
            except KeyError as e:
                c["err"] = str(e.args[0])
            score_cases.append(c)
    for lyr in LYRICS:
        used |= candidates(lyr, d, g2p_en._PUNCT)
    try:
        score_io._apply_g2p_en([], ps, "bogus")
        bad_style = None
    except ValueError as e:
        bad_style = str(e)

    res = {"phoneset": phoneset_json(ps, {}), "dict": {k: d[k] for k in sorted(used)},
           "lyric_cases": [read(x) for x in LYRICS], "score_cases": score_cases, "bad_style": bad_style}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT}: 歌詞 {len(LYRICS)}、並び {len(score_cases)}、辞書 {len(used)} 語")


if __name__ == "__main__":
    main()
