"""擬似 mel の対数レベル整形（子音の凹み）: 音響 ONNX の `x0_loggain` 入力（フレームごとの ln(gain)、0 = 素のまま）
に渡す曲線を、音素列と持続長から作る。voicebank.yaml の `x0_shape:` が有効なボイスバンクで、engine/pipeline.py の
_acoustic が使う（`apply: mel` なら同じ曲線を、擬似 mel ではなく出力の mel に掛ける）。
ブラウザ版のエンジンは ui/src/engine-web/x0shape.ts（同じ計算。ui/test/x0shape.test.ts で照らし合わせる）。

既定の形は 3 種類:
  - burst: 破裂音・無声破擦音は音素頭で床値に落とし、開放位置で元に戻す（burst_shape: nu なら ν 字の谷）。
    促音 cl は閉鎖のみ（開放は次の子音が受け持つ）。鼻音直後の有声破裂音は閉鎖が有声なので触らない。
  - ν（摩擦）: 有声の摩擦音・破擦音は、区間を ν 字（Hann 型の谷）に軽く凹ませる。
  - ν（鳴り子音）: 鼻音・流音・わたりなども同じく ν 字に凹ませる。

音符ごとの上書き（画面の音素のパネルで動かした凹み）: overrides = {句の中の音素の番号: {"amp": 谷の底の振幅, "w": 広さ}}。
上書きした音素は、既定の形の代わりに、音素の頭から w（音素の長さに対する割合）の区間の ν 字の谷になる
（谷の底 = 区間の真ん中 = 振幅 amp）。上書きはどの子音にも付けられる（既定で触らない s・h などにも）。
dips() は、子音ごとの既定の谷（振幅と広さ）を返す（画面が既定の凹みを描く・上書きの片方だけが決まったときに使う）。
"""
from __future__ import annotations

import numpy as np

# 言語ごとの音素の分類。cy = わたり（y など）を含むラベル（前半 cy_frac だけ整形する）
CLASSES = {
    "ja": dict(
        stop={"k", "ky", "t", "ty", "p", "py", "b", "by", "d", "dy", "g", "gy", "cl"},
        affr={"ch", "ts"},
        fric_nu={"z", "j"},                                  # 無声摩擦（s sh f fy）は元々小さいので触らない
        sonorant={"n", "ny", "m", "my", "y", "r", "ry", "w", "v"},   # h hy も触らない
        voiced_stop={"b", "by", "d", "dy", "g", "gy"},
        nasal={"N", "n", "ny", "m", "my"},
        cy={"ky", "gy", "ty", "dy", "py", "by", "fy", "hy"},
    ),
    "en": dict(
        stop={"b", "d", "dx", "g", "k", "p", "q", "t", "cl"},
        affr={"ch", "tr"},                                   # 無音の閉鎖を作るのは無声破擦のみ
        fric_nu={"z", "zh", "v", "dh", "jh", "dr"},          # 無声摩擦（s sh f th）は触らない（ja と同じ判断）
        sonorant={"m", "n", "ng", "l", "r", "w", "y"},       # hh も触らない
        voiced_stop={"b", "d", "g", "dx"},
        nasal={"m", "n", "ng"},
        cy=set(),
    ),
}

# 既定値は日本語のボイスバンクで聴いて決めたレシピ（voicebanks/leapsinger_ja_3singer/voicebank.yaml）と同じ。
# 谷の底の振幅 0.891 は名目 −1 dB（音響モデルの中で広がるので、出力の凹みはもっと深い）。画面で動かせる限界は −5 dB
DEFAULTS = dict(burst=True, burst_shape="nu", W=6, floor=0.891, pos_stop=0.30, pos_affr=0.10, closure_max=0,
                cy_frac=0.55, nasal_exc=True, fric=True, fric_depth=0.891,
                son=True, son_depth=0.891)
# closure_max: 閉鎖（頭から開放位置まで）の絶対上限（フレーム。0 = 上限なし）。pos_stop などは相対値なので、
# ゆっくりした曲では子音ラベルと一緒に無音も伸びてスタッカート気味になる。実際の閉鎖はテンポに比例しないため。
# burst_shape: "nu"（既定）で破裂・破擦の形を ν 字の谷にする（深さは floor）。"trap" は台形（床つき閉鎖 → 開放で復帰）。
#   谷は頭から開放位置（pos_stop / pos_affr・closure_max で決まる r）まで。促音 cl だけは台形のまま
#   （閉鎖が次の子音まで続くのが正しいので）。
# per_ph: {音素: {floor / pos_stop / pos_affr / depth / off / shape}} の音素ごとの上書き（省略可）。
#   off: true でその音素だけ整形しない。depth は ν 字（fric_depth / son_depth）の谷の振幅の上書き。
#   shape: "nu"/"trap" でその音素だけ形を変える（burst_shape より優先。cl にも効く）。ν の深さは depth、無ければ floor。
MIN_DIP = 3          # ν 字の谷の最短のフレーム数


def classes_of(language: str) -> dict | None:
    return CLASSES.get(language) or CLASSES.get(str(language).split("_")[0])


def params_of(cfg: dict | None) -> dict | None:
    """voicebank.yaml の x0_shape → 既定値で埋めた設定。整形しない（enabled: false）なら None。"""
    p = dict(DEFAULTS)
    p.update(cfg or {})
    return p if p.get("enabled", True) else None


def _pp(p: dict, ph: str) -> dict:
    o = (p.get("per_ph") or {}).get(ph)
    return dict(p, **o) if o else p


# 数値の扱い: 曲線 g は float32 の配列だが、式は float64 で計算し、g に入れる所だけ float32 に丸める。
# ブラウザ版（ui/src/engine-web/x0shape.ts）が同じ順に double で計算して Float32Array に入れるので、結果が揃う
# （float32 の cos・log の実装の違いを持ち込まない）。

def _ramp(g: np.ndarray, s: int, e: int, v0: float, v1: float):
    """g[s:e] を v0 → v1 の余弦（raised-cosine）ランプにする（角がない）"""
    n = e - s
    if n <= 0:
        return
    x = np.linspace(0.0, 1.0, n + 1)[1:]
    g[s:e] = v0 + (v1 - v0) * (1.0 - np.cos(np.pi * x)) / 2.0


def _hann_dip(g: np.ndarray, s: int, e: int, depth: float):
    """g[s:e] に Hann 窓型（ν 字）の谷を掛ける。谷の底の振幅 = depth"""
    if e - s < MIN_DIP:
        return
    x = (np.arange(e - s) + 0.5) / (e - s)
    g[s:e] *= 1.0 - (1.0 - float(depth)) * (1.0 - np.cos(2.0 * np.pi * x)) / 2.0


def _burst_plan(segs, c, p):
    """破裂・破擦の既定の形（音素ごと）: {音素の番号: ("cl_trap"|"trap"|"nu", 谷の終わり r, 振幅, W のランプ幅)}"""
    plan = {}
    W, cy_frac = int(p["W"]), float(p["cy_frac"])
    prev = None
    for i, (ph, s, e) in enumerate(segs):
        cls = "stop" if ph in c["stop"] else ("affr" if ph in c["affr"] else None)
        if cls is None:
            prev = ph
            continue
        if p.get("nasal_exc", True) and ph in c["voiced_stop"] and prev in c["nasal"]:
            prev = ph                                        # 例: ん+が。閉鎖は有声・鼻腔経由なので触らない
            continue
        q = _pp(p, ph)
        prev = ph
        if q.get("off"):
            continue
        floor = float(q["floor"])
        e_eff = s + max(2, int(round(cy_frac * (e - s)))) if ph in c["cy"] else e
        shape = q.get("shape") or (q.get("burst_shape") if ph != "cl" else None)
        if ph == "cl":                                       # 閉鎖のみ。開放は次の子音の整形が受け持つ
            if shape == "nu":                                # per_ph で明示したときだけ cl も ν
                plan[i] = ("nu", e, float(q.get("depth", floor)), 0)
            else:
                plan[i] = ("cl_trap", e, floor, max(1, min(W, (e - s) // 3)))
            continue
        pv = float(q["pos_stop" if cls == "stop" else "pos_affr"])
        cap = int(q.get("closure_max", 0) or 0)
        if shape == "nu":                                    # ν 字の谷。開放位置 r までで復帰（床なし）
            r = min(s + max(3, int(round(pv * (e_eff - s)))), e_eff)
            if cap > 0:
                r = min(r, s + max(3, cap))
            plan[i] = ("nu", r, float(q.get("depth", floor)), 0)
            continue
        w = max(1, min(W, (e_eff - s) // 3))                 # 台形はランプの分 w が開放位置の下限になる
        r = min(s + max(w, int(round(pv * (e_eff - s)))), e_eff)
        if cap > 0:
            r = min(r, s + max(w, cap))
        plan[i] = ("trap", r, floor, w)
    return plan


def _burst(T, segs, plan, skip):
    g = np.ones(T, np.float32)
    for i, (ph, s, e) in enumerate(segs):
        if i not in plan or i in skip:
            continue
        kind, r, amp, w = plan[i]
        v0 = float(g[s - 1]) if s > 0 else 1.0
        if kind == "nu":
            _hann_dip(g, s, r, amp)
        elif kind == "cl_trap":
            _ramp(g, s, s + w, v0, amp)
            g[s + w:e] = amp
        else:
            _ramp(g, s, s + w, v0, amp)
            g[s + w:max(s + w, r - w)] = amp
            _ramp(g, max(s + w, r - w), r, amp, 1.0)
    return g


def _nu_plan(segs, phoneme_set, key, p):
    plan = {}
    for i, (ph, s, e) in enumerate(segs):
        if ph not in phoneme_set:
            continue
        q = _pp(p, ph)
        if q.get("off"):
            continue
        plan[i] = float(q.get("depth", q[key]))
    return plan


def _segs(phonemes, ph_durs):
    segs, pos = [], 0
    for ph, d in zip(phonemes, ph_durs):
        segs.append((ph, pos, pos + int(d)))
        pos += int(d)
    return segs, pos


def _plans(segs, c, p):
    burst = _burst_plan(segs, c, p) if p.get("burst", True) else None
    fric = _nu_plan(segs, c["fric_nu"], "fric_depth", p) if p.get("fric", True) else None
    son = _nu_plan(segs, c["sonorant"], "son_depth", p) if p.get("son", True) else None
    return burst, fric, son


def dips(phonemes, ph_durs, cfg: dict | None, language: str, is_consonant) -> list:
    """子音ごとの既定の谷: 音素ごとに {"amp": 谷の底の振幅（1 = 凹みなし）, "w": 広さ（音素の長さに対する割合）}、
    子音でなければ None。is_consonant(音素) は言語パックの判定（pau・核でない音素）。
    台形（cl の閉鎖など）は、谷の底 = 床、広さ = 頭から開放位置まで、として表す。"""
    c = classes_of(language)
    p = params_of(cfg)
    segs, _ = _segs(phonemes, ph_durs)
    out = [{"amp": 1.0, "w": 1.0} if is_consonant(ph) else None for ph, _, _ in segs]
    if c is None or p is None:
        return out
    burst, fric, son = _plans(segs, c, p)
    for i, (ph, s, e) in enumerate(segs):
        if out[i] is None or e <= s:
            continue
        if burst is not None and i in burst:
            kind, r, amp, _ = burst[i]
            out[i] = {"amp": amp, "w": (r - s) / (e - s)}
        elif fric is not None and i in fric:
            out[i] = {"amp": fric[i], "w": 1.0}
        elif son is not None and i in son:
            out[i] = {"amp": son[i], "w": 1.0}
    return out


def _scale_plans(burst, fric, son, scales: dict):
    """既定の谷の深さに倍率を掛ける（ボイスパラメータの「滑舌」）: 振幅 amp → amp ** m（dB で m 倍）。m = 1 の音素は触らない。"""
    if burst is not None:
        burst = {i: ((k, r, a ** scales[i], w) if scales.get(i, 1.0) != 1.0 else (k, r, a, w))
                 for i, (k, r, a, w) in burst.items()}
    fric = None if fric is None else {i: (a ** scales[i] if scales.get(i, 1.0) != 1.0 else a) for i, a in fric.items()}
    son = None if son is None else {i: (a ** scales[i] if scales.get(i, 1.0) != 1.0 else a) for i, a in son.items()}
    return burst, fric, son


def loggain(phonemes, ph_durs, cfg: dict | None, language: str, overrides: dict | None = None,
            scales: dict | None = None) -> np.ndarray | None:
    """音素列 + 持続長（フレーム）→ ln(gain) [T]（float32）。整形なしなら None。
    cfg = voicebank.yaml の x0_shape（DEFAULTS に上書き）。language は 'ja'・'en' など。
    overrides = {句の中の音素の番号: {"amp": 振幅, "w": 広さ}}（どちらも必須。片方だけのときは dips() の既定で埋めてから渡す）。
    scales = {句の中の音素の番号: 倍率}（既定の谷の深さを dB で何倍にするか。上書きした音素には掛けない）。"""
    c = classes_of(language)
    p = params_of(cfg)
    if c is None or p is None:
        return None
    segs, T = _segs(phonemes, ph_durs)
    ov = {int(k): v for k, v in (overrides or {}).items() if 0 <= int(k) < len(segs)}
    burst, fric, son = _plans(segs, c, p)
    if scales:
        burst, fric, son = _scale_plans(burst, fric, son, {int(k): float(m) for k, m in scales.items()})
    curves = []
    if burst is not None:
        curves.append(_burst(T, segs, burst, ov))
    for plan in (fric, son):
        if plan is not None:
            g = np.ones(T, np.float32)
            for i, amp in plan.items():
                if i not in ov:
                    _hann_dip(g, segs[i][1], segs[i][2], amp)
            curves.append(g)
    if ov:
        g = np.ones(T, np.float32)
        for i, o in sorted(ov.items()):
            _, s, e = segs[i]
            n = min(e - s, max(MIN_DIP, int(round(float(o["w"]) * (e - s)))))
            _hann_dip(g, s, s + n, float(o["amp"]))
        curves.append(g)
    if not curves:
        return None
    g = curves[0]
    for x in curves[1:]:
        g = g * x
    return np.log(np.maximum(g.astype(np.float64), 1e-4)).astype(np.float32)
