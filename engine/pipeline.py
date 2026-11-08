"""楽譜 → 音素の長さ・F0 → mel → 波形。PrettyPitch の svs/render.py の段（①〜⑧）と同じ順で、
モデルだけを ONNX（onnxruntime）に置き換えたもの。

    analyze     ① 子音の自然長（cons_dur.onnx）→ ② 持続長の後処理 → ③ F0（f0.onnx）→ ④ ピッチ補正
    synthesize  ⑤ 音響（acoustic.onnx）→ ⑥ mel の後処理 → ⑦ ボコーダ（vocoder.onnx）

ホスト側の処理（G2P・音節分解・floor と圧縮・F0 の baseline と補正・mel の後処理）は PrettyPitch の
Python をそのまま呼ぶ。学習・推論と同じ関数を使うので、入力がずれない（Pages に移すときに TypeScript へ書き直す部分）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

from . import voiceparams as VP
from . import x0shape
from .voicebank import VocoderSpec, Voicebank

PAU = "pau"
HOLD = "-"                                  # 伸ばしの記号の既定（f0.json に hold_symbol が無いとき）
LYRIC_STYLES = ("sequential", "count")      # 英語の + の読み方（PrettyPitch の score_io.LYRIC_STYLES と同じ）


def _session(path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.enable_mem_pattern = False          # PrettyPitch の export/README: 速さは同じで中間のメモリが約 4 割減る
    so.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def _digest(*parts) -> str:
    """配列や値の並びの指紋（段ごとの取り置きの鍵）。"""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.dtype).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


def _retake(noise: np.ndarray, frame_note: np.ndarray, retakes: dict, salt: int) -> None:
    """AI リテイク: 音符 k のフレーム（最後の軸）の雑音だけを、種 retakes[k] の雑音に差し替える（その場で書き換える）。
    種 0 は差し替えない（元のテイク）。"""
    for k, s in retakes.items():
        m = frame_note == int(k)
        if int(s) and m.any():
            alt = np.random.default_rng([int(s), salt]).standard_normal(noise.shape).astype(noise.dtype)
            noise[..., m] = alt[..., m]


MIN_PH_SEC = 0.012  # 手で動かしても、音素はこれより短くしない
PAU_MIN_FRAMES = 4  # pau の後の音素の頭を左へ動かしても、pau はこのフレーム数より短くしない（ボコーダの波形の染み出し）
DIP_DB_MIN = -5.0   # 画面で動かせる子音の凹みの深さの限界（名目 dB。これより深いと出力の mel がほぼ無音になる）
PAU_MIN_EPS = 1e-4  # 秒。フレームへの丸め（境目ごとの round）で float32 の誤差により 3 フレームにならないための足し


def _apply_shifts(seq, note_index: list, shifts: dict, pau_min: float) -> list:
    """手で動かした音素の頭を、自然な長さ（seq.dur_sec）に当てる。頭を s 秒ずらすと、前の音素が s 延び、自分が s 縮む。
    前が pau（句の頭・句の途中の休符）の音素は、最初に作られた pau の中だけで、pau が pau_min 秒より短くならない所まで
    左へ動かせる（言語・歌い手ごとの閾値は要らない）。
    pau の頭（句の最後の音素の終わり・休符の前の終わり）は動かさない。戻り値は音素ごとに実際に当てたずれ（秒）。"""
    d = np.asarray(seq.dur_sec, np.float64).copy()
    applied = [0.0] * len(d)
    count: dict = {}
    for j, ni in enumerate(note_index):
        if ni < 0:
            continue
        idx = count.get(ni, 0)
        count[ni] = idx + 1
        s = (shifts.get(ni) or {}).get(idx)
        if not s or j == 0:
            continue
        if seq.phonemes[j - 1] == PAU:
            lo = min(-(d[j - 1] - pau_min), 0.0)
        else:
            lo = -(d[j - 1] - MIN_PH_SEC)
        s = float(np.clip(s, lo, d[j] - MIN_PH_SEC))
        d[j - 1] += s
        d[j] -= s
        applied[j] = s
    seq.dur_sec = d.astype(np.float32)
    return applied


class _LRU:
    def __init__(self, size: int):
        self.size, self.d = size, OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.size:
            self.d.popitem(last=False)


@dataclass
class PhraseAnalysis:
    """analyze の結果（1 句ぶん）。synthesize はこれと最後の F0 を受け取る。"""
    phonemes: list[str]
    ph_durs: np.ndarray        # 音素ごとのフレーム数（合計 = T）
    start_sec: float           # 句の頭（先頭の pau の頭）を置く絶対時刻
    plain_log2: np.ndarray     # 補正後の抑揚の線（log2 Hz）[T]
    vib_log2: np.ndarray       # ビブラートの成分（log2）[T]
    uv: np.ndarray             # 有声 = 1 [T]
    note_index: list[int]      # 音素ごとの、句の中の音符の番号（pau は -1）
    seed: int = 0              # 乱数の種（F0 の雑音・音響モデルの励振の雑音・ボコーダの雑音）
    shifts: list | None = None  # 音素ごとに当てた頭のずれ（秒。手で動かした分、前後に収まるよう縮めた後）
    # 抑揚の土台（log2 Hz）[T]: 音高の階段を σ 20 ms で丸めたもの（PrettyPitch の baseline）。
    # 抑揚 = plain_log2 − base_log2。画面の「抑揚の強さ」はこの差に掛ける
    base_log2: np.ndarray | None = None
    # 子音ごとの既定の凹み（engine/x0shape.py の dips。{"amp": 谷の底の振幅, "w": 広さ}、子音でなければ None）。
    # 凹みを掛けないボイスバンクでは None
    dips: list | None = None
    # 音素ごとの、伸ばしの音符で前の母音がそのまま続くか（PrettyPitch の PhonemeSequence.held。「skies」「-」の 2 つ目の ay）
    held: list | None = None


class PrettyPitchLeapSinger:
    """ボイスバンク 1 つぶんのモデル一式（ONNX）。歌い手（話者 id）は呼ぶたびに選べる。"""

    def __init__(self, vb: Voicebank):
        self.vb = vb
        pp_root = str(vb.code["prettypitch"])
        if pp_root not in sys.path:
            sys.path.insert(0, pp_root)
        # PrettyPitch のホスト側の処理（torch のモデルは使わない）
        from duration_model import phoneset as ps_mod
        from duration_model.features import extract_phrase_records, records_to_batch
        from duration_model.profile import from_dict
        from svs import duration_postprocess as dp
        from svs import render as R
        from svs import score_io
        from svs.cons_dur import _flatten
        import infer as pp_infer
        self.dp, self.R, self.score_io, self.pp = dp, R, score_io, pp_infer
        self._flatten, self._extract, self._to_batch = _flatten, extract_phrase_records, records_to_batch
        self.ps = ps_mod.load(vb.language)
        m = vb.models
        self.profile = from_dict(json.loads(m["cons_dur_profile"].read_text(encoding="utf-8")))
        self.f0cfg = json.loads(m["f0"].with_suffix(".json").read_text(encoding="utf-8"))
        spk = json.loads(m["speakers"].read_text(encoding="utf-8"))
        self.spk_vec = {int(k): np.asarray(v, np.float32) for k, v in spk["speakers"].items()}
        self.ph2id = {p: i for i, p in enumerate(spk["phonemes"])}
        # 歌詞に直接書ける音素: 音素表のうち F0 と音響の両方のモデルが知っているもの。
        # 空白で区切っていない所を切るときは、短い名前（3 文字まで。Edge・GlottalStop など内部向けの長い名前は除く）の最長一致
        f0_vocab = set(self.f0cfg["phonemes"])
        self.lyric_set = {p for p in self.ps.phonemes if p in self.ph2id and p in f0_vocab}
        self.lyric_greedy = sorted((p for p in self.lyric_set if len(p) <= 3 and p != "EP"), key=len, reverse=True)
        self.frame_rate = float(self.f0cfg["frame_rate"])
        self.s_cons = _session(m["cons_dur"])
        self.s_f0 = _session(m["f0"])
        self.s_ac = _session(m["acoustic"])
        self.ac_inputs = {i.name for i in self.s_ac.get_inputs()}
        # 伸ばしの音符で前の母音がそのまま続く音素（PhraseAnalysis.held）を、音響モデルには前の音素とまとめて 1 つで渡す
        # （F0 には音符ごとのまま）。音高が変わっても母音を切らないラベルで学習したボイスバンク向けで、voicebank.yaml の
        # acoustic_merge_melisma（既定は off）。二重母音を音符ごとに分けて渡すと、音素の中の位置が 2 回 0→1 になり、
        # 移り変わり（ay の a→i）が 2 回鳴るため
        self.merge_melisma = bool(vb.raw.get("acoustic_merge_melisma", False))
        # 伸ばしの記号で書いたラベルで学習した F0 モデル（f0.json の hold_symbol。無ければ音素表に「-」があるもの）には、
        # 伸ばしの母音をその記号で渡す（skies - → s k ay | - z）。ほかのモデルには、今までどおり母音を繰り返して渡す（s k ay | ay z）
        hs = self.f0cfg.get("hold_symbol") or HOLD
        self.f0_hold = hs if hs in self.f0cfg["phonemes"] else None
        # 子音の凹み（voicebank.yaml の x0_shape、engine/x0shape.py）。既定は擬似 mel に掛ける（音響 ONNX の x0_loggain）。
        # apply: mel なら、同じ曲線を出力の mel に掛ける（x0_loggain の無い ONNX でも使える）
        cfg = vb.raw.get("x0_shape")
        self.x0_cfg = cfg if isinstance(cfg, dict) else None
        self.x0_mel = bool(self.x0_cfg) and str(self.x0_cfg.get("apply", "")) == "mel"
        self.x0_on = (self.x0_cfg is not None and (self.x0_mel or "x0_loggain" in self.ac_inputs)
                      and x0shape.params_of(self.x0_cfg) is not None and x0shape.classes_of(vb.language) is not None)
        self.cons_inputs = [i.name for i in self.s_cons.get_inputs()]
        self._vocoders: dict[str, ort.InferenceSession] = {}
        self._ac_cache = _LRU(64)          # ⑤⑥ の結果（mel ①）
        self._voc_cache = _LRU(48)         # ⑦ の結果（調波・雑音・包絡。6 秒の句で約 5 MB）
        self._breaths: tuple[dict, np.ndarray] | None = None   # ブレスの mel の束（breaths.json・breaths.bin）。初めて要るとき読む
        self._breath_cache: dict[tuple[int, str], np.ndarray] = {}
        self.lock = threading.Lock()       # onnxruntime のセッションを 1 本ずつ使う

    # ── ブレス（録音ではなく mel を持ち、ボコーダだけで鳴らす。tools/breaths/build_breath_bank.py が作る） ──
    BREATH_F0 = 220.0   # 全フレーム無声なので調波は出ない。0 を避けるためのダミー
    BREATH_PAD = 4      # 前後に足す ln(1e-5) のフレーム（ボコーダのフィルタの立ち上がりを吸収し、後で切る）
    BREATH_RAMP_SEC = 0.05  # 両端の Tukey 窓の長さ（短い素材では長さの 1/4）

    @classmethod
    def _tukey(cls, sec: np.ndarray, length: float) -> np.ndarray:
        """Tukey 窓の値（0〜1）: 頭と尻尾を ramp 秒ずつ余弦で立ち上げ・下げる（ブラウザ版 core.ts の tukey と同じ）。"""
        r = min(cls.BREATH_RAMP_SEC, length / 4)
        if r <= 0:
            return np.ones_like(sec)
        d = np.minimum(sec, length - sec)
        return np.where(d >= r, 1.0, np.where(d <= 0, 0.0, 0.5 * (1 - np.cos(np.pi * d / r))))

    def load_breaths(self) -> tuple[dict, np.ndarray]:
        if self._breaths is None:
            p = self.vb.breaths
            if p and p.exists():
                idx = json.loads(p.read_text(encoding="utf-8"))
                data = np.frombuffer((p.parent / idx["bin"]).read_bytes(), dtype="<f2")
                self._breaths = (idx, data)
            else:
                self._breaths = ({"clips": []}, np.zeros(0, np.float16))
        return self._breaths

    def breath_index(self) -> dict:
        """画面に渡す一覧（ブラウザ版の core.ts の breaths と同じ中身）。"""
        idx, _ = self.load_breaths()
        return {"clips": [{"id": c["id"], "singer": int(c["singer"]), "frames": int(c["frames"]), "sec": float(c["sec"])}
                          for c in idx["clips"]]}

    def breath(self, index: int, voc: VocoderSpec) -> np.ndarray:
        """ブレス 1 つ → 波形 [256·frames]。ブレスはフレーズのボコーダには入れず（声の調整は音量だけ）、別に鳴らして画面が
        波形で足す。mel に詰め物を付け、f0 一定・uv = 1 でボコーダを通し、声と同じ後段（ボイスパラメータは既定値 → 調波 + 雑音と
        リミッタだけ）を経て詰め物を切り、両端に Tukey 窓を掛ける。ブラウザ版（core.ts の breath）と同じ手順。"""
        key = (int(index), voc.key)
        hit = self._breath_cache.get(key)
        if hit is not None:
            return hit
        idx, data = self.load_breaths()
        c = idx["clips"][index]
        F, P = int(c["frames"]), self.BREATH_PAD
        mel = data[int(c["offset"]):int(c["offset"]) + 128 * F].astype(np.float32).reshape(128, F)
        if c.get("gain_db"):
            mel = (mel + np.float32(np.log(10 ** (float(c["gain_db"]) / 20)))).astype(np.float32)
        T = F + 2 * P
        mel_p = np.pad(mel, ((0, 0), (P, P)), constant_values=np.float32(np.log(1e-5))).astype(np.float32)
        f0 = np.full(T, self.BREATH_F0, np.float32)
        uv_hard = np.zeros(T, np.float32)
        noise = np.random.default_rng([7000 + int(index), 2]).standard_normal((1, 1, T, 256)).astype(np.float32)
        with self.lock:
            harm, nz, wav0, env_h, env_n = self._vocode(voc, mel_p, f0, uv_hard, noise)
        g = np.float32(10 ** (voc.gain_db / 20)) if voc.gain_db else np.float32(1.0)
        if harm is not None and voc.voice_params:
            h, z = VP.spectral(harm, nz, env_h, env_n, {}, T)
            wav = VP.branch_mix(h, z, uv_hard, {}, np.zeros(T, bool)) * g
        else:
            wav = (np.clip(harm + nz, -1.0, 1.0) if wav0 is None else wav0) * g
        wav = VP.finish(wav[:T * 256], {}, T)[P * 256:(P + F) * 256]
        # 両端を Tukey 窓でなだらかに
        length = F * 256 / 44100
        wav = (wav * self._tukey((np.arange(len(wav)) + 0.5) / 44100, length)).astype(np.float32)
        self._breath_cache[key] = wav
        return wav

    # ── 楽譜 → 句 ─────────────────────────────────────────
    def phrases_from_notes(self, notes: list[dict], bpm: float, tpq: int = 480, lyric_style: str = "sequential"):
        """画面の音符（刻みの位置・長さ・音高・歌詞）→ PrettyPitch の Phrase の列と、句ごとの音符の id。

        音符の隙間は休符にする（0.5 秒以上の休符で句を切る）。重なった音符は前の音符を後ろの頭で切る。
        歌詞が空・-・ー・+ で始まる音符は伸ばし（前の母音を続ける）。英語だけは記号をそのまま PrettyPitch へ渡し、
        lyric_style（LYRIC_STYLES）で単語の音節を続く音符へ配る（+ は次の音節、- と +~ は伸ばし、+3 は単語の 3 番目の
        音素から）。ほかの言語は 1 音符に 1 音節（モーラ・音節）なので、+ で始まる記号はどれも伸ばし。
        """
        if lyric_style not in LYRIC_STYLES:
            raise ValueError(f"lyric_style は {LYRIC_STYLES} のどれか（{lyric_style!r}）")
        en = self.ps.g2p == "en"
        SN = self.score_io.ScoreNote
        ns = sorted(notes, key=lambda n: n["onset"])
        score, ids, cursor = [], [], 0
        for k, n in enumerate(ns):
            onset = int(n["onset"])
            end = onset + int(n["dur"])
            if k + 1 < len(ns):
                end = min(end, int(ns[k + 1]["onset"]))
            if onset > cursor:
                score.append(SN(dur_beats=(onset - cursor) / tpq, midi=0, lyric=None, is_rest=True))
            onset = max(onset, cursor)
            if end <= onset:
                continue
            lyric = str(n.get("lyric") or "").strip()
            melisma = lyric in ("", "-", "ー") or lyric.startswith("+")
            score.append(SN(dur_beats=(end - onset) / tpq, midi=int(n["pitch"]),
                            lyric=(lyric if en else None) if melisma else lyric,
                            phonemes=None if melisma else self._lyric_phonemes(lyric)))
            ids.append(n["id"])
            cursor = end
        # lyric_style は英語でだけ渡す（英語の G2P の無い前の版の PrettyPitch でも、ほかの言語はそのまま動くように）
        kw = {"lyric_style": lyric_style} if en else {}
        phrases = self.score_io.score_to_phrases(score, bpm, phoneset=self.ps, **kw)
        out, i = [], 0
        for ph in phrases:
            out.append((ph, ids[i:i + len(ph.notes)]))
            i += len(ph.notes)
        return out

    def _lyric_phonemes(self, lyric: str) -> list[str] | None:
        """英数字だけの歌詞を音素の列にする（仮名を含む歌詞は None。モーラ表に任せる）。
        音素はいくつ並んでもよい（核が複数「k a k i k u」・子音だけ「sh」「s t」・ローマ字の綴りにならない「k r a cl sh u」）。
        空白で区切った記号が音素ならそのまま使い、区切っていない所（「kakiku」「shu」）は音素の名前の最長一致で切る。
        英語は、音素の書き方（「[hh eh l ow]」・空白で区切った 2 つ以上の音素・子音 1 つ）ならその音素、それ以外は None
        （単語は PrettyPitch の英語の G2P が発音辞書で引き、続く + の音符へ音節を配る。辞書に無い語はそこで知らせる）。"""
        if self.ps.g2p == "en":
            from svs.g2p_en import explicit_phonemes
            ph = explicit_phonemes(lyric, self.ps)
            bad = [p for p in ph or [] if p not in self.lyric_set]
            if bad:
                raise KeyError(f"歌詞「{lyric}」の音素「{' '.join(bad)}」はこのボイスバンクのモデルにありません")
            return ph
        if not lyric.isascii():
            return None
        out: list[str] = []
        for tok in lyric.split():
            if tok in self.lyric_set:
                out.append(tok)
                continue
            i = 0
            while i < len(tok):
                m = next((p for p in self.lyric_greedy if tok.startswith(p, i)), None)
                if m is None:
                    raise KeyError(f"歌詞「{lyric}」の「{tok[i:]}」を音素として読めません")
                out.append(m)
                i += len(m)
        return out or None

    # ── ① 子音の自然長（ONNX）──────────────────────────────
    def _cons_naturals(self, phrase, spk_id: int):
        """svs.cons_dur.predict_cons_naturals と同じ組み立てで、モデルだけ cons_dur.onnx にしたもの。"""
        dp, ps = self.dp, self.ps
        onset_nat, coda_nat, core_nat = [], [], []
        for note in phrase.notes:
            onset, nuc, coda = dp.split_syllable(note.phonemes, ps)
            core_only = bool(nuc) and not any(ps.is_nucleus(v) or ps.is_pau(v) for v in nuc)
            onset_nat.append([dp.FALLBACK_CONS_DUR] * len(onset))
            coda_nat.append([dp.FALLBACK_CONS_DUR] * len(coda))
            core_nat.append([dp.FALLBACK_CONS_DUR] * len(nuc) if core_only else [])
        if not phrase.notes:
            return onset_nat, coda_nat, core_nat
        slots = {"onset": onset_nat, "coda": coda_nat, "core": core_nat}
        phs, durs, vals, where = self._flatten(phrase, ps)
        recs = [r for r in self._extract(phs, durs, vals, int(spk_id), 0, phoneset=ps) if r["idx"] in where]
        if recs:
            batch = self._to_batch(recs, device="cpu", phoneset=ps)
            feed = {}
            for name, meta in zip(self.cons_inputs, self.s_cons.get_inputs()):
                a = batch[name].cpu().numpy()
                feed[name] = a.astype(np.int64 if "int64" in meta.type else np.float32)
            pred = self.s_cons.run(["dur_sec"], feed)[0]
            for r, d in zip(recs, pred):
                j, slot, k = where[r["idx"]]
                slots[slot][j][k] = float(d)
        return onset_nat, coda_nat, core_nat

    # ── ③④ F0（ONNX）と補正 ───────────────────────────────
    def _f0(self, seq, spk_id: int, seed: int, note_index=None, retakes=None, held=None) -> dict:
        """PrettyPitch の generate_f0 + correct_f0 と同じ。モデルの部分だけ f0.onnx。
        ただし土台は学習時と同じく核音素の音高だけから作る（render の譜面どおりだと子音にも音高が付く）。
        retakes: {句の中の音符の番号: 種}。その音符のフレームだけ、拡散の雑音（x_T と各段の雑音）を差し替える。
        held: 音素ごとの、伸ばしの音符で前の母音が続くか（PhraseAnalysis.held）。f0_hold のモデルにはそこを伸ばしの記号で渡す。"""
        pp, cfg = self.pp, self.f0cfg
        score = self.R._f0_score_from_seq(seq, spk_id=spk_id, style_id=0)
        if self.f0_hold and held:
            for slot, h in zip(score["phonemes"], held):
                if h:
                    slot["ph"] = self.f0_hold      # 音高（midi）はその音符のまま。伸ばしの記号は核（f0.json の nucleus）
        ph_strs, ph_ids, dur_sec, values = pp._read_score(score, cfg)
        T, ph_durs = pp._frame_grid(dur_sec, self.frame_rate)
        from prettypitch.baseline import fill_values
        is_nuc = np.array([p in frozenset(cfg["nucleus"]) for p in ph_strs])
        # 土台は学習時と同じく核音素の音高だけから作る（子音は前の核の音高を引き継ぎ、段は核の頭に来る）。
        # render の譜面は子音にもその音符の音高を付けるので、そのまま使うと段が子音の頭に来て、
        # 有声子音の所で F0 が音程差ぶんずれる（抑揚は学習時の土台からのずれとして学習されている）
        sigma = float(cfg.get("baseline_sigma_sec", 0.02))
        baseline = pp.build_baseline(np.where(is_nuc, values, np.nan), dur_sec, T, self.frame_rate, sigma_sec=sigma)
        if baseline is None:  # 核音素の無い句（子音だけの音符など）: 子音の音高で作る。それも無ければ一定
            baseline = pp.build_baseline(values, dur_sec, T, self.frame_rate, sigma_sec=sigma)
        if baseline is None:
            baseline = np.full(T, np.log2(440.0) + (float(cfg["pitch_center"]) - 69.0) / 12.0, np.float32)
        filled = fill_values(np.where(is_nuc, values, np.nan))
        pitch_midi = np.where(np.isfinite(filled), filled, float(cfg["pitch_center"])).astype(np.float32)
        noise = np.random.default_rng(seed).standard_normal((1, int(cfg["n_noise"]), 2, T)).astype(np.float32)
        if retakes and note_index is not None and len(note_index) == len(ph_durs):
            _retake(noise, np.repeat(np.asarray(note_index), ph_durs)[:T], retakes, salt=3)
        n_styles = int(cfg.get("n_styles", 1))
        style_mix = np.zeros((1, n_styles), np.float32)
        style_mix[0, 0] = 1.0
        plain, vib, vuv = self.s_f0.run(["plain", "vibrato", "vuv"], {
            "tokens": ph_ids[None].astype(np.int64), "pitch_midi": pitch_midi[None],
            "durations": ph_durs[None].astype(np.int64), "spk_id": np.array([spk_id], np.int64),
            "style_mix": style_mix, "noise": noise})
        n = min(T, plain.shape[1])
        plain_delta = plain[0, :n].astype(np.float64)          # plain_std・vib_std はグラフの中で掛け済み
        vib_delta = vib[0, :n].astype(np.float64)
        baseline = baseline[:n]
        phonemes_pf, note_pitches_pf, note_bnd_pf = pp.build_per_frame_arrays(ph_strs, ph_durs, values, n)
        frame_is_pau = np.array([p in pp._SILENCE_PHS for p in phonemes_pf])
        plain_hz = np.power(2.0, baseline + plain_delta).astype(np.float32)
        out = {
            "f0_hz": (plain_hz * np.power(2.0, vib_delta)).astype(np.float32),
            "uv": ((vuv[0, :n] > 0.5) & ~frame_is_pau).astype(np.float32),
            "baseline_log2": baseline.astype(np.float32), "plain_delta": plain_delta.astype(np.float32),
            "vib_delta": vib_delta.astype(np.float32), "note_boundaries_pf": np.asarray(note_bnd_pf, bool),
            "frame_rate": np.float32(self.frame_rate), "plain_hz": plain_hz,
            "phonemes_pf": np.asarray(phonemes_pf), "note_pitches_pf": np.asarray(note_pitches_pf, np.float64),
            "frame_is_pau": frame_is_pau, "nucleus_phs": frozenset(cfg["nucleus"]),
        }
        return pp.correct_f0(out)

    def analyze(self, phrase, spk_id: int, seed: int = 0, f0_retakes: dict | None = None,
                ph_shifts: dict | None = None) -> PhraseAnalysis:
        """①〜④: 1 句の音素・長さ・F0 の部品。
        f0_retakes: {句の中の音符の番号: 種}（AI リテイク。その音符の所だけ F0 の乱数を差し替える）
        ph_shifts: {句の中の音符の番号: {音符の中の音素の番号: 秒}}（手で動かした音素の頭。自然な長さからのずれ）"""
        dp, ps, R = self.dp, self.ps, self.R
        with self.lock:
            ph2 = dp.split_phrase_syllables(phrase, ps)
            on, cd, co = self._cons_naturals(ph2, spk_id)
            seq = dp.postprocess_durations(ph2, self.profile, on, cd, core_nat=co, rest_before=1e9,
                                           phoneset=ps, spk_id=spk_id, style_id=0)
            # 句を置く位置は、手で動かす前の長さで決める（先頭の母音の頭を動かしても句ごと逆にずれないように）
            start_sec, _ = R.phrase_start_sec(ph2, seq, ps)
            # 音素 → 句の中の音符の番号（pau は -1）。音節の頭（note_event_starts）を数え、元の音符へ戻す。
            # 核が複数ある音符（「k a k i k u」など）は split_phrase_syllables で音節ごとに分かれているので、
            # 分けた後の番号のままだと後ろの音符の音素がずれる
            syl2note = [j for j, n in enumerate(phrase.notes) for _ in range(len(dp._syllabify(n.phonemes, ps)))]
            if len(syl2note) != len(ph2.notes):
                syl2note = list(range(len(ph2.notes)))
            note_index, k = [], -1
            for ph, st in zip(seq.phonemes, seq.note_event_starts):
                if st:
                    k += 1
                note_index.append(-1 if ph == PAU else syl2note[min(max(k, 0), len(syl2note) - 1)])
            # 伸ばしの音符の頭で、前の音符の母音がそのまま続く音素（PrettyPitch が楽譜の伸ばしの記号から決める。
            # held の無い前の版の PrettyPitch では、どこも続きとみなさない）
            h = getattr(seq, "held", None)
            held = [bool(x) for x in h] if h is not None else [False] * len(seq.phonemes)
            if self.merge_melisma and ph_shifts:
                # まとめる母音は画面でも前の音素につなげて見せ、頭を動かせないので、残っていたずれは当てない
                cnt, drop = {}, set()
                for ni, x in zip(note_index, held):
                    if ni >= 0:
                        cnt[ni] = cnt.get(ni, 0) + 1
                        if x:
                            drop.add((ni, cnt[ni] - 1))
                ph_shifts = {ni: {k: v for k, v in d.items() if (ni, k) not in drop} for ni, d in ph_shifts.items()}
            # pau の後の音素は、最初に作られた pau の中だけで左へ伸ばせる（pau は PAU_MIN_FRAMES 残す）
            applied = _apply_shifts(seq, note_index, ph_shifts or {}, PAU_MIN_FRAMES / self.frame_rate + PAU_MIN_EPS)
            ph_durs, _ = dp.to_frames(seq.dur_sec, self.frame_rate)
            f0o = self._f0(seq, spk_id, seed, note_index, f0_retakes, held)  # F0 も動かした長さで作る
        T = len(f0o["f0_hz"])
        ph_durs = np.asarray(R._align_durs(ph_durs, T), np.int64)
        dips = None
        if self.x0_on:
            dips = x0shape.dips(list(seq.phonemes), ph_durs, self.x0_cfg, self.vb.language,
                                lambda p: p != PAU and not ps.is_nucleus(p))
        return PhraseAnalysis(
            phonemes=list(seq.phonemes), ph_durs=ph_durs, start_sec=float(start_sec),
            plain_log2=np.log2(np.maximum(f0o["plain_hz"], 1e-3)).astype(np.float32),
            vib_log2=np.asarray(f0o["vib_delta"], np.float32), uv=np.asarray(f0o["uv"], np.float32),
            note_index=note_index, seed=int(seed), shifts=applied,
            base_log2=np.asarray(f0o["baseline_log2"], np.float32), dips=dips, held=held)

    def dip_overrides(self, an: PhraseAnalysis, ph_dips: dict | None) -> dict:
        """画面で動かした凹み {句の中の音符の番号: {音符の中の音素の番号: {"db": 深さ, "w": 広さ}}}
        → x0shape.loggain の overrides {句の中の音素の番号: {"amp": 振幅, "w": 広さ}}。片方だけのときは既定で埋める。
        深さ（dB）は擬似 mel に掛ける名目の値。出力の凹みは子音・前後の音・歌い手で大きく変わるので、画面は dB ではなく
        相対的な深さ（%。DIP_DB_MIN = 100%）で見せる。"""
        if not ph_dips or not self.x0_on or an.dips is None:
            return {}
        out, count = {}, {}
        for j, ni in enumerate(an.note_index):
            if ni < 0:
                continue
            idx = count.get(ni, 0)
            count[ni] = idx + 1
            o = (ph_dips.get(ni) or {}).get(idx)
            d = an.dips[j]
            if not o or d is None:
                continue
            db, w = o.get("db"), o.get("w")
            amp = d["amp"] if db is None else float(10.0 ** (min(0.0, max(DIP_DB_MIN, float(db))) / 20.0))
            w = d["w"] if w is None else float(min(1.0, max(0.0, float(w))))
            out[j] = {"amp": amp, "w": w}
        return out

    # ── ⑤⑥⑦ 音響・mel の後処理・ボコーダ ───────────────────
    def vocoder(self, spec: VocoderSpec) -> ort.InferenceSession:
        if spec.key not in self._vocoders:
            self._vocoders[spec.key] = _session(spec.onnx)
        return self._vocoders[spec.key]

    def synthesize(self, an: PhraseAnalysis, f0_hz: np.ndarray, spk_id: int, voc: VocoderSpec,
                   params: dict | None = None, want_plain: bool = False, mel_retakes: dict | None = None,
                   ph_dips: dict | None = None) -> dict:
        """⑤〜⑦: 最後の F0（画面で組み立てたもの）とボイスパラメータ（フレームごと）から mel と波形を作る。

        params: {名前: [T] の配列}（voiceparams.NEUTRAL の名前。無いものは既定値）
        want_plain: ボイスパラメータを掛けない音（mel の ② 用）も返す
        ph_dips: 画面で動かした子音の凹み（dip_overrides の入力の形）
        戻り値: mel_in（ボコーダに入れた mel）[128, T]、wav [256·T]、wav_plain（頼まれたとき）、uv_hard [T]
        """
        params = params or {}
        T = int(an.ph_durs.sum())
        f0 = np.asarray(f0_hz, np.float32)[:T]
        if len(f0) < T:
            f0 = np.pad(f0, (0, T - len(f0)), mode="edge")
        tokens, durs = self._acoustic_units(an)
        frame_ph = np.repeat(np.asarray(an.phonemes), an.ph_durs)[:T]
        uv_hard = ((an.uv[:T] > 0.5) & (frame_ph != PAU)).astype(np.float32)
        cons = np.array([not (p == PAU or self.ps.is_nucleus(p)) for p in frame_ph], bool)  # 子音のフレーム
        # 段ごとの取り置き: ⑤⑥ は F0・歌い手が同じなら、⑦ はさらにボコーダが同じなら使い回す
        # （ボイスパラメータはどれもボコーダの出力の後で掛けるので、動かしても混ぜ直すだけになる）
        rt = {int(k): int(s) for k, s in (mel_retakes or {}).items() if int(s)}
        ov = self.dip_overrides(an, ph_dips)
        sc = self.dip_scales(an, params.get("articulation"), ov)
        with self.lock:
            ac_key = _digest(an.seed, spk_id, tokens, durs, f0, sorted(rt.items()),
                             sorted((j, round(o["amp"], 6), round(o["w"], 6)) for j, o in ov.items()),
                             sorted((j, round(m, 4)) for j, m in sc.items()))
            mel_in = self._ac_cache.get(ac_key)
            if mel_in is None:
                mel_in = self._acoustic(an, tokens, durs, f0, spk_id, uv_hard, rt, ov, sc)
                self._ac_cache.put(ac_key, mel_in)
            voc_key = _digest(ac_key, voc.key)
            hit = self._voc_cache.get(voc_key)
            if hit is None:
                # 雑音も句の種から（同じ句なら毎回同じ音）
                vnoise = np.random.default_rng([an.seed, 2]).standard_normal((1, 1, T, 256)).astype(np.float32)
                hit = self._vocode(voc, mel_in, f0, uv_hard, vnoise)
                self._voc_cache.put(voc_key, hit)
            harm, noise, wav0, env_h, env_n = hit
        g = np.float32(10 ** (voc.gain_db / 20)) if voc.gain_db else np.float32(1.0)
        if harm is not None and voc.voice_params:
            # ② 用はボイスパラメータを掛けない音
            plain = np.clip(harm + noise, -1.0, 1.0)[:T * 256] * g if want_plain else None
            h, z = VP.spectral(harm, noise, env_h, env_n, params, T)       # テンション・ジェンダー・響き
            wav = VP.branch_mix(h, z, uv_hard, params, cons) * g            # 枝ごとの音量（ブレシネスなど）
        else:
            if wav0 is None:
                wav0 = np.clip(harm + noise, -1.0, 1.0)
            wav = wav0 * g                                                   # 枝の無いボコーダ（NSF-HiFiGAN）
            plain = wav0[:T * 256] * g if want_plain else None
        wav = VP.finish(wav[:T * 256], params, T)                           # うなり・音量・リミッタ
        return {"mel_in": np.asarray(mel_in, np.float32), "wav": wav, "wav_plain": plain, "uv_hard": uv_hard}

    def dip_scales(self, an: PhraseAnalysis, articulation, overrides: dict) -> dict:
        """ボイスパラメータの「滑舌」（フレームごとの倍率 [T]）→ 子音ごとの既定の凹みの倍率 {句の中の音素の番号: 倍率}。
        子音の真ん中のフレームの値を使う。手で動かした子音（overrides）と、倍率 1 の子音は入れない。
        滑舌が来なければ既定値（voiceparams.NEUTRAL、0 = 既定の凹み無し）。"""
        if not self.x0_on or an.dips is None:
            return {}
        a = np.asarray(articulation if articulation is not None else [VP.NEUTRAL["articulation"]], np.float32)
        out, f = {}, 0
        for j, d in enumerate(an.ph_durs):
            d = int(d)
            if an.dips[j] is not None and j not in overrides and d > 0 and len(a):
                m = float(min(2.0, max(0.0, a[min(len(a) - 1, f + d // 2)])))
                db0 = 20.0 * np.log10(max(an.dips[j]["amp"], 1e-4))
                if db0 < 0.0:
                    m = min(m, DIP_DB_MIN / db0)           # 倍率を掛けても −5 dB より深くしない（画面と同じ）
                if m != 1.0:
                    out[j] = m
            f += d
        return out

    def _acoustic_units(self, an: PhraseAnalysis) -> tuple[np.ndarray, np.ndarray]:
        """⑤ に渡す音素の番号と長さ（フレーム）。merge_melisma なら、伸ばしの音符で前の母音が続く音素（an.held）を
        前の音素にまとめる（skies + - → s k ay | ay z を s k ay z、ay は 2 つの音符の長さの合計）。
        フレームごとの音素は変わらない（同じ母音どうしをつなぐだけ）。"""
        ids = [self.ph2id[p] for p in an.phonemes]
        durs = [int(d) for d in an.ph_durs]
        if self.merge_melisma and an.held:
            t, d = [], []
            for i, n, h in zip(ids, durs, an.held):
                if h and t:
                    d[-1] += n
                else:
                    t.append(i)
                    d.append(n)
            ids, durs = t, d
        return np.asarray(ids, np.int64), np.asarray(durs, np.int64)

    def _acoustic(self, an: PhraseAnalysis, tokens, durs, f0, spk_id: int, uv_hard, retakes=None, dips=None,
                  scales=None) -> np.ndarray:
        """⑤ 音響モデル → ⑥ mel の後処理。戻り値はボコーダに入れる mel [128, T]。
        tokens・durs: _acoustic_units の音素と長さ。
        retakes: {句の中の音符の番号: 種}（AI リテイク。その音符の所だけ励振の雑音を差し替える）
        dips: 子音の凹みの上書き（x0shape.loggain の overrides）、scales: 既定の凹みの倍率（滑舌）。
        凹みはまとめる前の音素（an.phonemes）で作る（番号は画面の音素と同じ。まとめるのは母音どうしなので曲線は変わらない）"""
        T = len(f0)
        feeds = {"tokens": tokens[None], "durations": durs[None], "f0": np.asarray(f0, np.float32)[None],
                 "spk_embed": self.spk_vec[spk_id][None]}
        if "exc_noise" in self.ac_inputs:  # 励振の雑音も句の種から
            z = np.random.default_rng([an.seed, 1]).standard_normal((1, T * 256)).astype(np.float32)
            if retakes:
                fn = np.repeat(np.repeat(np.asarray(an.note_index), an.ph_durs)[:T], 256)
                _retake(z, fn, retakes, salt=1)
            feeds["exc_noise"] = z
        # 子音の凹み: フレームごとの ln(gain)。既定は擬似 mel に（x0_loggain）、apply: mel なら出力の mel に足す
        g = x0shape.loggain(an.phonemes, an.ph_durs, self.x0_cfg, self.vb.language, dips, scales) if self.x0_on else None
        if "x0_loggain" in self.ac_inputs:
            feeds["x0_loggain"] = (g[:T] if g is not None and not self.x0_mel else np.zeros(T, np.float32))[None]
        mel = self.s_ac.run(["mel"], feeds)[0]
        mel = np.asarray(mel[0], np.float32).T                                       # [128, T]
        mel_in = self.R.mel_postprocess(SimpleNamespace(frame_rate=self.frame_rate), mel, uv_hard)
        if self.x0_mel and g is not None:
            mel_in = mel_in + g[:T][None, :]
        return np.asarray(mel_in, np.float32)

    def _vocode(self, voc: VocoderSpec, mel_in: np.ndarray, f0: np.ndarray, uv_hard: np.ndarray,
                noise: np.ndarray | None = None):
        """mel（hop 256）→ (調波, 雑音, 波形, 調波の包絡, 雑音の包絡)。枝の無いボコーダは波形のほかが None。
        ボコーダの種類ごとの入力の形の違いをここで吸収する。ONNX が雑音の入力と包絡の出力
        （tools/export_nhv_params.py）を持っていれば使う。"""
        sess = self.vocoder(voc)
        if voc.kind == "nhvsing":
            names = {i.name for i in sess.get_inputs()}
            T = mel_in.shape[1]
            feeds = {"mel": np.ascontiguousarray(mel_in.T[None], np.float32),
                     "f0": f0[None, None], "uv": (1.0 - uv_hard)[None, None]}
            if "noise" in names:
                feeds["noise"] = noise if noise is not None else \
                    np.random.default_rng(0).standard_normal((1, 1, T, 256)).astype(np.float32)  # [1,1,T,hop]
            outs = sess.run(None, feeds)
            h, z = (np.asarray(o, np.float32).reshape(-1) for o in outs[1:3])
            # 取り置きを小さくするため、波形は調波 + 雑音から作り直せるので持たず、包絡は float16 で持つ
            env = [np.asarray(o[0], np.float16) for o in outs[3:5]] if len(outs) >= 5 else [None, None]
            return h, z, None, env[0], env[1]
        if voc.kind == "nsf_hifigan" and voc.hop == 512:
            # hop 256 → 512: 隣の 2 フレームを平均（ln-mel の平均、F0 は log の平均）。奇数なら最後を繰り返す
            m, f = mel_in, f0
            if m.shape[1] % 2:
                m = np.concatenate([m, m[:, -1:]], axis=1)
                f = np.concatenate([f, f[-1:]])
            m2 = 0.5 * (m[:, 0::2] + m[:, 1::2])
            f2 = np.exp2(0.5 * (np.log2(np.maximum(f[0::2], 1.0)) + np.log2(np.maximum(f[1::2], 1.0))))
            out = sess.run(["waveform"], {"mel": np.ascontiguousarray(m2.T[None], np.float32),
                                          "f0": f2[None].astype(np.float32)})[0]
            return None, None, np.asarray(out, np.float32).reshape(-1), None, None
        raise ValueError(f"このボコーダの種類には対応していません: {voc.kind}（hop {voc.hop}）")
