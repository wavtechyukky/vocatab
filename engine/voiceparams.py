"""ボイスパラメータのうち、ボコーダの出力（調波・雑音の波形）より後で掛けるもの。
ボコーダを回し直さずに済むので、スライダーを動かしても混ぜ直すだけで済む。

NHVSing の ONNX は調波（harmonic）と雑音（noise）の波形と、フィルタの滑らかな包絡を別々に返す。
周波数ごとの利得（STFT、窓 1024・hop 256 で掛ける。spectral）:
    テンション t（−1〜1）  調波に (f / 1 kHz)^(β·t)（β = TENSION_BETA で 1 オクターブあたり 6β dB の傾き）。
                          傾けた後の調波のエネルギーがフレームごとに変わらないよう戻す
    ジェンダー g（−1〜1）  両方の枝に exp(包絡を伸び縮みさせたもの − 包絡)。0〜4 kHz は周波数を一様に α 倍
                          （出力の f が元の α·f を読む）、4 kHz〜22.05 kHz は mel 軸の上で α·4 kHz から 22.05 kHz まで
                          直線でつなぐ（22.05 kHz では変形しない）。α = 2^(GENDER_OCT·g)。g > 0 で包絡が下がる（男性的）。
                          倍音そのものは動かない（包絡に沿って倍音の大きさが変わる）
    響き r（−1〜1）        調波の RESONANCE_BAND に Hann の形の山（中心 3 kHz、±RESONANCE_DB dB）
枝ごとの音量（branch_mix）:
    ブレシネス b（−1〜1）  有声の所だけ。雑音を 10^(9b/20) 倍にし、調波を全体のエネルギーが保たれるように合わせる
    有声度 v（0〜1）       調波 × v（線形。先例: hifisampler の Hv）
    子音の強さ c（−1〜1）  子音の音素のフレームの両方の枝 × 10^(12c/20)
波形に掛けるもの（finish）:
    うなり（0〜1）         hifisampler の growl（Apache-2.0）をそのまま使う
    音量（dB）             最後に掛け、軽いリミッタで音割れを防ぐ
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

HOP = 256
SR = 44100

# パラメータの既定値（何もしない値）
NEUTRAL = {"tension": 0.0, "breathiness": 0.0, "gender": 0.0, "voicing": 1.0,
           "loudness": 0.0, "consonant": 0.0, "resonance": 0.0, "growl": 0.0,
           "articulation": 1.0}   # 滑舌: 子音の凹みの既定の深さの倍率（ボコーダの後ではなく音響モデルの前で効く。pipeline.py）
# 周波数ごとの利得で掛けるもの
SPECTRAL_KEYS = ("tension", "gender", "resonance")
CONSONANT_DB = 12.0
TENSION_BETA = 0.6      # テンション ±1 で ±3.6 dB/オクターブ
TENSION_PIVOT = 1000.0  # 傾きの支点 [Hz]
TENSION_FMIN = 100.0    # これより低い周波数は支点からの傾きを延ばさない（低域を持ち上げすぎない）
GENDER_OCT = 0.3        # ジェンダー ±1 で 4 kHz までの包絡を 2^±0.3 倍（約 ±23 %）
GENDER_SPLIT = 4000.0   # ここまでは一様に伸び縮み、ここから 22.05 kHz で元に戻す
RESONANCE_DB = 10.0
RESONANCE_BAND = (2000.0, 4000.0)
N_FFT = 1024            # ボコーダのフィルタと同じ（包絡の bin と揃う）


def frame_param(params: dict, key: str, T: int) -> np.ndarray:
    """フレームごとの値 [T]（無ければ既定値）。"""
    a = params.get(key)
    if a is None:
        return np.full(T, NEUTRAL[key], np.float32)
    a = np.asarray(a, np.float32)[:T]
    return np.pad(a, (0, T - len(a)), mode="edge") if len(a) < T else a


def is_neutral(params: dict, keys) -> bool:
    return all(params.get(k) is None or np.allclose(params[k], NEUTRAL[k]) for k in keys)


def _to_samples(g: np.ndarray, n: int) -> np.ndarray:
    """フレームの値 → サンプルの値（フレームの中心を結ぶ線形補間。3 フレームの移動平均で段差を消す）。"""
    if len(g) > 2:
        g = np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    centers = (np.arange(len(g)) + 0.5) * HOP
    return np.interp(np.arange(n), centers, g).astype(np.float32)


def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, np.float64) / 700.0)


_NH = N_FFT // 2 + 1
_F = np.arange(_NH) * SR / N_FFT
_LO, _HI = RESONANCE_BAND
_BUMP = np.where((_F > _LO) & (_F < _HI), 0.5 - 0.5 * np.cos(2 * np.pi * (_F - _LO) / (_HI - _LO)), 0.0) \
    * RESONANCE_DB * np.log(10) / 20                                     # 自然対数の振幅
_LNF = np.log(np.maximum(_F, TENSION_FMIN) / TENSION_PIVOT)


def _gender_src(alpha: np.ndarray) -> np.ndarray:
    """フレームごとの α [n] → 出力の各 bin が読みに行く元の bin（小数）[n, bin]。"""
    a = alpha[:, None]
    m, m4, mn = _mel(_F)[None], _mel(GENDER_SPLIT), _mel(SR / 2)
    m_a4 = _mel(a * GENDER_SPLIT)
    hz_up = 700.0 * (10.0 ** ((m_a4 + (m - m4) * (mn - m_a4) / (mn - m4)) / 2595.0) - 1.0)
    src = np.where(_F[None] <= GENDER_SPLIT, _F[None] * a, hz_up) * (N_FFT / SR)
    return np.clip(src, 0.0, _NH - 1.0)


def _warp(env: np.ndarray, src: np.ndarray) -> np.ndarray:
    i0 = np.minimum(np.floor(src).astype(np.int64), _NH - 2)
    fr = src - i0
    a = np.take_along_axis(env, i0, 1)
    return a + (np.take_along_axis(env, i0 + 1, 1) - a) * fr


def _at_stft(x: np.ndarray, n_st: int) -> np.ndarray:
    """ボコーダのフレーム（中心 (t+0.5)·hop）の値 → STFT のフレーム（中心 i·hop）の値（線形補間）。"""
    pos = np.clip(np.arange(n_st) - 0.5, 0.0, len(x) - 1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (pos - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def spectral(harm: np.ndarray, noise: np.ndarray, env_h, env_n, params: dict, T: int):
    """テンション・ジェンダー・響きを、調波・雑音の波形に周波数ごとの利得として掛ける。
    env_h・env_n: ボコーダのフィルタの滑らかな包絡 [T, 513]（自然対数の振幅）。"""
    if env_h is None or is_neutral(params, SPECTRAL_KEYS):
        return harm, noise
    n = len(harm)
    kw = dict(fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros", padded=True)
    _, _, H = ss.stft(harm, **kw)
    _, _, Z = ss.stft(noise, **kw)
    n_st = H.shape[1]
    t = _at_stft(frame_param(params, "tension", T), n_st)
    g = _at_stft(frame_param(params, "gender", T), n_st)
    r = _at_stft(frame_param(params, "resonance", T), n_st)
    ln_h = np.zeros((n_st, _NH))
    ln_n = np.zeros((n_st, _NH))
    if np.any(g != 0):
        src = _gender_src(2.0 ** (GENDER_OCT * g))
        eh = _at_stft(np.asarray(env_h, np.float64), n_st)
        en = _at_stft(np.asarray(env_n, np.float64), n_st)
        ln_h += _warp(eh, src) - eh
        ln_n += _warp(en, src) - en
    if np.any(r != 0):
        ln_h += r[:, None] * _BUMP[None]
    if np.any(t != 0):
        # 傾けた後の調波のエネルギーがフレームごとに変わらないよう戻す
        p = np.abs(H.T) ** 2 * np.exp(2 * ln_h)
        tilt = (TENSION_BETA * t)[:, None] * _LNF[None]
        e0 = p.sum(1)
        e1 = (p * np.exp(2 * tilt)).sum(1)
        norm = 0.5 * np.log(np.maximum(e0, 1e-20) / np.maximum(e1, 1e-20))
        ln_h += tilt + np.clip(norm, -3.0, 3.0)[:, None]
    ln_h, ln_n = np.clip(ln_h, -8.0, 8.0), np.clip(ln_n, -8.0, 8.0)
    kw.pop("padded")
    _, h2 = ss.istft(H * np.exp(ln_h.T), **{**kw, "boundary": True})
    _, z2 = ss.istft(Z * np.exp(ln_n.T), **{**kw, "boundary": True})
    fit = lambda x: np.pad(x, (0, max(0, n - len(x))))[:n].astype(np.float32)  # noqa: E731
    return fit(h2), fit(z2)


def branch_mix(harm: np.ndarray, noise: np.ndarray, uv_hard: np.ndarray, params: dict,
               cons: np.ndarray | None = None) -> np.ndarray:
    """調波と雑音を、ブレシネス・有声度・子音の強さで重みを付けて足す。
    cons: 子音の音素のフレーム [T]（bool）。無ければ無声のフレームを子音とみなす。"""
    T = len(uv_hard)
    n = min(len(harm), len(noise), T * HOP)
    h, z = harm[:n].astype(np.float64), noise[:n].astype(np.float64)
    b = frame_param(params, "breathiness", T)
    v = np.clip(frame_param(params, "voicing", T), 0.0, 1.0)
    c = frame_param(params, "consonant", T)
    # フレームごとのエネルギー
    nf = n // HOP
    eh = np.r_[np.mean(h[:nf * HOP].reshape(nf, HOP) ** 2, axis=1), np.zeros(T - nf)][:T]
    en = np.r_[np.mean(z[:nf * HOP].reshape(nf, HOP) ** 2, axis=1), np.zeros(T - nf)][:T]
    voiced = uv_hard[:T] > 0.5
    gn = np.where(voiced, 10 ** (9.0 * b / 20.0), 1.0)
    # 有声の所: 雑音を増やした分だけ調波を下げ（減らした分だけ上げ）、全体のエネルギーを保つ
    gh = np.sqrt(np.maximum(0.0, eh + en - gn ** 2 * en) / np.maximum(eh, 1e-12))
    gh = np.where(voiced, np.clip(gh, 0.0, 2.0), 1.0) * v
    # 子音の強さ: 子音の音素のフレーム（有声の子音も含む）を両方の枝ごと ±CONSONANT_DB
    cm = (~voiced if cons is None else np.asarray(cons, bool)[:T])
    gc = np.where(cm, 10 ** (CONSONANT_DB * c / 20.0), 1.0)
    gh, gn = gh * gc, gn * gc
    return (h * _to_samples(gh, n) + z * _to_samples(gn, n)).astype(np.float32)


def growl(audio: np.ndarray, strength: np.ndarray, sr: int = 44100, frequency: float = 80.0,
          freq_low: float = 400.0) -> np.ndarray:
    """うなり。hifisampler（Apache-2.0）の util/growl.py と同じ: 400 Hz より上の帯域を 80 Hz の矩形の LFO で
    ±100·strength セントだけ時間方向にゆがめ、その帯域の RMS を戻す。strength はサンプルごとの配列。"""
    strength = np.asarray(strength, np.float64)
    if np.all(strength == 0):
        return audio
    x = np.asarray(audio, np.float64)
    sos = ss.butter(4, np.clip(freq_low / (sr / 2), 0.01, 0.99), "high", output="sos")
    band = np.asarray(ss.sosfilt(sos, x))
    low = x - band
    lfo = ss.square(2 * np.pi * frequency * np.arange(len(x)) / sr)
    ratio = 2 ** (lfo * strength * 100.0 / 1200.0)
    drift = (np.cumsum(ratio) - ratio[0]) - np.arange(len(band)) * np.mean(ratio)
    if len(band) > 100:
        drift = ss.sosfilt(ss.butter(2, 20.0 / (sr / 2), "high", output="sos"), drift)
    n = np.arange(len(band))
    mod = np.interp(np.clip(n + drift, 0, len(band) - 1), n, band)
    r0, r1 = np.sqrt(np.mean(band ** 2)), np.sqrt(np.mean(mod ** 2))
    if r1 > 1e-10:
        mod = mod * (r0 / r1)
    return (low + mod).astype(np.float32)


def finish(wav: np.ndarray, params: dict, T: int) -> np.ndarray:
    """うなりと音量を掛け、軽いリミッタを通す（0.9 を超える所だけを tanh で丸め、1.0 を超えない）。"""
    n = len(wav)
    g = frame_param(params, "growl", T)
    if np.any(g != 0):
        wav = growl(wav, _to_samples(np.clip(g, 0.0, 1.0), n))
    db = frame_param(params, "loudness", T)
    if np.any(db != 0):
        wav = wav * (10 ** (_to_samples(db, n) / 20.0))
    a = np.abs(wav)
    over = a > 0.9
    if over.any():
        wav = np.where(over, np.sign(wav) * (0.9 + 0.1 * np.tanh((a - 0.9) / 0.1)), wav)
    return np.asarray(wav, np.float32)
