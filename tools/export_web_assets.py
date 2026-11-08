"""ブラウザ版のエンジンが読む設定を JSON に書き出す（Python の PrettyPitch・yaml を読まずに済むように）。

    web_index.json                      ボイスバンクとボコーダの一覧（/api/info の代わり）とファイルの場所
    voicebanks/<名前>/web.json          音素の表・モーラ表・楽譜の記号・子音の長さの下限（歌い手ごとに引いた値）・
                                        F0 と子音の長さのモデルの設定・話者ベクトル

パスはすべてリポジトリの頭（ページの 1 つ上）から見た相対。ボイスバンクやボコーダを足したら作り直す。
使い方: python tools/export_web_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.voicebank import scan  # noqa: E402


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def phoneset_json(ps, mora: dict) -> dict:
    return {"lang": ps.lang, "phonemes": list(ps.phonemes), "nuclei": sorted(ps.nuclei), "vowels": sorted(ps.vowels),
            "vowel_normalize": dict(ps.vowel_normalize), "silence": sorted(ps.silence), "consonants": list(ps.consonants),
            "sil_map": dict(ps.sil_map), "case_fold": ps.case_fold, "strip_stress_digits": ps.strip_stress_digits,
            # 韓国語: 句末の終声も子音の長さを予測する・規則の G2P（ui/src/engine-web/g2p_ko.ts）
            "final_consonants": bool(getattr(ps, "final_consonants", False)), "g2p": getattr(ps, "g2p", None),
            "score": dict(ps.score), "mora": mora}


def profile_json(profile, ps, singers: list[int]) -> dict:
    """ConsDurProfile を、使う引き方（v_min・c_min・lead_c_min・weight）ごとに歌い手別の表へ展開する。
    鍵 "*" は歌い手を指定しないとき（spk_id=None）。style_id は 0 だけ（エディタはスタイルを使わない）。"""
    out = {"enabled": profile.enabled}
    for spk in [None] + singers:
        st = None if spk is None else 0
        out["*" if spk is None else str(spk)] = {
            "v_min": profile.v_min(spk, st),
            "c_min": {p: profile.c_min(p, spk, st) for p in ps.phonemes},
            "lead_c_min": {p: profile.lead_c_min(p, spk, st) for p in ps.phonemes},
            "weight": {p: profile.weight(p, spk, st) for p in ps.phonemes},
        }
    return out


def main():
    vbs, vocs = scan(ROOT)
    index = {"voicebanks": [], "vocoders": []}
    for key, vb in vbs.items():
        if not vb.publish:    # voicebank.yaml の publish: false（公開しないもの。Python のエンジンでは使える）
            print(f"{key}: publish: false なので web_index.json には載せない")
            continue
        pp_root = str(vb.code["prettypitch"])
        if pp_root not in sys.path:
            sys.path.insert(0, pp_root)
        from duration_model import phoneset as ps_mod
        from duration_model.profile import from_dict
        from svs import score_io
        ps = ps_mod.load(vb.language)
        mora = score_io._as_mora_table(None, ps)  # エンジン（score_to_phrases）と同じ読み方。モーラ表の無い言語は空
        m = vb.models
        profile = from_dict(json.loads(m["cons_dur_profile"].read_text(encoding="utf-8")), ps)
        f0cfg = json.loads(m["f0"].with_suffix(".json").read_text(encoding="utf-8"))
        cons_cfg = json.loads(m["cons_dur"].with_suffix(".json").read_text(encoding="utf-8"))
        spk = json.loads(m["speakers"].read_text(encoding="utf-8"))
        grid = vb.raw.get("grid") or {"sample_rate": 44100, "hop": 256}
        web = {"key": key, "name": vb.name, "language": vb.language, "grid": grid,
               "frame_rate": float(f0cfg["frame_rate"]),
               "models": {"acoustic": rel(m["acoustic"]), "f0": rel(m["f0"]), "cons_dur": rel(m["cons_dur"])},
               "phoneset": phoneset_json(ps, mora), "profile": profile_json(profile, ps, [s.id for s in vb.singers]),
               "f0": f0cfg, "cons_dur": cons_cfg, "speakers": spk}
        if vb.breaths and vb.breaths.exists():
            web["breaths"] = rel(vb.breaths)   # ブレスの mel の一覧（breaths.json。bin はその中に書いてある）
        if isinstance(vb.raw.get("x0_shape"), dict):
            web["x0_shape"] = vb.raw["x0_shape"]   # 子音の凹みのレシピ（ブラウザ版は engine-web/x0shape.ts が読む）
        # 音響モデルの入力の名前（ブラウザ版が、モデルを読む前に exc_noise・x0_loggain の有無を知るため）
        import onnxruntime as ort
        web["acoustic_inputs"] = [i.name for i in ort.InferenceSession(
            str(m["acoustic"]), providers=["CPUExecutionProvider"]).get_inputs()]
        path = vb.dir / "web.json"
        path.write_text(json.dumps(web, ensure_ascii=False), encoding="utf-8")
        index["voicebanks"].append({**vb.to_json(), "web": rel(path)})
        print(f"{path}: {path.stat().st_size / 1e3:.0f} KB")
    for key, v in vocs.items():
        if not v.publish:     # vocoder.yaml の publish: false（配布しないもの。Python のエンジンでは使える）
            print(f"{key}: publish: false なので web_index.json には載せない")
            continue
        y = yaml.safe_load((v.dir / "vocoder.yaml").read_text(encoding="utf-8"))
        index["vocoders"].append({**v.to_json(), "gain_db": v.gain_db, "onnx": rel(v.onnx),
                                  "bytes": v.onnx.stat().st_size, "grid": y.get("grid") or {}})
    for vb in index["voicebanks"]:
        d = vbs[vb["key"]]
        vb["bytes"] = {k: d.models[k].stat().st_size for k in ("acoustic", "f0", "cons_dur")}
    (ROOT / "web_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{ROOT / 'web_index.json'}")


if __name__ == "__main__":
    main()
