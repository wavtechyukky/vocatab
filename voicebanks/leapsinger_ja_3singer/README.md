# leapsinger_ja_3singer — 日本語 3 人のボイスバンク

| 歌い手 | id | 元の歌声データベース |
|---|---|---|
| oniku（御丹宮くるみ） | 0 | 御丹宮くるみ歌声データベース — https://onikuru.info/db-download/ |
| natsume（夏目悠李） | 1 | 夏目悠李 歌声データベース（歌声DB制作: アマノケイ／音声提供者: 霧野蒼太） — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs |
| ritsu（波音リツ） | 2 | 「波音リツ」歌声データベース（音源収録: カノン） — https://www.canon-voice.com/voicebanks/ |

## 中身

| ファイル | 何か |
|---|---|
| `acoustic.onnx`, `speakers.json` | 音響モデル LeapSinger v2（1 step、励起の倍音は 1 つの ONNX Loop）と話者ベクトル |
| `f0.onnx`, `f0.json` | F0 モデル PrettyPitch（拡散 10 step） |
| `cons_dur.onnx`, `cons_dur.json`, `cons_dur_profile.json` | 子音の長さのモデルと、歌い手ごとの長さの下限 |
| `breaths.json`, `breaths.bin` | ブレス（息継ぎ）の素材: 歌い手ごとのブレスの ln-mel（float16）。録音そのものではなく、ボコーダで鳴らして使う |
| `web.json` | ブラウザ版のエンジンが読む設定（`tools/export_web_assets.py` が `voicebank.yaml` から作る） |
| `夏目悠李_利用規約.txt` | 夏目悠李の出力音声に関する利用規約（同梱が条件） |

## 音響モデルの書き出し方

`acoustic.onnx` と `speakers.json` は、LeapSinger v2.0.0 の重み（Release の `3singer_v2_student.pt`）を、LeapSinger の書き出しで作ったものです（LeapSinger のフォルダで）:

    python -m export.cli \
      --ckpt 3singer_v2_student.pt \
      --out export/acoustic --model-name acoustic \
      --variant diffsinger --hop 256 --speaker embed --exc-noise --x0-gain --verify

書き出された ONNX と `acoustic.speakers.json` を、`acoustic.onnx`・`speakers.json` としてこのフォルダに置きます。

- `--exc-noise`: 励起の雑音を入力 `exc_noise` にします。エンジンが句ごとに種を決めて渡すので、同じ句は毎回同じ音になり、AI リテイクでは一部の区間だけ歌い直せます。
- `--x0-gain`: 子音の凹みの入力 `x0_loggain` を足します（0 なら凹みの無い音と同じ）。凹みの形は `voicebank.yaml` の `x0_shape` で決めます。

## 利用条件

このフォルダのモデルと素材は、上の 3 つの歌声データベースから作った派生物で、各データベースの利用規約に従います。

- **個人・非商用**での利用を前提としています。商用利用や法人での利用は、各データベースの規約に従って事前の許諾・問い合わせが必要な場合があります。
- モデル・素材の**再配布はしないでください**（このフォルダの中身を単体や別の製品に入れて配ること）。
- 出力した音を公開するときは、使った歌い手のクレジットを表示してください。夏目悠李は「歌声DB制作: アマノケイ／音声提供者: 霧野蒼太」を表示し、`夏目悠李_利用規約.txt` の内容に従ってください。波音リツは「波音リツ」と「カノン」の両方を表示してください。
- 正確な条件は各データベースの公式の規約が優先します。この README は概要です。

ブレスの素材について: 各データベースの録音から、ラベルの `br` と休符の中のブレスを検出し、聴いて選んだものを mel にしてあります。音響モデルと同じく「データベースから作った派生物」として、同じ条件で扱います。

モデルのコード（学習・書き出し）は MIT で公開しています: [LeapSinger](https://github.com/wavtechyukky/LeapSinger)・[PrettyPitch](https://github.com/wavtechyukky/PrettyPitch)。
