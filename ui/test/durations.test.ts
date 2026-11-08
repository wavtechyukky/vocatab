// 楽譜 → 句 → 音素の長さ（score.ts・durations.ts）を、Python のエンジンの正解（tools/golden/dump_golden.py）と照らし合わせる。
// 子音の長さのモデルの出力は正解のもの（cons_run.outputs.dur_sec）を使い、ホスト側の処理だけを確かめる。
import { alignDurs, finishDurations, prepareDurations } from "../src/engine-web/durations";
import { phrasesFromNotes } from "../src/engine-web/score";
import type { VoicebankWeb } from "../src/engine-web/types";
import { cases, close, done, hasVb, loadCase, readRepo, same, vbFor } from "./golden";

const vb = JSON.parse(readRepo("voicebanks/leapsinger_ja_3singer/web.json")) as VoicebankWeb;
const vbKo = hasVb("korean") ? (vbFor("korean") as VoicebankWeb) : null;

const flat = (xs: number[][]): number[] => xs.flatMap((x) => x);
const lens = (xs: number[][]): number[] => xs.map((x) => x.length);
const num = (a: ArrayLike<number | bigint>): number[] => Array.from(a as ArrayLike<number>, (x) => Number(x));

for (const name of cases()) {
  const vb = vbFor(name) as VoicebankWeb;
  const c = loadCase(name);
  const inp = c.input;
  const got = phrasesFromNotes(vb, inp.notes, inp.bpm);
  same(`${name}: 句の数`, got.length, c.phrases.length);
  c.phrases.forEach((g: any, pi: number) => {
    const tag = `${name} 句 ${pi}`;
    const ours = got[pi];
    if (!ours) return;
    same(`${tag}: phrases_from_notes の句`, ours.phrase, g.phrase);
    same(`${tag}: 音符の id`, ours.ids, g.ids);

    // ① の前半: 音節に分け、モデルへの入力を作る（正解の句から始めて、前の段の誤差を持ち込まない）
    const prep = prepareDurations(vb, g.phrase, inp.singer);
    same(`${tag}: split_phrase_syllables`, prep.ph2, g.split.out);
    if (g.cons_run) {
      const wf = g.cons_run.feeds;
      if (!prep.feeds) {
        same(`${tag}: cons_dur の入力がある`, false, true);
      } else {
        same(`${tag}: cons_dur の入力の名前と順`, Object.keys(prep.feeds), Object.keys(wf));
        for (const k of Object.keys(wf)) {
          const f = prep.feeds[k];
          if (!f) continue;
          const isInt = wf[k] instanceof Int32Array;
          same(`${tag}: feed ${k} の型・形`, [f.type, f.dims], [isInt ? "int64" : "float32", [wf[k].length]]);
          if (isInt) same(`${tag}: feed ${k}`, num(f.data), num(wf[k]));
          else close(`${tag}: feed ${k}`, f.data as Float32Array, wf[k]);
        }
      }
    } else {
      same(`${tag}: cons_dur の入力が無い`, prep.feeds, null);
    }

    // ① の後半〜②
    const pred: Float32Array | null = g.cons_run ? g.cons_run.outputs.dur_sec : null;
    const r = finishDurations(vb, prep, pred, inp.singer, g.ph_shifts ?? {});
    const pa = g.postprocess.args;
    for (const [k, w] of [["onset", pa.onset_nat], ["coda", pa.coda_nat], ["core", pa.core_nat]] as const) {
      const mine = r.naturals[k as "onset" | "coda" | "core"];
      same(`${tag}: 自然長 ${k} の形`, lens(mine), lens(w));
      close(`${tag}: 自然長 ${k}`, flat(mine), flat(w));
    }
    const po = g.postprocess.out;
    same(`${tag}: 後処理の音素`, r.seq.phonemes, po.phonemes);
    close(`${tag}: 後処理の dur_sec（動かす前）`, r.seq0DurSec, po.dur_sec);
    same(`${tag}: note_midi`, num(r.seq.note_midi), num(po.note_midi));
    same(`${tag}: is_vowel`, num(r.seq.is_vowel), num(po.is_vowel));
    same(`${tag}: note_event_starts`, num(r.seq.note_event_starts), num(po.note_event_starts));
    close(`${tag}: leading_pau_sec`, [r.seq.leading_pau_sec], [po.leading_pau_sec]);
    close(`${tag}: phrase_start_sec`, [r.startSec, r.onset0Sec], g.start_sec.out, 1e-9);

    const as = g.apply_shifts;
    same(`${tag}: note_index`, r.noteIndex, as.note_index);
    close(`${tag}: _apply_shifts の当てたずれ`, r.applied, as.applied, 1e-9);
    close(`${tag}: _apply_shifts 後の dur_sec`, r.seq.dur_sec, as.dur_after);

    const [wd, wT] = g.to_frames.out;
    same(`${tag}: to_frames のフレーム数`, num(r.phDurs), num(wd));
    same(`${tag}: to_frames の総フレーム数`, r.totalFrames, wT);
    const al = alignDurs(r.phDurs, g.align.args.T);
    same(`${tag}: _align_durs`, num(al), num(g.align.out));
    same(`${tag}: 結果の ph_durs`, num(al), num(g.result.ph_durs));
    same(`${tag}: 結果の note_index`, r.noteIndex, g.result.note_index);
    close(`${tag}: 結果の shifts`, r.applied, g.result.shifts, 1e-9);
    close(`${tag}: 結果の start_sec`, [r.startSec], [g.result.start_sec], 1e-9);
  });
}

// 歌詞が読めないときのエラーの文面（Python と同じ形）
const err = (lyric: string): string => {
  try {
    phrasesFromNotes(vb, [{ id: 1, onset: 0, dur: 480, pitch: 60, lyric }], 120);
    return "";
  } catch (e) {
    return (e as Error).message;
  }
};
same("エラー: 音素として読めない英字", err("kaxq"), "歌詞「kaxq」の「xq」を音素として読めません");
same("エラー: モーラ表に無い文字", err("あ漢"), "歌詞 'あ漢' の '漢' 付近がモーラ表にありません（phonemes を明示するか表に追加してください）");

// 「か」の後の 2 つ目の音符の歌詞 → 音素（Python の phrases_from_notes で確かめた値）
const second = (lyric: string): string[] =>
  phrasesFromNotes(vb, [{ id: 1, onset: 0, dur: 480, pitch: 60, lyric: "か" }, { id: 2, onset: 480, dur: 480, pitch: 60, lyric }], 120)[0]
    .phrase.notes[1].phonemes;
same("歌詞: 半角の伸ばし棒「ｰ」（メリスマ記号の表から）", second("ｰ"), ["a"]);
same("歌詞: 途中の伸ばし棒「あーい」", second("あーい"), ["a", "a", "i"]);
same("歌詞: 区切らないローマ字「shu」", second("shu"), ["sh", "u"]);
same("歌詞: 空白で区切った音素「a b」", second("a b"), ["a", "b"]);

// 韓国語の楽譜への G2P の当て方（PrettyPitch の tests/test_g2p_ko.py の score_to_phrases の試験と同じ）。
// 韓国語のボイスバンクが手元にあるときだけ
if (vbKo) {
  const ns = [[0, 480, 60, "음"], [480, 480, 62, "악"], [1440, 480, 64, "밤"], [1920, 480, 65, "-"], [2400, 480, 67, "-"],
    [3120, 480, 60, "사랑"]] as const;
  const got = phrasesFromNotes(vbKo, ns.map(([o, d, p, l], i) => ({ id: i + 1, onset: o, dur: d, pitch: p, lyric: l })), 120)
    .flatMap((x) => x.phrase.notes.map((n) => n.phonemes));
  same("韓国語: 連音は休符をまたがない・終声は伸ばしの最後へ・1 音符に 2 音節", got,
       [["eu"], ["m", "a", "K"], ["b", "a"], ["a"], ["a", "M"], ["s", "a", "r", "a", "NG"]]);
}
done();
