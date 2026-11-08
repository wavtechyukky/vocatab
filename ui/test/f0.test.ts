// F0 のホスト側（f0host.ts）の照らし合わせ: Python の _f0 と analyze の最後と、段ごとに比べる。
// モデルの出力と雑音は正解のものを使うので、ここで比べるのはホストの計算だけ。
import { f0Feeds, finishF0, frameNoteIndex, prepareF0 } from "../src/engine-web/f0host";
import type { PhonemeSequence, VoicebankWeb } from "../src/engine-web/types";
import { cases, close, done, loadCase, same, shapeOf, vbFor } from "./golden";

/** NaN の位置も含めて比べる（NaN 同士は一致とみなす） */
function closeNaN(label: string, got: ArrayLike<number>, want: ArrayLike<number>, atol = 0): void {
  const g = Array.from(got, (x) => (Number.isNaN(x) ? -12345 : x));
  const w = Array.from(want, (x) => (Number.isNaN(x) ? -12345 : x));
  close(label, g, w, atol);
}

const big = (a: BigInt64Array | Float32Array) => Array.from(a as ArrayLike<number | bigint>, (x) => Number(x));

for (const name of cases()) {
  const vb = vbFor(name) as VoicebankWeb;
  const c = loadCase(name);
  const spk = Number(c.input.singer);
  c.phrases.forEach((p: any, pi: number) => {
    const tag = `${name}[${pi}]`;
    console.log(`\n── ${tag} ──`);
    const pp = p.postprocess.out;
    // _f0 が受け取るのは、手で動かした後の長さ（_apply_shifts がその場で seq.dur_sec を書き換える）
    same(`${tag} 動かした後の音素の並びが後処理と同じ`, p.apply_shifts.phonemes, pp.phonemes);
    const seq: PhonemeSequence = {
      phonemes: pp.phonemes, dur_sec: p.apply_shifts.dur_after as Float32Array, note_midi: pp.note_midi,
      is_vowel: pp.is_vowel, note_event_starts: pp.note_event_starts, leading_pau_sec: pp.leading_pau_sec,
      held: pp.held ?? new Uint8Array(pp.phonemes.length), // 伸ばしの母音（伸ばしの記号で学習した F0 モデルにはその記号で渡す）
    };
    const prep = prepareF0(vb, seq, spk);
    const fd = p.f0_run.feeds;
    const T = shapeOf(fd.noise)[3];
    same(`${tag} T`, prep.T, T);
    same(`${tag} durations（ph_durs）`, Array.from(prep.phDurs), Array.from(fd.durations));
    const feeds = f0Feeds(vb, prep, spk, fd.noise as Float32Array);
    same(`${tag} tokens`, big(feeds.tokens.data), Array.from(fd.tokens));
    same(`${tag} durations の feed`, big(feeds.durations.data), Array.from(fd.durations));
    same(`${tag} spk_id`, big(feeds.spk_id.data), Array.from(fd.spk_id));
    close(`${tag} pitch_midi`, feeds.pitch_midi.data as Float32Array, fd.pitch_midi, 0);
    close(`${tag} style_mix`, feeds.style_mix.data as Float32Array, fd.style_mix, 0);
    same(`${tag} 形`, Object.fromEntries(Object.entries(feeds).map(([k, v]) => [k, v.dims])),
         Object.fromEntries(Object.keys(feeds).map((k) => [k, shapeOf(fd[k])])));
    same(`${tag} 雑音はそのまま`, feeds.noise.data === fd.noise, true);

    const o = p.f0_run.outputs;
    const r = finishF0(vb, prep, { plain: o.plain, vibrato: o.vibrato, vuv: o.vuv });
    const pf = p.per_frame;
    same(`${tag} build_per_frame_arrays の n`, r.T, pf.args.n);
    closeNaN(`${tag} build_per_frame_arrays の values`, prep.values, pf.args.values);
    same(`${tag} phonemes_pf`, r.phonemesPf, pf.out[0]);
    close(`${tag} note_pitches_pf`, r.notePitchesPf, pf.out[1], 0);
    same(`${tag} note_boundaries_pf`, Array.from(r.noteBoundariesPf), Array.from(pf.out[2]));

    const ci = p.correct_f0.args.in;
    close(`${tag} baseline_log2`, r.baseLog2, ci.baseline_log2, 1e-6);
    close(`${tag} plain_delta`, r.plainDelta, ci.plain_delta, 0);
    close(`${tag} vib_delta`, r.vibLog2, ci.vib_delta, 0);
    close(`${tag} uv`, r.uv, ci.uv, 0);
    same(`${tag} frame_is_pau`, Array.from(r.frameIsPau), Array.from(ci.frame_is_pau));
    close(`${tag} plain_hz（補正の前、相対）`, r.plainHzBeforeCorrection, ci.plain_hz, 0, 1e-6);
    const co = p.correct_f0.out;
    close(`${tag} plain_hz（補正の後、相対）`, r.plainHz, co.plain_hz, 0, 1e-6);
    close(`${tag} f0_hz（補正の後、相対）`, r.f0Hz, co.f0_hz, 0, 1e-6);

    const res = p.result;
    close(`${tag} plain_log2`, r.plainLog2, res.plain_log2, 1e-5);
    close(`${tag} vib_log2`, r.vibLog2, res.vib_log2, 0);
    close(`${tag} uv（結果）`, r.uv, res.uv, 0);
    close(`${tag} base_log2`, r.baseLog2, res.base_log2, 1e-6);

    // AI リテイクのフレームごとの音符の番号: np.repeat(note_index, ph_durs)[:T] を素直に作ったものと比べる
    const ni: number[] = res.note_index;
    const want: number[] = [];
    ni.forEach((k, j) => { for (let f = 0; f < prep.phDurs[j]; f++) want.push(k); });
    same(`${tag} frameNoteIndex`, Array.from(frameNoteIndex(ni, prep.phDurs, prep.T)), want.slice(0, prep.T));
  });
}
done();
