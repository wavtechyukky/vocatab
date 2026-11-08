// 合成のホスト側（synthhost・voiceparams・melspec・x0shape）を Python の正解と照らし合わせる。
// モデルの出力（音響モデルの mel・ボコーダの出力）と雑音は正解のものを入れ、ホスト側の計算だけを比べる。
import { acousticFeeds, finishSynth, melPostprocess, nhvFeeds, nsfFeeds, phraseDips, roundFloat16,
         x0Curves } from "../src/engine-web/synthhost";
import * as VP from "../src/engine-web/voiceparams";
import * as X0 from "../src/engine-web/x0shape";
import { wavToMel } from "../src/engine-web/melspec";
import type { Feed, VoicebankWeb } from "../src/engine-web/types";
import { cases, close, done, loadCase, same, vbFor } from "./golden";

const num = (f: Feed): number[] => Array.from(f.data as ArrayLike<number | bigint>, (x) => Number(x));
const ms = (t0: number): string => `${(performance.now() - t0).toFixed(0)} ms`;

// scipy.signal.butter の sos（Python で書き出した値）と、TS で作った値の差
{
  const want4 = [[0.9282322751998473, -1.8564645503996946, 0.9282322751998473, 1.0, -1.896929503200547, 0.9000141954472615],
    [1.0, -2.0, 1.0, 1.0, -1.9541573210865661, 0.9573350743594449]];
  const want2 = [[0.997987115675119, -1.995974231350238, 0.997987115675119, 1.0, -1.9959701796428286, 0.9959782830576475]];
  close("butter(4, 400 Hz, high) の sos", VP.butterHighSos(4, 400 / 22050).flat(), want4.flat(), 1e-15);
  close("butter(2, 20 Hz, high) の sos", VP.butterHighSos(2, 20 / 22050).flat(), want2.flat(), 1e-15);
}

// 正解の楽譜では 0.9 を超えないので、リミッタ（と音量）は小さな作り物の入力で numpy の値と比べる
{
  const linspace = (a: number, b: number, n: number): Float32Array =>
    Float32Array.from({ length: n }, (_, i) => (i === n - 1 ? b : i * ((b - a) / (n - 1)) + a));
  const want1 = [-0.9999987483024597, -0.9999908804893494, -0.9999328851699829, -0.9995054602622986, -0.9964027404785156,
    -0.976159393787384, -0.8999999761581421, -0.800000011920929, -0.699999988079071, -0.6000000238418579, -0.5,
    -0.4000000059604645, -0.30000001192092896, -0.20000000298023224, -0.10000000149011612, 0.0, 0.10000000149011612,
    0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071,
    0.800000011920929, 0.8999999761581421, 0.976159393787384, 0.9964027404785156, 0.9995054602622986, 0.9999328851699829,
    0.9999908804893494, 0.9999987483024597];
  close("finish のリミッタ（作り物）", VP.finish(linspace(-1.5, 1.5, 31), {}, 1), want1, 0);
  const want2 = [-0.8001129031181335, -0.684245228767395, -0.568377673625946, -0.4525100290775299, -0.341215044260025,
    -0.2294291853904724, -0.11177487671375275, 0.011973119340837002, 0.1420476734638214, 0.27868953347206116,
    0.4221475422382355, 0.5638622045516968, 0.7015711069107056, 0.8392800688743591];
  const w2 = VP.finish(linspace(-0.6, 0.6, 512), { loudness: Float32Array.of(2.5, 4.0) }, 2);
  close("finish の音量（作り物、2 フレーム）", w2.filter((_, i) => i % 37 === 0), want2, 0);
}

for (const name of cases()) {
  const vb = vbFor(name) as VoicebankWeb;
  const c = loadCase(name);
  c.phrases.forEach((ph: any, pi: number) => {
    const r = ph.result;
    // 子音ごとの既定の凹み（解析で作る）
    const dips = phraseDips(vb, r.phonemes, r.ph_durs);
    same(`${name} 句${pi} 既定の凹み（dips）`, dips, r.dips ?? null);
    for (const s of ph.synth) {
      const tag = `${name} 句${pi} ${s.kind}`;
      console.log(`\n── ${tag}`);
      const T = Array.from(r.ph_durs as Int32Array).reduce((a, b) => a + b, 0);
      const gf = s.ac_run.feeds;
      // 子音の凹み: 手で動かした凹み（句の中の音符の番号ごと）と滑舌の倍率 → x0_loggain
      const ov = X0.dipOverrides(dips, r.note_index, s.ph_dips ?? {});
      const sc = X0.dipScales(dips, r.ph_durs, s.params.articulation ?? Float32Array.of(VP.NEUTRAL.articulation), ov);
      const x0 = x0Curves(vb, { phonemes: r.phonemes, phDurs: r.ph_durs, dips }, ov, sc, T, "x0_loggain" in gf);
      // ⑤ 音響モデルへの入力
      // 伸ばしの母音は、acoustic_merge_melisma のボイスバンクなら前の音素とまとめて渡す（tokens・durations が短くなる）
      const ac = acousticFeeds(vb, { phonemes: r.phonemes, phDurs: r.ph_durs, uv: r.uv, noteIndex: r.note_index, held: r.held },
        s.f0_hz, c.input.singer, gf.exc_noise ?? null, x0.feed);
      same(`${tag} tokens`, num(ac.feeds.tokens), Array.from(gf.tokens));
      same(`${tag} durations`, num(ac.feeds.durations), Array.from(gf.durations));
      const nUnits = gf.tokens.length;
      if (nUnits !== r.phonemes.length) console.log(`    まとめた伸ばしの母音: ${r.phonemes.length - nUnits} 個`);
      same(`${tag} 入力の形`, Object.fromEntries(Object.entries(ac.feeds).map(([k, f]) => [k, f.dims])),
        { tokens: [1, nUnits], durations: [1, nUnits], f0: [1, T], spk_embed: [1, 256],
          ...(gf.exc_noise ? { exc_noise: [1, T * 256] } : {}), ...(gf.x0_loggain ? { x0_loggain: [1, T] } : {}) });
      if (gf.x0_loggain) {
        const g = ac.feeds.x0_loggain.data as Float32Array;
        close(`${tag} x0_loggain（凹み ${g.filter((x) => x < 0).length} フレーム）`, g, gf.x0_loggain, 0);
      }
      close(`${tag} f0`, ac.feeds.f0.data as Float32Array, gf.f0, 0);
      close(`${tag} spk_embed`, ac.feeds.spk_embed.data as Float32Array, gf.spk_embed, 0);
      close(`${tag} uv_hard`, ac.uvHard, s.result.uv_hard, 0);
      if (s.vp_branch) close(`${tag} cons（子音のフレーム）`, ac.cons, s.vp_branch.args.cons, 0);

      // ⑥ mel の後処理
      let t0 = performance.now();
      const melIn = melPostprocess(s.ac_run.outputs.mel, T, ac.uvHard, c.frame_rate);
      console.log(`    mel の後処理 ${ms(t0)}`);
      close(`${tag} mel_postprocess → mel_in`, melIn, s.result.mel_in, 1e-5);

      // ⑦ ボコーダへの入力（正解の mel_in から）
      const vf = s.voc_run.feeds;
      if (s.kind === "nhvsing") {
        const f = nhvFeeds(s.result.mel_in, T, ac.f0, ac.uvHard, vf.noise);
        close(`${tag} voc mel`, f.mel.data as Float32Array, vf.mel, 0);
        close(`${tag} voc f0`, f.f0.data as Float32Array, vf.f0, 0);
        close(`${tag} voc uv`, f.uv.data as Float32Array, vf.uv, 0);
        same(`${tag} voc の形`, [f.mel.dims, f.f0.dims, f.uv.dims, f.noise.dims], [[1, T, 128], [1, 1, T], [1, 1, T], [1, 1, T, 256]]);
      } else {
        const f = nsfFeeds(s.result.mel_in, T, ac.f0);
        close(`${tag} voc mel（hop 512）`, f.mel.data as Float32Array, vf.mel, 0);
        close(`${tag} voc f0（hop 512、相対）`, f.f0.data as Float32Array, vf.f0, 0, 1e-6);
        same(`${tag} voc の形`, [f.mel.dims, f.f0.dims], [[1, Math.ceil(T / 2), 128], [1, Math.ceil(T / 2)]]);
      }

      // ボイスパラメータの段ごと（正解の入力から）
      const params = s.params as Record<string, Float32Array>;
      if (s.vp_spectral) {
        const a = s.vp_spectral.args;
        close(`${tag} 包絡の float16 の丸め（env_h）`, Float32Array.from(s.voc_run.outputs.out3, roundFloat16), a.env_h, 0);
        close(`${tag} 包絡の float16 の丸め（env_n）`, Float32Array.from(s.voc_run.outputs.out4, roundFloat16), a.env_n, 0);
        t0 = performance.now();
        const [h, z] = VP.spectral(a.harm, a.noise, a.env_h, a.env_n, a.params, a.T);
        console.log(`    spectral ${ms(t0)}`);
        close(`${tag} spectral 調波`, h, s.vp_spectral.out[0], 1e-5);
        close(`${tag} spectral 雑音`, z, s.vp_spectral.out[1], 1e-5);
      }
      if (s.vp_branch) {
        const a = s.vp_branch.args;
        t0 = performance.now();
        const w = VP.branchMix(a.h, a.z, a.uv_hard, a.params, a.cons);
        console.log(`    branch_mix ${ms(t0)}`);
        close(`${tag} branch_mix`, w, s.vp_branch.out, 1e-5);
      }
      {
        const a = s.vp_finish.args;
        t0 = performance.now();
        const w = VP.finish(a.wav, a.params, a.T);
        console.log(`    finish ${ms(t0)}`);
        close(`${tag} finish`, w, s.vp_finish.out, 1e-5);
      }

      // ボコーダの後をまとめて（ボコーダの出力の名前は ONNX の出力の名前: out0..4 = waveform・harmonic・noise_out・env_h・env_n）
      const vo = s.voc_run.outputs;
      const out = s.kind === "nhvsing"
        ? { waveform: vo.out0, harmonic: vo.out1, noise_out: vo.out2, env_h: vo.out3, env_n: vo.out4 }
        : { waveform: vo.waveform };
      t0 = performance.now();
      const fin = finishSynth({ kind: s.kind, voice_params: s.voice_params, gain_db: s.gain_db }, T, out, params,
        ac.uvHard, ac.cons, true);
      console.log(`    finishSynth ${ms(t0)}`);
      close(`${tag} wav`, fin.wav, s.result.wav, 1e-5);
      close(`${tag} wav_plain`, fin.wavPlain ?? [], s.result.wav_plain, 1e-5);

      // 出た音の mel（正解の wav から、と TS の wav から）
      t0 = performance.now();
      const m = wavToMel(s.result.wav);
      console.log(`    wav_to_mel ${ms(t0)}`);
      same(`${tag} mel_out のフレーム数`, m.frames, T);
      close(`${tag} mel_out（正解の wav から、ln）`, m.mel, s.mel_out, 1e-4);
      close(`${tag} mel_out（TS の wav から、ln）`, wavToMel(fin.wav).mel, s.mel_out, 1e-4);
    }
  });
}
done();
