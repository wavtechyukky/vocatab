// 全体の照らし合わせ（ブラウザで動かす）: ブラウザ版のエンジン（core.ts）に、Python 版と同じ楽譜と同じ雑音を入れ、
// 実際に 3 つのモデル（onnxruntime-web）を動かして、解析（F0）と合成（mel・波形）の結果を正解と比べる。
// tools/run_e2e.py がヘッドレスの Chrome で ui/test/e2e.html を開き、window.__result を受け取る。
import { Engine } from "../src/engine-web/core";

declare const ort: any;

const BASE = new URL("../../", location.href).href; // svs_editor_browser の頭
const lines: string[] = [];
let failures = 0;
const log = (s: string) => {
  lines.push(s);
  (document.getElementById("log") as HTMLElement).textContent = lines.join("\n");
};

// ── 正解の読み込み（ui/test/golden.ts のブラウザ版） ──
async function resolve(v: any, dir: string): Promise<any> {
  if (Array.isArray(v)) return Promise.all(v.map((x) => resolve(x, dir)));
  if (v && typeof v === "object") {
    if ("$dtype" in v) {
      const dt = v.$dtype as string;
      if (dt === "str") return v.$data;
      if (v.$bin) {
        const buf = await (await fetch(dir + v.$bin)).arrayBuffer();
        return dt === "float64" ? new Float64Array(buf) : dt === "int32" || dt === "int64" ? new Int32Array(buf)
          : dt === "uint8" || dt === "bool" ? new Uint8Array(buf) : new Float32Array(buf);
      }
      const d = (v.$data as Array<number | null>).map((x) => x ?? NaN);
      return dt === "float64" ? Float64Array.from(d) : dt === "int32" || dt === "int64" ? Int32Array.from(d)
        : dt === "uint8" || dt === "bool" ? Uint8Array.from(d) : Float32Array.from(d);
    }
    const o: any = {};
    for (const [k, x] of Object.entries(v)) o[k] = await resolve(x, dir);
    return o;
  }
  return v;
}

async function loadCase(name: string): Promise<any> {
  const dir = `${BASE}ui/test/golden/${name}/`;
  return resolve(await (await fetch(dir + "case.json")).json(), dir);
}

/** |差| ≤ atol をすべて満たし、平均の差が meanTol 以下なら合格（meanTol は省くと調べない） */
function close(label: string, got: ArrayLike<number>, want: ArrayLike<number>, atol: number, meanTol = Infinity): void {
  if (got.length !== want.length) {
    failures++;
    log(`NG  ${label}: 長さが違う（${got.length} と正解 ${want.length}）`);
    return;
  }
  let maxd = 0, bad = 0, sum = 0;
  for (let i = 0; i < got.length; i++) {
    const d = Math.abs(Number(got[i]) - Number(want[i]));
    sum += d;
    if (!(d <= atol)) bad++;
    if (d > maxd || Number.isNaN(d)) maxd = d;
  }
  const mean = sum / Math.max(1, got.length);
  const ng = bad > 0 || !(mean <= meanTol);
  if (ng) failures++;
  log(`${ng ? "NG" : "OK"}  ${label}: 最大の差 ${maxd.toExponential(2)}、平均 ${mean.toExponential(2)}${bad ? `、外れ ${bad}/${got.length}` : ""}`);
}

function same(label: string, got: unknown, want: unknown): void {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) failures++;
  log(`${g === w ? "OK" : "NG"}  ${label}${g === w ? "" : `\n    得た値: ${g.slice(0, 300)}\n    正解:   ${w.slice(0, 300)}`}`);
}

async function runCase(name: string): Promise<void> {
  log(`── ${name}`);
  const c = await loadCase(name);
  const inp = c.input;
  const eng = new Engine(ort);
  // 画面と同じく既定は CPU（WASM）。?gpu=1 で WebGPU（F0・NSF-HiFiGAN）を試す
  const index = await eng.init(BASE, false, new URLSearchParams(location.search).get("gpu") === "1");
  log(`計算の場所: F0 ${eng.device("f0")}、音響 ${eng.device("acoustic")}、NHVSing ${eng.device("nhvsing")}、NSF-HiFiGAN ${eng.device("nsf_hifigan")}`);
  // 雑音は Python 版と同じもの（句の順に使う）
  const f0q = c.phrases.map((p: any) => p.f0_run.feeds.noise as Float32Array);
  const excq = c.phrases.map((p: any) => p.synth[0].ac_run.feeds.exc_noise as Float32Array);
  const vocq = c.phrases.map((p: any) => p.synth[0].voc_run.feeds.noise as Float32Array);
  eng.noise = { f0: () => f0q.shift() ?? null, exc: () => excq.shift() ?? null, voc: () => vocq.shift() ?? null };
  const vbKey = inp.vb ?? "leapsinger_ja_3singer"; // 英語のケースは手元の英語のボイスバンク（dump_golden.py の GOLDEN_EN_VB）
  const t0 = performance.now();
  const an = await eng.analyze({
    voicebank: vbKey, singer: inp.singer, bpm: inp.bpm, seed: inp.seed,
    notes: inp.notes.map((n: any) => ({ ...n, retake_f0: inp.retake_f0?.[n.id] ?? 0, ph_shift: inp.ph_shift?.[n.id] ?? {} })),
    lyric_style: inp.lyric_style ?? "sequential",
  });
  log(`解析 ${(performance.now() - t0).toFixed(0)} ms`);
  same("句の数", an.phrases.length, c.phrases.length);
  for (let i = 0; i < Math.min(an.phrases.length, c.phrases.length); i++) {
    const g = an.phrases[i], w = c.phrases[i], res = w.result;
    same(`句 ${i} の音素`, g.phonemes.map((p) => p.ph), res.phonemes);
    same(`句 ${i} の音素の長さ（フレーム）`, g.phonemes.map((p) => p.frames), Array.from(res.ph_durs as Int32Array));
    same(`句 ${i} の頭のフレーム`, g.start_frame, Math.round(res.start_sec * c.frame_rate));
    close(`句 ${i} の抑揚の線（log2）`, g.plain, res.plain_log2, 1e-4);
    close(`句 ${i} のビブラート（log2）`, g.vib, res.vib_log2, 1e-4);
    close(`句 ${i} の有声`, g.uv, res.uv, 0);
    close(`句 ${i} の土台（log2）`, g.base, res.base_log2, 1e-5);
    for (let j = 0; j < w.synth.length; j++) {
      const s = w.synth[j];
      if (!index.vocoders.some((v) => v.key === s.vocoder)) {
        // 正解は Python 版が持つボコーダ全部で作るが、ブラウザ版の一覧（web_index.json）に無いもの（publish: false）は比べない
        log(`飛ばす: ${s.vocoder}（ブラウザ版の一覧に無い）`);
        continue;
      }
      const log2 = new Float32Array(s.f0_hz.length);
      for (let k = 0; k < log2.length; k++) log2[k] = Math.log2(s.f0_hz[k]);
      const retakeMel: Record<string, number> = {};
      for (const [id, v] of Object.entries(inp.retake_mel ?? {})) retakeMel[id] = Number(v);
      const t1 = performance.now();
      const r = await eng.synthesize({ voicebank: vbKey, singer: inp.singer, key: g.key, f0_log2: log2, vocoder: s.vocoder,
                                       params: s.params ?? {}, mel_out: true, mel_plain: true, retake_mel: retakeMel,
                                       ph_dip: inp.ph_dip ?? {} });
      log(`合成（${s.vocoder}）${(performance.now() - t1).toFixed(0)} ms`);
      // 音響モデルの出力は、onnxruntime-web と手元の onnxruntime の計算の違いで、最大 0.09・平均 0.0012（ln）ほどずれる
      // （同じ入力をモデルだけに入れても同じ差になる。?model_only= で確かめられる）。前後の処理の誤りは
      // node の試験（durations・f0・synth）でビット単位まで見ているので、ここではその違いの範囲かどうかを見る
      close(`句 ${i} ${s.kind} の mel（ボコーダの入力）`, r.mel_in, s.result.mel_in, 0.15, 3e-3);
      close(`句 ${i} ${s.kind} の波形`, r.wav, s.result.wav, 5e-3, 1e-4);
      close(`句 ${i} ${s.kind} の出た音の mel`, r.mel_out!, s.mel_out, 0.15, 3e-3);
    }
  }
}

/**
 * モデルだけの比べ: 正解と同じ入力を音響モデルにそのまま入れ、出力を正解（手元の onnxruntime の出力）と比べる。
 * 前後の処理を通さないので、ここで出る差は onnxruntime-web と手元の onnxruntime の計算の違いだけ
 */
async function modelOnly(name: string): Promise<void> {
  const c = await loadCase(name);
  const s = c.phrases[0].synth[0];
  const sess = await ort.InferenceSession.create(`${BASE}voicebanks/leapsinger_ja_3singer/acoustic.onnx`,
                                                 { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const feeds: Record<string, any> = {};
  const shapes: Record<string, number[]> = { tokens: [1, -1], durations: [1, -1], f0: [1, -1], spk_embed: [1, 256], exc_noise: [1, -1],
                                             x0_loggain: [1, -1] };
  for (const [k, v] of Object.entries(s.ac_run.feeds as Record<string, any>)) {
    const dims = shapes[k].map((d) => (d === -1 ? (v as ArrayLike<number>).length : d));
    feeds[k] = v instanceof Int32Array ? new ort.Tensor("int64", BigInt64Array.from(Array.from(v, (x) => BigInt(x))), dims)
      : new ort.Tensor("float32", v, dims);
  }
  const out = await sess.run(feeds, ["mel"]);
  const got = out.mel.data as Float32Array, want = s.ac_run.outputs.mel as Float32Array;
  close(`${name} 音響モデルだけ（同じ入力）の mel`, got, want, 1e9);
  const d = Array.from(got, (x, i) => Math.abs(x - want[i])).sort((a, b) => a - b);
  log(`    差の分布: 中央 ${d[d.length >> 1].toExponential(2)}、99% ${d[Math.floor(d.length * 0.99)].toExponential(2)}、99.9% ${d[Math.floor(d.length * 0.999)].toExponential(2)}`);
}

(async () => {
  const q = new URLSearchParams(location.search);
  if (q.get("model_only")) {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;
    try {
      await modelOnly(q.get("model_only")!);
    } catch (e) {
      log(`エラー: ${e instanceof Error ? e.stack : String(e)}`);
    }
    (window as any).__result = { failures: 0, log: lines.join("\n") };
    return;
  }
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = 1;
  try {
    for (const name of (q.get("cases") ?? "sakura,mixed,params,dips").split(",")) await runCase(name);
    log(failures ? `\n外れ ${failures} 件` : "\nすべて一致");
  } catch (e) {
    failures++;
    log(`エラー: ${e instanceof Error ? e.stack : String(e)}`);
  }
  (window as any).__result = { failures, log: lines.join("\n") };
})();
