// 子音の凹み（x0shape.ts）を Python（engine/x0shape.py）と照らし合わせる。node で動く。
// 正解は tools/golden/dump_x0shape.py が書き出す（ランダムな句 × レシピの変種 × 手で動かした凹み × 滑舌の倍率）。
//   npx esbuild@0.25.0 ui/test/x0shape.test.ts --bundle --platform=node --format=esm --outfile=<作業用>/t.mjs && node <作業用>/t.mjs
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { dips, loggain } from "../src/engine-web/x0shape";
import { done, repoRoot, same } from "./golden";

const g = JSON.parse(readFileSync(join(repoRoot(), "ui/test/golden/x0shape.json"), "utf-8")) as {
  nucleus: Record<string, string[]>;
  cases: Array<{ lang: string; cfg: Record<string, any>; phonemes: string[]; durs: number[];
                 overrides: Record<string, { amp: number; w: number }>; scales: Record<string, number>;
                 dips: Array<{ amp: number; w: number } | null>; loggain: number[] | null }>;
};
let badDips = 0, badGain = 0, nVals = 0;
const shown: string[] = [];
g.cases.forEach((c, n) => {
  const nuc = new Set(g.nucleus[c.lang]);
  const isCons = (p: string) => p !== "pau" && !nuc.has(p);
  const d = dips(c.phonemes, c.durs, c.cfg, c.lang, isCons);
  if (JSON.stringify(d) !== JSON.stringify(c.dips)) {
    badDips++;
    if (shown.length < 5) shown.push(`句 ${n} の dips: ${JSON.stringify(d).slice(0, 200)} / 正解 ${JSON.stringify(c.dips).slice(0, 200)}`);
  }
  const ov = new Map(Object.entries(c.overrides).map(([k, v]) => [Number(k), v]));
  const sc = new Map(Object.entries(c.scales).map(([k, v]) => [Number(k), v]));
  const got = loggain(c.phonemes, c.durs, c.cfg, c.lang, ov.size ? ov : null, sc.size ? sc : null);
  if ((got === null) !== (c.loggain === null)) {
    badGain++;
    if (shown.length < 5) shown.push(`句 ${n} の loggain: 片方だけ null`);
    return;
  }
  if (!got || !c.loggain) return;
  nVals += got.length;
  if (got.length !== c.loggain.length || got.some((x, i) => x !== c.loggain![i])) {
    badGain++;
    if (shown.length < 5) {
      const i = got.findIndex((x, k) => x !== c.loggain![k]);
      shown.push(`句 ${n} の loggain: ${i} 番目 ${got[i]} / 正解 ${c.loggain[i]}（長さ ${got.length} / ${c.loggain.length}）`);
    }
  }
});
for (const s of shown) console.log(`    ${s}`);
same(`dips が Python と一致（${g.cases.length} 句）`, badDips, 0);
same(`loggain が Python とビット一致（${g.cases.length} 句、${nVals} 値）`, badGain, 0);
done();
