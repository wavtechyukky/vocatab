// 韓国語の G2P（g2p_ko.ts）を Python（PrettyPitch の svs/g2p_ko.py）と照らし合わせる。
// 正解は tools/golden/dump_g2p_ko.py が書き出す（ランダムな音節の並びと、音節をまたぐ組み合わせの総当たり）。
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { lyricSyllables, syllablesToPhonemes } from "../src/engine-web/g2p_ko";
import type { VoicebankWeb } from "../src/engine-web/types";
import { done, hasVb, repoRoot, same, vbFor } from "./golden";

const g2p = (word: string): string => syllablesToPhonemes(Array.from(word)).map((p) => p.join(" ")).join(" | ");

// PrettyPitch の tests/test_g2p_ko.py の単語と同じ
const WORDS: Array<[string, string]> = [
  ["사랑해", "s a | r a NG | h e"], ["음악", "eu | m a K"], ["있어", "i | ss eo"], ["좋아", "j o | a"],
  ["좋다", "j o | t a"], ["많다", "m a N | t a"], ["싫어", "s i | r eo"], ["놓는", "n o N | n eu N"],
  ["읽어", "i L | g eo"], ["값이", "g a P | ss i"], ["앉아", "a N | j a"], ["닭", "d a K"], ["맑게", "m a L | kk e"],
  ["굳이", "g u | j i"], ["같이", "g a | ch i"], ["꽃이", "kk o | ch i"], ["축하", "ch u | k a"],
  ["국물", "g u NG | m u L"], ["밥만", "b a M | m a N"], ["종로", "j o NG | n o"], ["국립", "g u NG | n i P"],
  ["신라", "s i L | r a"], ["칼날", "k a L | r a L"], ["흘러", "h eu L | r eo"], ["학교", "h a K | kk y o"],
  ["없어", "eo P | ss eo"], ["희망", "h i | m a NG"], ["의자", "eu i | j a"], ["저", "j eo"], ["져", "j eo"],
  ["왜", "w e"], ["외", "w e"], ["귀", "g w i"],
];
for (const [w, want] of WORDS) same(`単語 ${w}`, g2p(w), want);
same("歌詞から音節だけを取り出す", lyricSyllables("아리 랑~!"), ["아", "리", "랑"]);

// Python の結果と総当たりで比べる（外れは最初の 10 個まで出す）
const cases = JSON.parse(readFileSync(join(repoRoot(), "ui/test/golden/g2p_ko.json"), "utf-8")) as
  Array<{ in: string[]; out: string[][] }>;
let bad = 0;
for (const c of cases) {
  const got = syllablesToPhonemes(c.in);
  if (JSON.stringify(got) !== JSON.stringify(c.out)) {
    if (bad < 10) same(`Python と同じ: ${c.in.join("")}`, got, c.out);
    bad++;
  }
}
same(`Python と同じ（${cases.length} 並び）: 外れの数`, bad, 0);

// 出てくる音素はすべて韓国語の音素表にある（韓国語のボイスバンクが手元にあるときだけ）
if (hasVb("korean")) {
  const vb = vbFor("korean") as VoicebankWeb;
  const known = new Set(vb.phoneset.phonemes);
  const unknown = new Set<string>();
  for (const c of cases) for (const p of syllablesToPhonemes(c.in).flat()) if (!known.has(p)) unknown.add(p);
  same("音素表に無い音素", [...unknown], []);
} else {
  console.log("（音素表の確かめは飛ばす: 韓国語のボイスバンクが手元に無い）");
}
done();
