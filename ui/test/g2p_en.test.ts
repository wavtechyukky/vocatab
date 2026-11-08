// 英語の G2P（g2p_en.ts）を Python（PrettyPitch の svs/g2p_en.py と score_io._apply_g2p_en）と照らし合わせる。
// 正解は tools/golden/dump_g2p_en.py が書き出す（en_full の音素表と、使う語だけの発音辞書も入っているので、
// 英語のボイスバンクが無くても動く）。
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { applyG2pEn, type EnDict, enMark, lyricPhonemes, parseDict } from "../src/engine-web/g2p_en";
import { PhoneSet } from "../src/engine-web/phoneset";
import type { ScoreNote } from "../src/engine-web/types";
import { done, repoRoot, same } from "./golden";

const g = JSON.parse(readFileSync(join(repoRoot(), "ui/test/golden/g2p_en.json"), "utf-8")) as {
  phoneset: any;
  dict: Record<string, string[]>;
  lyric_cases: Array<{ lyric: string; out: string[] | null; err: string | null }>;
  score_cases: Array<{ style: string; notes: Array<{ lyric: string | null; is_rest: boolean; phonemes: string[] | null }>;
                       out?: Array<{ lyric: string | null; phonemes: string[] | null }>; err?: string }>;
  bad_style: string;
};
const ps = new PhoneSet(g.phoneset);
const d: EnDict = new Map(Object.entries(g.dict));
const errOf = (f: () => unknown): string | null => {
  try {
    f();
    return null;
  } catch (e) {
    return (e as Error).message;
  }
};

// 発音辞書の読み方（コメント・空行を飛ばし、同じ見出しは後ろの行が勝つ）
same("辞書の読み方", [...parseDict("# c\nfoo f uw\n\nbar b aa r\r\nfoo f ow\n")], [["foo", ["f", "ow"]], ["bar", ["b", "aa", "r"]]]);
// 記号の種類（score_io._en_mark）
same("記号の種類", ["+", "++", "+++", "+~", "+*", "-", "", "ー", "+3", "+x", "+-", "love", null].map((x) => enMark(x)),
     [["next", 1], ["next", 2], ["next", 3], ["hold", 0], ["hold", 0], ["hold", 0], ["hold", 0], ["hold", 0], ["at", 3],
      ["next", 1], ["next", 1], null, ["hold", 0]]);

// 歌詞 1 つずつの読み
let bad = 0;
for (const c of g.lyric_cases) {
  let out: string[] | null = null;
  const err = errOf(() => { out = lyricPhonemes(ps, d, c.lyric); });
  const got = { out, err };
  const want = { out: c.out, err: c.err };
  if (JSON.stringify(got) !== JSON.stringify(want)) {
    if (bad < 10) same(`歌詞 ${JSON.stringify(c.lyric)}`, got, want);
    bad++;
  }
}
same(`歌詞の読みが Python と同じ（${g.lyric_cases.length} 個）: 外れの数`, bad, 0);

// 単語に続く記号の音符への音節の配り方（2 つの読み方）
bad = 0;
for (const c of g.score_cases) {
  const score: ScoreNote[] = c.notes.map((n) => ({ dur_beats: 1, midi: n.is_rest ? 0 : 60, lyric: n.lyric, is_rest: n.is_rest,
                                                  phonemes: n.phonemes }));
  let out: Array<{ lyric: string | null; phonemes: string[] | null }> | undefined;
  const err = errOf(() => { out = applyG2pEn(score, ps, d, c.style).map((n) => ({ lyric: n.lyric, phonemes: n.phonemes })); });
  const got = err === null ? { out } : { err };
  const want = c.err !== undefined ? { err: c.err } : { out: c.out };
  if (JSON.stringify(got) !== JSON.stringify(want)) {
    if (bad < 10) same(`並び ${c.style} ${JSON.stringify(c.notes.map((n) => (n.is_rest ? "R" : n.lyric)))}`, got, want);
    bad++;
  }
}
same(`音節の配り方が Python と同じ（${g.score_cases.length} 並び）: 外れの数`, bad, 0);
same("知らない読み方のエラー", errOf(() => applyG2pEn([], ps, d, "bogus")), g.bad_style);
done();
