// 英語の G2P（PrettyPitch の svs/g2p_en.py と、svs/score_io.py の _en_mark・_distribute_word・_apply_g2p_en と同じ）。
// 歌詞 → en_full の表記の音素。単語は発音辞書（PrettyPitch の dict/en_cmudict.txt の写し。英語のボイスバンクを使うときだけ
// 読む）で引き、休符を挟まずに続く記号の音符へ音節を前から順に配る（+ は次の音節、- と +~ は伸ばし、+3 は単語の 3 番目の
// 音素から）。ui/test/g2p_en.test.ts で Python と照らし合わせる。
import { syllabify } from "./durations";
import type { PhoneSet } from "./phoneset";
import { PY_WS, pyRepr, pySplit, pyStripChars } from "./pystr";
import type { ScoreNote } from "./types";

export type EnDict = Map<string, string[]>;

/** 英語の + の読み方（score_io.LYRIC_STYLES）。違うのは ++ と、音符より多い音節の扱いだけ
 *   sequential: + ごとに次の 1 音節（++ も + と同じ）。余った音節は、音節を受け取った最後の音符にまとめて歌う
 *   count:      + の数だけ音節を歌う（++ で 2 音節）。単語の音符は、+ が続くときは最初の 1 音節だけ。余った音節は歌わない */
export const LYRIC_STYLES = ["sequential", "count"] as const;
export type LyricStyle = (typeof LYRIC_STYLES)[number];

const PUNCT = ".,!?;:\"“”()[]{}…«»¡¿";
const BRACKET = new RegExp(`^[${PY_WS}]*([^\\[\\]]*)\\[([^\\[\\]]*)\\][${PY_WS}]*$`);
const EN_HOLD = new Set(["", "-", "ー", "ｰ"]); // 伸ばし: 前の母音を続ける（音節は進めない）

// ── 発音辞書 ──

// str.splitlines() が行の切れ目とみなす文字
const LINE_BREAKS = new RegExp("\\r\\n|[\\n\\r\\x0b\\x0c\\x1c\\x1d\\x1e\\x85\\u2028\\u2029]");

/** dict/en_cmudict.txt の中身 → 単語（小文字）→ 音素の列（g2p_en.load_dict。同じ見出しは後ろの行が勝つ） */
export function parseDict(text: string): EnDict {
  const out: EnDict = new Map();
  for (const line of text.split(LINE_BREAKS)) {
    if (!line || line.startsWith("#")) continue;
    const toks = pySplit(line);
    if (!toks.length) continue;
    out.set(toks[0], toks.slice(1));
  }
  return out;
}

// 読み込んだ辞書（web.json の g2p_dict の場所ごと。英語のボイスバンクが同じ辞書を使い回す）
const loaded = new Map<string, EnDict>();
export const registerDict = (path: string, d: EnDict): void => void loaded.set(path, d);
export const dictAt = (path: string): EnDict | undefined => loaded.get(path);

/** 歌詞の 1 語 → 辞書の見出し（無ければ null）。前後の句読点を捨て、それでも無ければ前後の ' も捨てる（'cause は残す） */
function normWord(d: EnDict, tok: string): string | null {
  let t = pyStripChars(tok.replace(/[’‘]/g, "'").toLowerCase(), PUNCT);
  if (d.has(t)) return t;
  t = pyStripChars(t, "'");
  return d.has(t) ? t : null;
}

/** 歌詞 → 辞書の見出しの列（g2p_en.lyric_words）。語が 1 つも無いか、辞書に無い語があれば null */
export function lyricWords(d: EnDict, lyric: string): string[] | null {
  const words: string[] = [];
  for (const tok of pySplit(lyric)) {
    for (const part of tok.split("-")) {
      if (!pyStripChars(part, PUNCT + "'’‘")) continue; // ハイフンや句読点だけの切れ端
      const w = normWord(d, part);
      if (w === null) return null;
      words.push(w);
    }
  }
  return words.length ? words : null;
}

const fold = (ps: PhoneSet, tok: string): string => (tok.toUpperCase() === "AH0" ? "ax" : ps.fold(tok));

/**
 * 歌詞が音素の書き方なら音素の列、そうでなければ null（g2p_en.explicit_phonemes）。書き方は
 *   [hh eh l ow]（hello[hh eh l ow] のように前に単語を書いてもよい）・空白で区切った 2 つ以上の音素・子音の音素 1 つ。
 * 角かっこの中に音素表に無い記号があれば例外
 */
export function explicitPhonemes(ps: PhoneSet, lyric: string): string[] | null {
  const m = BRACKET.exec(lyric);
  if (m) {
    const toks = pySplit(m[2]).map((x) => fold(ps, x));
    const bad = toks.filter((x) => !ps.ph2id.has(x));
    if (!toks.length || bad.length) {
      const shown = bad.length ? `[${bad.map(pyRepr).join(", ")}]` : "（空）";
      throw new Error(`歌詞 ${pyRepr(lyric)} の [ ] の中に音素表に無い記号があります: ${shown}`);
    }
    return toks;
  }
  const toks = pySplit(lyric).map((x) => fold(ps, x));
  if (toks.length >= 2 && toks.every((x) => ps.ph2id.has(x))) return toks;
  if (toks.length === 1 && ps.consSet.has(toks[0])) return toks;
  return null;
}

/** 歌詞 → 音素の列（音素の書き方なら音素、そうでなければ辞書。g2p_en.lyric_phonemes）。どちらでも読めなければ null */
export function lyricPhonemes(ps: PhoneSet, d: EnDict, lyric: string): string[] | null {
  const ph = explicitPhonemes(ps, lyric);
  if (ph !== null) return ph;
  const words = lyricWords(d, lyric);
  return words ? words.flatMap((w) => d.get(w)!) : null;
}

// ── 単語の音節を音符へ配る ──

export type EnMark = ["hold" | "next" | "at", number];

/**
 * 英語の歌詞が単語に続く記号なら [種類, 数]、記号でなければ null（score_io._en_mark）。種類は
 * hold（伸ばし: 歌詞なし・空・-・ー・+~・+*）、next（次の音節へ。数は + の数）、at（+3: 単語の 3 番目の音素から）。
 * + で始まるその他の歌詞は + と同じ
 */
export function enMark(lyric: string | null): EnMark | null {
  if (lyric === null || EN_HOLD.has(lyric)) return ["hold", 0];
  if (!lyric.startsWith("+")) return null;
  const body = lyric.slice(1);
  const c = body.slice(0, 1);
  if (c === "~" || c === "*") return ["hold", 0];
  if (/^[0-9]+$/.test(body)) return ["at", Number(body)]; // Python の isdigit は全角の数字なども含むが、ここは半角だけ
  return /^\+*$/.test(body) ? ["next", lyric.length] : ["next", 1];
}

/**
 * 単語の音素 phs を、単語の音符（0 番）と続く記号の音符（1 番〜、marks）に配る（score_io._distribute_word）。
 * 音符ごとの音素（配らない音符は null）。受け取る音節が残っていない + は伸ばしになる。伸ばしの音符は直前の音節の
 * 母音を続け、その音節の終わりの子音は伸ばしの最後の音符へ移す（「skies」「-」→ s k ay | ay z）
 */
export function distributeWord(phs: string[], marks: EnMark[], ps: PhoneSet, counted: boolean): Array<string[] | null> {
  const L = phs.length;
  const bounds: number[] = []; // 各音節の終わり（= 次の音節の頭。最後は L）
  let s = 0;
  for (const syl of syllabify(phs, ps)) {
    s += syl.length;
    bounds.push(s);
  }
  const endOf = (head: number, k: number): number => { // head から k 音節ぶんの終わり（足りなければ単語の終わり）
    const later = bounds.filter((b) => b > head);
    if (!later.length) return L;
    const i = Math.min(k, later.length) - 1;
    return later[i < 0 ? later.length + i : i]; // Python の負の添字と同じ
  };
  const takes = [0];
  const heads = [0]; // 音節を受け取る音符の番号と、その頭の音素の番号
  let end = counted && marks.some((m) => m[0] !== "hold") ? endOf(0, 1) : L;
  marks.forEach(([kind, k0], q) => {
    const n = q + 1;
    if (kind === "hold") return;
    let head: number;
    let k = k0;
    if (kind === "at" && heads[heads.length - 1] < k - 1 && k - 1 < L) {
      head = k - 1;
      k = 1;
    } else {
      head = counted ? end : endOf(heads[heads.length - 1], 1);
      k = counted && kind === "next" ? k : 1;
    }
    if (head >= L) return; // 受け取る音節が残っていない: 伸ばし
    takes.push(n);
    heads.push(head);
    end = endOf(head, k);
  });
  const per: Array<string[] | null> = new Array(marks.length + 1).fill(null);
  takes.forEach((n, t) => {
    const head = heads[t];
    const tail = t + 1 < heads.length ? heads[t + 1] : counted ? end : L;
    let chunk = phs.slice(head, Math.max(head, tail));
    const heldTo = t + 1 < takes.length ? takes[t + 1] : marks.length + 1; // 伸ばしの音符は [n + 1, heldTo)
    let nuc = -1;
    chunk.forEach((p, m) => { if (ps.isNucleus(p)) nuc = m; });
    if (heldTo > n + 1 && nuc >= 0) { // 核の無い音節を伸ばす音符は null のまま（直前の母音を伸ばす）
      const vowel = chunk[nuc];
      const coda = chunk.slice(nuc + 1);
      chunk = chunk.slice(0, nuc + 1);
      for (let h = n + 1; h < heldTo; h++) per[h] = [vowel];
      per[heldTo - 1] = [vowel, ...coda];
    }
    per[n] = chunk;
  });
  return per;
}

/**
 * 英語（言語パックの g2p: en）: 歌詞を音素にし、休符を挟まずに続く記号の音符へ音節を前から順に配る（score_io._apply_g2p_en）。
 * 単語に続いていない記号の音符と、音節を受け取らなかった記号の音符は歌詞なし（前の母音を伸ばす）。
 * 辞書にも無く音素でもない歌詞は例外
 */
export function applyG2pEn(score: ScoreNote[], ps: PhoneSet, d: EnDict, style: string): ScoreNote[] {
  if (!(LYRIC_STYLES as readonly string[]).includes(style)) {
    throw new Error(`lyric_style は (${LYRIC_STYLES.map((x) => pyRepr(x)).join(", ")}) のどれか（${pyRepr(style)}）`);
  }
  const out = score.slice();
  const marks = out.map((sn) => (sn.is_rest || sn.phonemes !== null ? null : enMark(sn.lyric)));
  let i = 0;
  while (i < out.length) {
    const sn = out[i];
    if (sn.is_rest || marks[i] !== null) {
      i += 1;
      continue;
    }
    const phs = sn.phonemes !== null ? sn.phonemes.slice() : lyricPhonemes(ps, d, sn.lyric ?? "");
    if (phs === null) {
      throw new Error(`歌詞 ${pyRepr(sn.lyric ?? "")} は英語の発音辞書にありません`
        + "（音素で書くときは [hh eh l ow] のように [ ] で囲んでください）");
    }
    let j = i + 1; // 続く記号の音符 [i + 1, j)
    while (j < out.length && marks[j] !== null) j += 1;
    distributeWord(phs, marks.slice(i + 1, j) as EnMark[], ps, style === "count").forEach((p, k) => {
      if (p !== null) out[i + k] = { ...out[i + k], phonemes: p };
    });
    i = j;
  }
  // 音素の決まらなかった記号の音符は、歌詞なし（score_to_phrases が前の母音を伸ばす）
  return out.map((sn, q) => (marks[q] !== null && sn.phonemes === null ? { ...sn, lyric: null } : sn));
}
