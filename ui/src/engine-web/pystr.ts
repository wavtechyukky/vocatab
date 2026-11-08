// Python の文字列の扱いに合わせる道具（歌詞の読み方を PrettyPitch の Python と同じにするため）。

// Python の str.split()・str.strip()・正規表現の \s が空白とみなす文字（JS の \s とは少し違う: \x1c〜\x1f・\x85 を含み、﻿ を含まない）
export const PY_WS = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WS_RUN = new RegExp(`[${PY_WS}]+`);
const WS_EDGE = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, "g");

/** str.split()（引数なし）と同じ: 空白の並びで切り、空の切れ端は捨てる */
export const pySplit = (s: string): string[] => s.split(WS_RUN).filter((t) => t !== "");
/** str.strip() と同じ */
export const pyStrip = (s: string): string => s.replace(WS_EDGE, "");
/** str.strip(chars) と同じ: 両端から chars のどれかの文字を除く（chars はコードポイントの集まり） */
export function pyStripChars(s: string, chars: string): string {
  const set = new Set(Array.from(chars));
  const cs = Array.from(s);
  let a = 0;
  let b = cs.length;
  while (a < b && set.has(cs[a])) a++;
  while (b > a && set.has(cs[b - 1])) b--;
  return cs.slice(a, b).join("");
}
/** str.isascii() と同じ（空文字も真） */
export const isAscii = (s: string): boolean => /^[\x00-\x7f]*$/.test(s);

/** 文字列の repr（エラーの文面を Python と揃えるため。印字できる文字はそのまま） */
export function pyRepr(s: string): string {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (ch === "\\") out += "\\\\";
    else if (ch === q) out += "\\" + q;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (c < 0x20 || c === 0x7f) out += "\\x" + c.toString(16).padStart(2, "0");
    else out += ch;
  }
  return out + q;
}
