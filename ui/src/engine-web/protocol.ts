// 画面とエンジンのやり取りの形（engine/server.py の /api/analyze・/api/synthesize と同じ中身。配列は base64 にせずそのまま）。
import type { EPhoneme } from "../engine";
import type { WebIndex } from "./types";

export interface AnalyzeReq {
  voicebank: string;
  singer: number;
  bpm: number;
  seed: number;
  notes: Array<{ id: number; onset: number; dur: number; pitch: number; lyric: string; retake_f0?: number;
                 ph_shift?: Record<string, number> }>;
}

export interface AnalyzedPhrase {
  key: string;
  note_ids: number[];
  start_frame: number;
  /** 句の頭（先頭の pau の頭）の絶対時刻 [s]（丸める前の値。待機から起きたエンジンへ解析の結果を戻すときに要る） */
  start_sec: number;
  n_frames: number;
  phonemes: EPhoneme[];
  plain: Float32Array;
  vib: Float32Array;
  uv: Float32Array;
  base: Float32Array;
  /** 乱数の種（F0 の雑音・音響モデルの励振の雑音・ボコーダの雑音。AnalyzeReq.seed） */
  seed: number;
}

export interface AnalyzeResp {
  frame_rate: number;
  phrases: AnalyzedPhrase[];
}

export interface SynthReq {
  voicebank: string;
  singer: number;
  key: string;
  f0_log2: Float32Array;
  vocoder: string;
  params: Record<string, Float32Array>;
  mel_out: boolean;
  mel_plain: boolean;
  retake_mel: Record<string, number>;
  /** 動かした子音の凹み: 音符の id → {音符の中の音素の番号: 深さ dB・広さ}（ブラウザ版のエンジンはまだ使わない） */
  ph_dip?: Record<string, Record<string, { db?: number; w?: number }>>;
}

export interface SynthResp {
  n_frames: number;
  wav: Float32Array;
  mel_in: Float32Array;
  mel_out?: Float32Array;
  mel_plain?: Float32Array;
}

/**
 * 待機から起こしたエンジンへ戻す、1 句ぶんの解析の結果（analyze が返したものを画面が持っていたもの）。
 * エンジンはここから解析の取り置き（Analysis）を組み立て直すので、中身の変わっていない句は解析し直さずに済む
 */
export interface RestoreEntry {
  key: string;
  vb: string;
  singer?: number;
  ids: number[];
  phonemes: EPhoneme[];
  start_sec: number;
  plain: Float32Array;
  vib: Float32Array;
  uv: Float32Array;
  base: Float32Array;
  seed: number;
}

export interface RestoreResp {
  restored: number;
}

/** ボイスバンクのブレスのパターンの一覧（engine/server.py の /api/breaths と同じ中身）。index は breath に渡す番号 */
export interface BreathIndexResp {
  clips: Array<{ id: string; singer: number; frames: number; sec: number }>;
}

/** ブレス 1 つ（/api/breath）: ブレスの mel をボコーダだけで鳴らした波形。フレーズとは別に作り、画面側で混ぜる
 *  （フレーズを合成し直さずに済み、同じパターンは 1 回鳴らせばよい） */
export interface BreathReq {
  voicebank: string;
  index: number;
  vocoder: string;
}

export interface BreathResp {
  frames: number;
  wav: Float32Array; // [256·frames]
}

/** Worker への呼び出し（id は client.ts が付ける） */
export type WorkerCall =
  | { type: "init"; base: string; gpu: boolean }
  | { type: "analyze"; req: AnalyzeReq }
  | { type: "synthesize"; req: SynthReq }
  | { type: "restore"; entries: RestoreEntry[] }
  | { type: "breaths"; voicebank: string }
  | { type: "breath"; req: BreathReq };

export type WorkerMsg = WorkerCall & { id: number };

export type WorkerReply =
  | { type: "result"; id: number; result: WebIndex | AnalyzeResp | SynthResp | RestoreResp | BreathIndexResp | BreathResp }
  | { type: "error"; id: number; error: string }
  | { type: "progress"; loaded: number; total: number };
