// ブラウザ版のエンジンが使う形。Python のエンジン（engine/pipeline.py と PrettyPitch のホスト側の処理）と同じ中身を持つ。
// 設定は tools/export_web_assets.py が書き出す web_index.json・voicebanks/<名前>/web.json から読む。

/** 1 言語ぶんの音素の表（PrettyPitch の duration_model/phoneset.py の PhoneSet と同じ中身）とモーラ表。 */
export interface PhoneSetData {
  lang: string;
  phonemes: string[]; // [0] は pau
  nuclei: string[];
  vowels: string[];
  vowel_normalize: Record<string, string>;
  silence: string[];
  consonants: string[];
  sil_map: Record<string, string>;
  case_fold: "lower" | "upper" | null;
  strip_stress_digits: boolean;
  /** 後ろに核の無い子音（句末・休みの前の終声）も子音の長さのモデルで予測するか（韓国語）。古い web.json には無い */
  final_consonants?: boolean;
  /** 規則の G2P（"ko" = g2p_ko.ts）。無ければモーラ表だけ。古い web.json には無い */
  g2p?: string | null;
  score: { rest_lyrics?: string[]; melisma_lyrics?: string[]; melisma_vowel?: string; [k: string]: unknown };
  mora: Record<string, string[]>; // かな → 音素の列
}

/** 子音の長さの下限など（ConsDurProfile を歌い手ごとに引いた値）。 */
export interface ProfileTable {
  v_min: number;
  c_min: Record<string, number>;
  lead_c_min: Record<string, number>;
  weight: Record<string, number>;
}

/** voicebanks/<名前>/web.json */
export interface VoicebankWeb {
  key: string;
  name: string;
  language: string;
  grid: { sample_rate: number; hop: number };
  frame_rate: number;
  models: { acoustic: string; f0: string; cons_dur: string }; // リポジトリの頭から見た相対パス
  phoneset: PhoneSetData;
  /** 鍵は歌い手の id（文字列）、"*" は歌い手を指定しないとき。enabled は表全体の有効・無効 */
  profile: { enabled: boolean } & Record<string, ProfileTable | boolean>;
  f0: Record<string, any>; // f0.json（frame_rate・phonemes・nucleus・pitch_center・n_noise・n_styles・baseline_sigma_sec など）
  cons_dur: Record<string, any>; // cons_dur.json
  speakers: { phonemes: string[]; speakers: Record<string, number[]> };
  /** ブレスの mel の一覧（breaths.json）の場所。無いボイスバンクはブレスを入れられない */
  breaths?: string;
  /** 子音の凹みのレシピ（voicebank.yaml の x0_shape。engine/x0shape.py・x0shape.ts）。無ければ凹みを作らない */
  x0_shape?: Record<string, any>;
  /** 音響モデルの入力の名前（exc_noise・x0_loggain を持つかを、モデルを読む前に知るため） */
  acoustic_inputs?: string[];
}

/** voicebanks/<名前>/breaths.json（tools/breaths/build_breath_bank.py が書く）。mel は float16 の bin に [128, frames] 行優先で並ぶ */
export interface BreathBank {
  version: number;
  bin: string;        // breaths.json から見た相対
  bin_bytes: number;
  n_mels: number;
  hop: number;
  clips: Array<{ id: string; singer: number; offset: number; frames: number; sec: number; gain_db: number }>;
}

/** web_index.json の 1 行 */
export interface VocoderWeb {
  key: string;
  name: string;
  kind: "nhvsing" | "nsf_hifigan" | string;
  voice_params: boolean;
  hop: number;
  gain_db: number;
  onnx: string;
  bytes: number;
  grid: { sample_rate?: number; hop?: number };
}

/** 歌い手: name は内部の名前（プロジェクトに書かれる）、display は画面に出すフルネーム（言語 → 名前。voicebank.yaml から） */
export interface SingerInfo {
  id: number;
  name: string;
  display?: Record<string, string>;
}

export interface WebIndex {
  voicebanks: Array<{ key: string; name: string; language: string; singers: Array<SingerInfo>;
                      default_vocoder: string; credits: string; web: string; bytes: Record<string, number> }>;
  vocoders: VocoderWeb[];
}

// ── 楽譜と句（PrettyPitch の svs/score_io.py・svs/duration_postprocess.py と同じ） ──

/** 楽譜の 1 音符（休符を含む）。ScoreNote と同じ */
export interface ScoreNote {
  dur_beats: number;
  midi: number; // 休符は 0
  lyric: string | null;
  is_rest: boolean;
  phonemes: string[] | null; // 明示した音素の列（あれば優先）
}

/** 句の中の 1 音符。Note と同じ */
export interface PNote {
  phonemes: string[];
  pitch_midi: number;
  dur_sec: number;
  gap_after_sec: number;
}

/** 句。Phrase と同じ */
export interface Phrase {
  notes: PNote[];
  leading_pau_sec: number;
  trailing_pau_sec: number;
  start_sec: number;
}

/** 長さの後処理の出力。PhonemeSequence と同じ（dur_sec は float32） */
export interface PhonemeSequence {
  phonemes: string[];
  dur_sec: Float32Array;
  note_midi: Int32Array;
  is_vowel: Uint8Array;
  note_event_starts: Uint8Array;
  leading_pau_sec: number;
}

// ── ONNX のモデルへの入力 ──

export interface Feed {
  type: "float32" | "int64";
  data: Float32Array | BigInt64Array;
  dims: number[];
}
export type Feeds = Record<string, Feed>;

export const f32 = (data: Float32Array, dims: number[]): Feed => ({ type: "float32", data, dims });
export const i64 = (data: ArrayLike<number>, dims: number[]): Feed =>
  ({ type: "int64", data: BigInt64Array.from(Array.from(data, (x) => BigInt(Math.trunc(x)))), dims });
