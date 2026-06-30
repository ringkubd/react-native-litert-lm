/**
 * LiteRT-LM React Native — TypeScript Type Definitions
 *
 * This package wraps Google LiteRT-LM for on-device LLM inference on Android.
 * All processing runs locally — no network calls, no data leaves the device.
 *
 * @module react-native-litert-lm
 */

/** Configuration for loading a LiteRT-LM model into memory. */
export type LiteRTLMModelConfig = {
  /**
   * Absolute path to the `.task` model file on the device.
   *
   * Example: `/data/user/0/com.example.app/files/gemma.litertlm`
   *
   * For bundled assets, use `expo-file-system` or `react-native-fs` to
   * copy the file to a writable location first.
   */
  modelPath: string;

  /**
   * Maximum context length (input + output tokens).
   *
   * Higher values use more RAM. Default: 512
   */
  maxTokens?: number;

  /**
   * Preferred compute backend.
   *
   * - `"auto"`  : detect best available (GPU > CPU)
   * - `"cpu"`   : force CPU inference (lowest RAM, works everywhere)
   * - `"gpu"`   : use GPU delegate (faster, requires OpenGL/OpenCL/Vulkan)
   * - `"npu"`   : use NPU delegate (fastest, hardware-dependent)
   *
   * Falls back gracefully if the requested backend is unavailable.
   * Default: `"auto"`
   */
  preferredBackend?: 'auto' | 'cpu' | 'gpu' | 'npu';
};

/** Configuration for a single text-generation call. */
export type LiteRTLMGenerationConfig = {
  /** The input prompt text. */
  prompt: string;

  /**
   * Sampling temperature (0.0–1.0).
   *
   * Lower = more deterministic, higher = more creative.
   * Default: 0.0
   */
  temperature?: number;

  /**
   * Top-K sampling: limits next-token candidates to the K most probable.
   * Default: 40
   */
  topK?: number;

  /**
   * Top-P (nucleus) sampling: cumulative probability threshold.
   * Default: 0.9
   */
  topP?: number;

  /**
   * Maximum number of tokens to generate.
   * Default: 256
   */
  maxOutputTokens?: number;

  /**
   * Sequences where generation stops (e.g. ["\n\n", "<|im_end|>"]).
   * Default: []
   */
  stopSequences?: string[];
};

/** Result returned after a successful non-streaming generation. */
export type LiteRTLMGenerationResult = {
  /** The generated text (without the input prompt). */
  text: string;

  /** Number of tokens generated. */
  tokenCount: number;

  /** Wall-clock generation time in milliseconds. */
  timeMs: number;

  /** Tokens per second (tokens / timeMs * 1000). */
  tokensPerSecond: number;
};

/** Error payload emitted when generation fails. */
export type LiteRTLMGenerationError = {
  /** Human-readable error message. */
  message: string;

  /**
   * Machine-readable error code.
   * One of: "model_not_loaded" | "generation_failed" | "cancelled" |
   *         "invalid_prompt" | "out_of_memory" | "backend_unavailable"
   */
  code: string;
};

/** Device-support information. */
export type LiteRTLMDeviceSupport = {
  /** Whether LiteRT-LM is available on this device / Android version. */
  supported: boolean;

  /** Android SDK version (API level), e.g. 33 for Android 13. */
  androidVersion: number;

  /** Backends that the device reports as available. */
  availableBackends: string[];

  /** Human-readable explanation if `supported` is false. */
  reason?: string;
};

/**
 * A known model that can be downloaded from a remote URL.
 * Used with `LiteRTLM.downloadModel()` and the built-in registry.
 */
export type LiteRTLMModelInfo = {
  /** Display name shown in UI (e.g. "Gemma 2 2B"). */
  name: string;
  /** Unique identifier (e.g. "gemma-2-2b"). */
  id: string;
  /** Download URL for the `.task` file. */
  url: string;
  /** Expected file size in bytes (for progress tracking). */
  sizeBytes: number;
  /** Recommended max tokens for this model. */
  maxTokens: number;
  /** Recommended backend. */
  backend?: 'auto' | 'cpu' | 'gpu' | 'npu';
  /** Short description of strengths. */
  description?: string;
};

/** Progress information emitted during model download. */
export type LiteRTLMDownloadProgress = {
  /** Bytes downloaded so far. */
  bytesDownloaded: number;
  /** Total bytes to download. */
  bytesTotal: number;
  /** Progress fraction (0.0 – 1.0). */
  progress: number;
};

/** Configuration for downloading a model from a URL. */
export type LiteRTLMDownloadConfig = {
  /** Source URL of the `.task` model file. */
  url: string;
  /** Local path to save the file (must be writable). */
  destinationPath: string;
  /** Expected total size in bytes (optional, for progress %). */
  expectedSizeBytes?: number;
};

/** Performance statistics for the loaded model. */
export type LiteRTLMPerfStats = {
  /** Time taken to load and initialise the model (ms). */
  loadTimeMs: number;

  /** Total tokens generated across all sessions. */
  totalTokensGenerated: number;

  /** Total generation time across all sessions (ms). */
  totalGenerationTimeMs: number;

  /** Whether warm-up inference has been run. */
  warmedUp: boolean;
};

/**
 * A subscription handle that can be used to cancel streaming generation.
 */
export type LiteRTLMStreamSubscription = {
  /** Cancel the in-progress generation. No more tokens will be emitted. */
  cancel: () => void;
};

/**
 * Handlers for streaming token output.
 */
export type LiteRTLMStreamHandlers = {
  /** Called for each partial token as it is generated. */
  onToken?: (token: string) => void;

  /** Called once when generation completes successfully. */
  onComplete?: (result: LiteRTLMGenerationResult) => void;

  /** Called if an error occurs during generation. */
  onError?: (error: LiteRTLMGenerationError) => void;
};
