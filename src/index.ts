/**
 * react-native-litert-lm
 *
 * On-device LLM inference for React Native & Expo using Google LiteRT-LM.
 * Fully offline — no network calls, no data leaves the device.
 *
 * @example
 * ```ts
 * import { LiteRTLM } from "react-native-litert-lm";
 *
 * const model = await LiteRTLM.loadModel({
 *   modelPath: "/data/user/0/com.example/files/gemma.task",
 * });
 *
 * const result = await model.generate({
 *   prompt: "Extract expense JSON from: bought rice 500 taka",
 * });
 * console.log(result.text);
 *
 * await model.release();
 * ```
 *
 * @module react-native-litert-lm
 */

import { NativeModule, type NativeLiteRTLM } from './NativeLiteRTLM';
import type {
  LiteRTLMModelConfig,
  LiteRTLMGenerationConfig,
  LiteRTLMGenerationResult,
  LiteRTLMGenerationError,
  LiteRTLMDeviceSupport,
  LiteRTLMPerfStats,
  LiteRTLMPerfStats,
  LiteRTLMStreamSubscription,
  LiteRTLMStreamHandlers,
  LiteRTLMModelInfo,
  LiteRTLMDownloadProgress,
  LiteRTLMDownloadConfig,
} from './types';

export type {
  LiteRTLMModelConfig,
  LiteRTLMGenerationConfig,
  LiteRTLMGenerationResult,
  LiteRTLMGenerationError,
  LiteRTLMDeviceSupport,
  LiteRTLMPerfStats,
  LiteRTLMPerfStats,
  LiteRTLMStreamSubscription,
  LiteRTLMStreamHandlers,
  LiteRTLMModelInfo,
  LiteRTLMDownloadProgress,
  LiteRTLMDownloadConfig,
};

// ─── Error Codes ───────────────────────────────────────────────────────────────

// ─── Built-in Model Registry ──────────────────────────────────────────────────
//
// Users can pick from this list in a UI, or provide a custom URL.
// Models are downloaded from HuggingFace / Google AI Edge.

export const BUILTIN_MODELS: LiteRTLMModelInfo[] = [
  // ── Embedding (Personalization / RAG) ───────────────────────────────
  // Generic model works on ALL devices (CPU/GPU). Auto-detects NPU variant at runtime.
  {
    name: 'EmbeddingGemma 300M',
    id: 'embeddinggemma-300m',
    url: 'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq512_mixed-precision.tflite',
    sizeBytes: 150_000_000,
    maxTokens: 512,
    backend: 'auto',
    description: '768-dim text embeddings for personalization, RAG, and semantic search. Works on all devices.',
  },

  // ── Flagship: Gemma 4 ────────────────────────────────────────────────
  {
    name: 'Gemma 4 E2B',
    id: 'gemma-4-e2b',
    url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
    sizeBytes: 2_583_000_000,
    maxTokens: 8192,
    backend: 'gpu',
    description: 'Best quality, multi-modal (text+vision+audio). 1M+ downloads. Needs 8GB+ RAM.',
  },
  {
    name: 'Gemma 4 E4B',
    id: 'gemma-4-e4b',
    url: 'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm',
    sizeBytes: 4_200_000_000,
    maxTokens: 8192,
    backend: 'gpu',
    description: 'Larger Gemma 4. Best reasoning. Needs 12GB+ RAM.',
  },

  // ── Mid-range: Gemma 3 + Qwen ───────────────────────────────────────
  {
    name: 'Gemma 3 1B',
    id: 'gemma-3-1b',
    url: 'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it.litertlm',
    sizeBytes: 700_000_000,
    maxTokens: 8192,
    backend: 'auto',
    description: 'Fast & efficient. Works on 4GB+ devices.',
  },
  {
    name: 'Llama 3.2 3B',
    id: 'llama-3.2-3b',
    url: 'https://huggingface.co/litert-community/Llama-3.2-3B/resolve/main/model.litertlm',
    sizeBytes: 2_500_000_000,
    maxTokens: 8192,
    backend: 'gpu',
    description: 'Meta Llama 3.2. Strong reasoning. Needs 8GB+ RAM.',
  },
  {
    name: 'Llama 3.2 1B',
    id: 'llama-3.2-1b',
    url: 'https://huggingface.co/litert-community/Llama-3.2-1B/resolve/main/model.litertlm',
    sizeBytes: 900_000_000,
    maxTokens: 4096,
    backend: 'auto',
    description: 'Lightweight Llama. Works on 4GB+ devices.',
  },

  // ── Lightweight: Qwen 3.5 ───────────────────────────────────────────
  // Note: Once Qwen 3.5 .litertlm is published on HuggingFace litert-community,
  // add it here. Until then, users can provide a custom model path.
  // {
  //   name: 'Qwen 3.5 1.5B',
  //   id: 'qwen-3.5-1.5b',
  //   url: 'https://huggingface.co/litert-community/Qwen3.5-1.5B/resolve/main/model.litertlm',
  //   sizeBytes: 1_200_000_000,
  //   maxTokens: 8192,
  //   backend: 'auto',
  //   description: 'Qwen 3.5 optimized for Bengali & English. Works on 4GB+ devices.',
  // },
];

const ERR = {
  MODULE_NOT_FOUND: 'native_module_not_found',
  MODEL_NOT_LOADED: 'model_not_loaded',
  GENERATION_FAILED: 'generation_failed',
  CANCELLED: 'cancelled',
  INVALID_PROMPT: 'invalid_prompt',
  OUT_OF_MEMORY: 'out_of_memory',
  BACKEND_UNAVAILABLE: 'backend_unavailable',
  MODEL_NOT_LOADED_MSG: 'Model is not loaded. Call LiteRTLM.loadModel() first.',
} as const;

function makeError(message: string, code: string): LiteRTLMGenerationError {
  return { message, code };
}

function assertModule(): NativeLiteRTLM {
  if (!NativeModule) {
    throw makeError(
      'Native module not found. This package requires a custom dev build (not Expo Go).',
      ERR.MODULE_NOT_FOUND,
    );
  }
  return NativeModule;
}

// ─── Internal Event-Name Constants ────────────────────────────────────────────

const EVENT_ON_TOKEN = 'onToken';
const EVENT_ON_COMPLETE = 'onComplete';
const EVENT_ON_ERROR = 'onError';

// ─── Loaded-Model Registry ────────────────────────────────────────────────────

let _modelHandle: number | null = null;
let _loadTimeMs = 0;
let _streamingActive = false;

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * LiteRTLM — main entry point for on-device LLM inference.
 *
 * All methods are static. Only one model can be loaded at a time.
 * Call `LiteRTLM.loadModel()` → use the returned model → `model.release()`.
 */
export const LiteRTLM = {
  // ── Download Model ─────────────────────────────────────────────────────────

  /**
   * Download a model file from a URL with progress tracking.
   *
   * Emits `onProgress` callbacks as bytes are received.
   * The downloaded file is saved to `destinationPath` and can be
   * passed directly to `loadModel()`.
   *
   * @returns The absolute path to the downloaded file.
   */
  async downloadModel(
    config: LiteRTLMDownloadConfig,
    onProgress?: (progress: LiteRTLMDownloadProgress) => void,
  ): Promise<string> {
    const mod = assertModule();

    // Subscribe to progress events from native side
    let subscription: any = null;
    if (onProgress && (mod as any).addEventEmitter) {
      subscription = (mod as any).addEventEmitter(
        (eventName: string, data: any) => {
          if (eventName === 'onDownloadProgress' && data) {
            onProgress({
              bytesDownloaded: data.bytesDownloaded ?? 0,
              bytesTotal: data.bytesTotal ?? 1,
              progress: data.progress ?? 0,
            });
          }
        },
      );
    }

    try {
      const { filePath } = await mod.downloadModel(
        config.url,
        config.destinationPath,
        config.expectedSizeBytes ?? 0,
      );
      return filePath;
    } finally {
      subscription?.remove();
    }
  },

  /**
   * Cancel an in-progress model download.
   */
  async cancelDownload(): Promise<void> {
    const mod = assertModule();
    await mod.cancelDownload();
  },

  /**
   * Get the list of built-in models the user can choose from.
   * Filter by available RAM and backend automatically.
   */
  async getAvailableModels(ramGB?: number): Promise<LiteRTLMModelInfo[]> {
    const deviceRam = ramGB ?? 4;
    return BUILTIN_MODELS.filter((m) => {
      // Filter out models that need more RAM than available
      if (m.id.includes('q4')) return deviceRam >= 3;
      if (m.id.includes('cpu')) return deviceRam >= 4;
      return deviceRam >= 8;
    });
  },

  /**
   * Get the best embedding model URL for the current device.
   *
   * Auto-detects SoC vendor (Qualcomm, MediaTek, Exynos, Tensor) and
   * picks the NPU-optimized variant if available, falling back to the
   * generic model that works on ALL devices.
   *
   * @param socHint – optional SoC identifier (auto-detected from device if omitted).
   */
  async selectBestEmbeddingModel(socHint?: string): Promise<LiteRTLMModelInfo> {
    const soc = (socHint ?? await getDeviceSoC()).toLowerCase();
    const seqLen = 512;

    // NPU-optimized variants
    const npuMap: Record<string, string> = {
      // Qualcomm Snapdragon
      sm8550: 'qualcomm.sm8550',  // SD 8 Gen 2
      sm8650: 'qualcomm.sm8650',  // SD 8 Gen 3
      sm8750: 'qualcomm.sm8750',  // SD 8 Gen 4
      sm8850: 'qualcomm.sm8850',  // SD 8 Gen 5
      // MediaTek Dimensity
      mt6991: 'mediatek.mt6991',
      mt6993: 'mediatek.mt6993',
      // Google Tensor
      g5: 'google.tensor_g5',
    };

    for (const [chip, variant] of Object.entries(npuMap)) {
      if (soc.includes(chip)) {
        return {
          name: `EmbeddingGemma 300M (${chip.toUpperCase()} NPU)`,
          id: `embeddinggemma-300m-${chip}`,
          url: `https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq${seqLen}_mixed-precision.${variant}.tflite`,
          sizeBytes: 150_000_000,
          maxTokens: seqLen,
          backend: 'npu',
          description: `NPU-optimized for ${chip}. Auto-detected for best performance.`,
        };
      }
    }

    // Fallback: generic model (works on ALL devices — Qualcomm, MediaTek,
    // Exynos, Tensor, HiSilicon, UNISOC, and any other Android SoC)
    return BUILTIN_MODELS.find((m) => m.id === 'embeddinggemma-300m')!;
  },

  /**
   * Select the best LLM automatically based on device RAM and backends.
   */
  async selectBestModel(ramGB?: number): Promise<LiteRTLMModelInfo> {
    const models = await LiteRTLM.getAvailableModels(ramGB);
    const deviceInfo = await LiteRTLM.isSupported();

    // Prefer NPU if available
    if (deviceInfo.availableBackends.includes('npu')) {
      const npu = models.find((m) => m.id.includes('npu'));
      if (npu) return npu;
    }

    // Prefer GPU
    if (deviceInfo.availableBackends.includes('gpu')) {
      const gpu = models.find((m) => m.id.includes('gpu') || !m.id.includes('cpu'));
      if (gpu) return gpu;
    }

    // Fallback to CPU
    const cpu = models.find((m) => m.id.includes('cpu'));
    if (cpu) return cpu;

    // Last resort: first available
    return models[0] ?? BUILTIN_MODELS[0];
  },

  // ── Device Support ─────────────────────────────────────────────────────────

  /**
   * Check whether LiteRT-LM is supported on the current device.
   */
  async isSupported(): Promise<LiteRTLMDeviceSupport> {
    const mod = assertModule();
    return mod.isSupported();
  },

  // ── Load Model ─────────────────────────────────────────────────────────────

  /**
   * Load a LiteRT-LM model into memory.
   *
   * After loading, automatically runs a quick warm-up inference
   * so the first real generation is faster.
   *
   * @param config – model path, optional max tokens, and preferred backend.
   * @returns A `LiteRTLMModelHandle` to use for generation.
   */
  async loadModel(config: LiteRTLMModelConfig): Promise<LiteRTLMModelHandle> {
    const mod = assertModule();

    if (!config.modelPath || config.modelPath.trim().length === 0) {
      throw makeError('modelPath is required.', ERR.INVALID_PROMPT);
    }

    // Release previous model if still loaded
    if (_modelHandle !== null) {
      await mod.releaseModel(_modelHandle);
      _modelHandle = null;
    }

    const maxTokens = config.maxTokens ?? 512;
    const backend = config.preferredBackend ?? 'auto';

    const { handle, loadTimeMs } = await mod.loadModel(
      config.modelPath,
      maxTokens,
      backend,
    );

    _modelHandle = handle;
    _loadTimeMs = loadTimeMs;

    const modelHandleObj = new LiteRTLMModelHandle(handle);

    // ── Warm-up ──────────────────────────────────────────────────────────────
    // Run a tiny generation immediately so GPU/caches are primed.
    try {
      await mod.warmUp(handle);
    } catch {
      // Warm-up failure is non-fatal
    }

    return modelHandleObj;
  },

  // ── Embeddings (RAG / Personalization) ──────────────────────────────────────

  /**
   * Generate a text embedding vector using an embedding model (e.g. EmbeddingGemma).
   *
   * Embeddings turn text into a float vector that captures meaning.
   * Use them for:
   *   - **Personalization**: find similar past transactions to understand user context
   *   - **RAG**: retrieve relevant user data before AI generation
   *   - **Categorization**: classify transactions by semantic similarity
   *   - **Search**: find similar transactions, notes, or reminders
   *
   * Recommended model: EmbeddingGemma 300M (768-dim, .tflite)
   * Download: https://huggingface.co/litert-community/embeddinggemma-300m
   *
   * @param modelPath – path to the .tflite embedding model file.
   * @param text      – input text to embed (e.g. a transaction description).
   * @param maxSeqLen – sequence length: 256 (fastest), 512 (balanced), 1024, 2048.
   * @returns A float array (embedding vector) and timing.
   *
   * @example
   * ```ts
   * const { embedding } = await LiteRTLM.generateEmbedding(
   *   "/data/user/0/com.app/files/embeddinggemma.tflite",
   *   "bazar theke chal 500 taka kinlam",
   *   512,
   * );
   * console.log(`Embedding dim: ${embedding.length}`); // 768
   * ```
   */
  async generateEmbedding(
    modelPath: string,
    text: string,
    maxSeqLen: number = 512,
  ): Promise<LiteRTLMEmbeddingResult> {
    const mod = assertModule();
    return mod.generateEmbedding(modelPath, text, maxSeqLen);
  },

  // ── Convenience: One-shot ──────────────────────────────────────────────────

  /**
   * Load a model, generate text, and release — all in one call.
   */
  async once(
    modelConfig: LiteRTLMModelConfig,
    generationConfig: LiteRTLMGenerationConfig,
  ): Promise<LiteRTLMGenerationResult> {
    const model = await LiteRTLM.loadModel(modelConfig);
    try {
      const result = await model.generate(generationConfig);
      return result;
    } finally {
      await model.release();
    }
  },

  // ── Release All ────────────────────────────────────────────────────────────

  /**
   * Release the currently loaded model (if any) and free native memory.
   */
  async releaseAll(): Promise<void> {
    if (_modelHandle !== null) {
      const mod = assertModule();
      await mod.releaseModel(_modelHandle);
      _modelHandle = null;
    }
  },
};

// ─── Model Handle ─────────────────────────────────────────────────────────────

/**
 * A loaded model instance returned by `LiteRTLM.loadModel()`.
 *
 * Use this to generate text, stream tokens, and release memory.
 */
export class LiteRTLMModelHandle {
  /** @internal */
  private handle: number;

  /** @internal */
  constructor(handle: number) {
    this.handle = handle;
  }

  // ── Non‑streaming generation ───────────────────────────────────────────────

  /**
   * Generate text from a prompt (non-streaming).
   *
   * @returns The full generated text and performance metadata.
   */
  async generate(
    config: LiteRTLMGenerationConfig,
  ): Promise<LiteRTLMGenerationResult> {
    const mod = assertModule();

    if (_modelHandle !== this.handle) {
      throw makeError(ERR.MODEL_NOT_LOADED_MSG, ERR.MODEL_NOT_LOADED);
    }

    if (!config.prompt || config.prompt.trim().length === 0) {
      throw makeError('prompt is required.', ERR.INVALID_PROMPT);
    }

    try {
      const result = await mod.generate(
        this.handle,
        config.prompt,
        serializeConfig(config),
      );
      return result;
    } catch (raw: any) {
      // Re-throw structured errors from native side
      if (raw?.code) throw raw;
      throw makeError(
        raw?.message ?? 'Generation failed',
        ERR.GENERATION_FAILED,
      );
    }
  }

  // ── Streaming generation ───────────────────────────────────────────────────

  /**
   * Generate text with streaming token output.
   *
   * Tokens are delivered to `handlers.onToken` as they are produced.
   * Returns a subscription that can be used to cancel generation.
   */
  generateStream(
    config: LiteRTLMGenerationConfig,
    handlers: LiteRTLMStreamHandlers,
  ): LiteRTLMStreamSubscription {
    const mod = assertModule();
    let cancelled = false;

    if (_modelHandle !== this.handle) {
      const err = makeError(ERR.MODEL_NOT_LOADED_MSG, ERR.MODEL_NOT_LOADED);
      handlers.onError?.(err);
      throw err;
    }

    if (!config.prompt || config.prompt.trim().length === 0) {
      const err = makeError('prompt is required.', ERR.INVALID_PROMPT);
      handlers.onError?.(err);
      throw err;
    }

    // Subscribe to native events
    const subscription = (mod as any).addEventEmitter
      ? (mod as any).addEventEmitter(
          (eventName: string, data: any) => {
            if (cancelled) return;

            if (eventName === EVENT_ON_TOKEN) {
              handlers.onToken?.(data.token ?? '');
            } else if (eventName === EVENT_ON_COMPLETE) {
              handlers.onComplete?.({
                text: data.text ?? '',
                tokenCount: data.tokenCount ?? 0,
                timeMs: data.timeMs ?? 0,
                tokensPerSecond: data.tokensPerSecond ?? 0,
              });
            } else if (eventName === EVENT_ON_ERROR) {
              handlers.onError?.({
                message: data.message ?? 'Unknown error',
                code: data.code ?? ERR.GENERATION_FAILED,
              });
            }
          },
        )
      : null;

    // Start generation on the native side
    mod
      .startStreaming(this.handle, config.prompt, serializeConfig(config))
      .catch((err: any) => {
        handlers.onError?.({
          message: err?.message ?? 'Failed to start streaming',
          code: err?.code ?? ERR.GENERATION_FAILED,
        });
      });

    return {
      cancel: () => {
        cancelled = true;
        mod.cancelStreaming(this.handle).catch(() => {});
        subscription?.remove();
      },
    };
  }

  // ── Count Tokens ───────────────────────────────────────────────────────────

  /**
   * Count how many tokens a given text contains.
   *
   * Useful for checking prompt length before generation.
   *
   * @param text – input text to tokenize.
   * @returns The number of tokens.
   */
  async countTokens(text: string): Promise<number> {
    const mod = assertModule();
    if (_modelHandle !== this.handle) {
      throw makeError(ERR.MODEL_NOT_LOADED_MSG, ERR.MODEL_NOT_LOADED);
    }
    const { tokenCount } = await mod.countTokens(this.handle, text);
    return tokenCount;
  }

  // ── Release ────────────────────────────────────────────────────────────────

  /**
   * Release the model and free native memory.
   * After calling this, the handle is no longer usable.
   */
  async release(): Promise<void> {
    if (_modelHandle !== this.handle) return;
    const mod = assertModule();
    await mod.releaseModel(this.handle);
    _modelHandle = null;
  }
}

// ─── SoC Detection ────────────────────────────────────────────────────────────

/**
 * Detect the device SoC (System-on-Chip) identifier.
 *
 * Reads `ro.soc.model` or `ro.board.platform` from the Android system
 * properties via the native module. Falls back to an empty string.
 */
async function getDeviceSoC(): Promise<string> {
  try {
    const mod = assertModule();
    // The native module can read the SoC model from Build.SOC_MODEL
    const info = await mod.isSupported();
    return (info as any).socModel ?? '';
  } catch {
    return '';
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Serialise generation config to a JSON string for the native bridge. */
function serializeConfig(config: LiteRTLMGenerationConfig): string {
  return JSON.stringify({
    temperature: config.temperature ?? 0.0,
    topK: config.topK ?? 40,
    topP: config.topP ?? 0.9,
    maxOutputTokens: config.maxOutputTokens ?? 256,
    stopSequences: config.stopSequences ?? [],
  });
}
