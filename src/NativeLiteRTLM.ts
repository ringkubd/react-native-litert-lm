import type { NativeModule } from 'react-native';
import { requireNativeModule } from 'expo-modules-core';

/**
 * Low-level native bridge to the LiteRT-LM Kotlin module.
 *
 * This module is registered by LiteRTLMModule.kt on the Android side.
 * It is not exported directly — use the `LiteRTLM` class from `index.ts`.
 *
 * @internal
 */
export interface NativeLiteRTLM extends NativeModule {
  // ── Download ─────────────────────────────────────────────────────────────

  /**
   * Download a model file from a URL, with progress events.
   * Progress is emitted via the `onDownloadProgress` event.
   */
  downloadModel(
    url: string,
    destinationPath: string,
    expectedSizeBytes: number,
  ): Promise<{ filePath: string; timeMs: number }>;

  /**
   * Cancel an in-progress download.
   */
  cancelDownload(): Promise<void>;

  // ── Device Support ───────────────────────────────────────────────────────
  /**
   * Check whether LiteRT-LM is supported on this device.
   */
  isSupported(): Promise<{
    supported: boolean;
    androidVersion: number;
    availableBackends: string[];
    socModel?: string;
    socManufacturer?: string;
    hardware?: string;
    reason?: string;
  }>;

  /**
   * Load a model file into memory.
   *
   * @param modelPath – absolute path to the `.task` model file.
   * @param maxTokens – maximum context size (optional, default 512).
   * @param backend   – preferred backend string: "auto" | "cpu" | "gpu" | "npu".
   * @returns A model handle (integer ID) used in subsequent calls.
   */
  loadModel(
    modelPath: string,
    maxTokens: number,
    backend: string,
  ): Promise<{ handle: number; loadTimeMs: number }>;

  /**
   * Generate text (non-streaming).
   *
   * @param handle  – model handle from `loadModel`.
   * @param prompt  – input text.
   * @param config  – JSON-stringified generation parameters.
   * @returns The generated text and metadata.
   */
  generate(
    handle: number,
    prompt: string,
    config: string,
  ): Promise<{
    text: string;
    tokenCount: number;
    timeMs: number;
    tokensPerSecond: number;
  }>;

  /**
   * Start streaming generation. Tokens are emitted via the `onToken` event.
   *
   * @param handle – model handle from `loadModel`.
   * @param prompt – input text.
   * @param config – JSON-stringified generation parameters.
   */
  startStreaming(handle: number, prompt: string, config: string): Promise<void>;

  /**
   * Cancel an in-progress streaming generation.
   */
  cancelStreaming(handle: number): Promise<void>;

  /**
   * Unload the model and free native memory.
   *
   * @param handle – model handle from `loadModel`.
   */
  releaseModel(handle: number): Promise<void>;

  /**
   * Get performance statistics since the model was loaded.
   *
   * @param handle – model handle from `loadModel`.
   */
  getPerfStats(handle: number): Promise<{
    loadTimeMs: number;
    totalTokensGenerated: number;
    totalGenerationTimeMs: number;
    warmedUp: boolean;
  }>;

  /**
   * Run a warm-up inference on load to prime GPU/caches.
   * This makes the first real generation faster.
   *
   * @param handle – model handle from `loadModel`.
   */
  warmUp(handle: number): Promise<void>;

  /**
   * Count how many tokens a given text contains.
   * Useful for checking prompt length before generation.
   *
   * @param handle – model handle from `loadModel`.
   * @param text   – input text to tokenize.
   * @returns The token count.
   */
  countTokens(handle: number, text: string): Promise<{ tokenCount: number }>;

  /**
   * Generate a text embedding vector using an embedding model (e.g. EmbeddingGemma).
   *
   * Embedding models convert text to a float vector that can be used for
   * semantic search, personalization, and RAG.
   *
   * @param modelPath – path to the .tflite embedding model file.
   * @param text      – input text to embed.
   * @param maxSeqLen – maximum sequence length (256 / 512 / 1024 / 2048).
   * @returns Float array embedding vector and timing.
   */
  generateEmbedding(
    modelPath: string,
    text: string,
    maxSeqLen: number,
  ): Promise<{ embedding: number[]; timeMs: number }>;

  /**
   * List models that are currently loaded (for multi-model support).
   */
  getLoadedModels(): Promise<Array<{ handle: number; modelPath: string; backend: string }>>;
}

// On Android, the module is registered automatically by Expo autolinking.
// On iOS / other platforms this will be null.
let NativeModule: NativeLiteRTLM | null = null;

try {
  NativeModule = requireNativeModule('LiteRTLM') as NativeLiteRTLM;
} catch {
  console.warn(
    '[react-native-litert-lm] Native module not found. ' +
      'Ensure you are using a custom dev build (not Expo Go).',
  );
}

export { NativeModule };
