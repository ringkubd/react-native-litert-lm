import { NativeModules } from 'react-native';
import type { NativeModule, NativeEventEmitter } from 'react-native';

/**
 * Low-level native bridge to the LiteRT-LM module.
 *
 * Registered by LiteRTLMPackage.kt on Android via React Native autolinking.
 * Not exported directly — use the `LiteRTLM` class from `index.ts`.
 *
 * @internal
 */
export interface NativeLiteRTLM extends NativeModule {
  // Required by React Native's NativeModule interface
  addListener?: (eventName: string) => void;
  removeListeners?: (count: number) => void;

  downloadModel(
    url: string,
    destinationPath: string,
    expectedSizeBytes: number,
  ): Promise<{ filePath: string; timeMs: number }>;
  cancelDownload(): Promise<void>;
  isSupported(): Promise<{
    supported: boolean;
    androidVersion: number;
    availableBackends: string[];
    socModel?: string;
    socManufacturer?: string;
    hardware?: string;
    reason?: string;
  }>;
  loadModel(
    modelPath: string,
    maxTokens: number,
    backend: string,
  ): Promise<{ handle: number; loadTimeMs: number }>;
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
  startStreaming(handle: number, prompt: string, systemPrompt: string, config: string): Promise<void>;
  cancelStreaming(handle: number): Promise<void>;
  releaseModel(handle: number): Promise<void>;
  getPerfStats(handle: number): Promise<{
    loadTimeMs: number;
    totalTokensGenerated: number;
    totalGenerationTimeMs: number;
    warmedUp: boolean;
  }>;
  warmUp(handle: number): Promise<void>;
  countTokens(handle: number, text: string): Promise<{ tokenCount: number }>;
  generateEmbedding(
    modelPath: string,
    text: string,
    maxSeqLen: number,
  ): Promise<{ embedding: number[]; timeMs: number }>;
  getLoadedModels(): Promise<Array<{ handle: number; modelPath: string; backend: string }>>;
}

// On Android, the module is auto-linked by React Native.
// On iOS / other platforms this will be null.
const NativeLiteRTLMModule: NativeLiteRTLM | null =
  (NativeModules as any).LiteRTLM ?? null;

if (!NativeLiteRTLMModule) {
  console.warn(
    '[react-native-litert-lm] Native module not found. ' +
      'Ensure you are using a custom dev build (not Expo Go).',
  );
}

export { NativeLiteRTLMModule as NativeModule };
