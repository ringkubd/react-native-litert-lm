package com.reactnativelitertlm

import android.util.Log
import expo.modules.kotlin.Promise
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.launch

/**
 * LiteRTLMModule — Expo Module that bridges LiteRT-LM to JavaScript.
 *
 * This module is auto-linked by Expo when the package is installed.
 * It exposes async functions that run inference on a background thread
 * and emit events for streaming token output.
 */
class LiteRTLMModule : Module() {

    companion object {
        private const val TAG = "LiteRTLM"
        private const val MODULE_NAME = "LiteRTLM"
    }

    /** Core engine that wraps LiteRT-LM native calls. */
    private val engine = LiteRTLMEngine()

    // ── Module Definition ─────────────────────────────────────────────────────

    override fun definition() = ModuleDefinition {

        Name(MODULE_NAME)

        // ── Events ────────────────────────────────────────────────────────────
        // Used by streaming generation to push tokens to JS.

        Events("onToken", "onComplete", "onError", "onDownloadProgress")

        // ── Async Functions ───────────────────────────────────────────────────

        /**
         * Check device support.
         */
        AsyncFunction("isSupported") {
            engine.isSupported()
        }

        /**
         * Load a .task model file.
         *
         * @param modelPath Absolute path to the model file.
         * @param maxTokens Maximum context tokens.
         * @param backend   Preferred backend: "auto", "cpu", "gpu", "npu".
         * @return { handle: number, loadTimeMs: number }
         */
        AsyncFunction("loadModel") { modelPath: String, maxTokens: Int, backend: String ->
            engine.loadModel(modelPath, maxTokens, backend)
        }

        /**
         * Generate text (non-streaming).
         *
         * @param handle    Model handle from loadModel.
         * @param prompt    Input text.
         * @param config    JSON string of generation params.
         * @return { text, tokenCount, timeMs, tokensPerSecond }
         */
        AsyncFunction("generate") { handle: Int, prompt: String, config: String ->
            engine.generate(handle, prompt, config)
        }

        /**
         * Start streaming generation.
         *
         * Tokens are dispatched via the "onToken" event.
         * On completion, "onComplete" fires with the full result.
         * On error, "onError" fires with the error details.
         *
         * @param handle    Model handle.
         * @param prompt    Input text.
         * @param config    JSON string of generation params.
         */
        AsyncFunction("startStreaming") { handle: Int, prompt: String, config: String ->
            engine.startStreaming(
                handle = handle,
                prompt = prompt,
                configJson = config,
                onToken = { token ->
                    sendEvent("onToken", mapOf("token" to token))
                },
                onComplete = { result ->
                    sendEvent("onComplete", result)
                },
                onError = { code, message ->
                    sendEvent("onError", mapOf("code" to code, "message" to message))
                },
            )
        }

        /**
         * Download a model file from a URL with progress events.
         */
        AsyncFunction("downloadModel") { url: String, destinationPath: String, expectedSizeBytes: Int ->
            engine.downloadModel(
                url = url,
                destinationPath = destinationPath,
                expectedSize = expectedSizeBytes.toLong(),
                onProgress = { bytes, total, progress ->
                    sendEvent("onDownloadProgress", mapOf(
                        "bytesDownloaded" to bytes,
                        "bytesTotal" to total,
                        "progress" to progress,
                    ))
                },
            )
        }

        /**
         * Cancel model download.
         */
        AsyncFunction("cancelDownload") {
            engine.cancelDownload()
        }

        /**
         * Run warm-up inference to prime GPU/caches.
         */
        AsyncFunction("warmUp") { handle: Int ->
            engine.warmUp(handle)
        }

        /**
         * Count tokens in text.
         */
        AsyncFunction("countTokens") { handle: Int, text: String ->
            engine.countTokens(handle, text)
        }

        /**
         * List loaded models.
         */
        AsyncFunction("getLoadedModels") {
            engine.getLoadedModels()
        }

        /**
         * Cancel streaming generation.
         */
        AsyncFunction("cancelStreaming") { handle: Int ->
            engine.cancelStreaming(handle)
        }

        /**
         * Release model from memory.
         */
        AsyncFunction("releaseModel") { handle: Int ->
            engine.releaseModel(handle)
        }

        /**
         * Get performance stats for a loaded model.
         */
        AsyncFunction("getPerfStats") { handle: Int ->
            engine.getPerfStats(handle)
        }
    }
}
