package com.reactnativelitertlm

import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * LiteRTLMEngine — wraps Google LiteRT-LM native inference.
 *
 * Each loaded model gets a unique handle and runs on a background coroutine.
 * The engine is responsible for:
 *   - Loading .task model files
 *   - Running text generation (sync + streaming)
 *   - Cancelling in-progress generation
 *   - Releasing native memory
 *   - Collecting performance statistics
 *
 * @property tag         Used for logcat filtering.
 * @property modelHandle Counter for assigning unique handles.
 * @property sessions    Map of handle → SessionState for active models.
 */
class LiteRTLMEngine {

    companion object {
        private const val TAG = "LiteRTLM"
    }

    // ── Session State ─────────────────────────────────────────────────────────

    /** Internal state for a loaded model session. */
    private data class SessionState(
        val handle: Int,
        val modelPath: String,
        val maxTokens: Int,
        val backend: String,
        val loadTimeMs: Long,
        var model: Any? = null,          // LiteRT-LM model object
        var interpreter: Any? = null,     // LiteRT interpreter handle
        var isLoaded: Boolean = false,
        var isGenerating: Boolean = false,
        var cancelled: Boolean = false,
        var warmedUp: Boolean = false,
        var totalTokensGenerated: Int = 0,
        var totalGenerationTimeMs: Long = 0L,
    )

    private val modelHandle = AtomicInteger(1000)
    private val sessions = ConcurrentHashMap<Int, SessionState>()

    // ── Model Download ────────────────────────────────────────────────────────

    /**
     * Download a model file from a remote URL.
     *
     * @param url             Source URL of the .task model.
     * @param destinationPath Absolute path to save the file.
     * @param expectedSize    Expected file size (for progress).
     * @param onProgress      Called with bytes downloaded / total.
     * @return Map with filePath and timeMs.
     */
    suspend fun downloadModel(
        url: String,
        destinationPath: String,
        expectedSize: Long,
        onProgress: (Long, Long, Float) -> Unit,
    ): Map<String, Any> = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        var downloadCancelled = false
        _cancelDownload = { downloadCancelled = true }

        try {
            val connection = java.net.URL(url).openConnection() as java.net.HttpURLConnection
            connection.connectTimeout = 30_000
            connection.readTimeout = 30_000
            connection.connect()

            val totalBytes = if (expectedSize > 0) expectedSize else connection.contentLengthLong.coerceAtLeast(1)
            val inputStream = connection.inputStream
            val outputStream = java.io.FileOutputStream(destinationPath)

            val buffer = ByteArray(8192)
            var bytesRead: Int
            var totalRead = 0L

            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                if (downloadCancelled) {
                    outputStream.close()
                    inputStream.close()
                    java.io.File(destinationPath).delete()
                    throw kotlinx.coroutines.CancellationException("Download cancelled by user")
                }
                outputStream.write(buffer, 0, bytesRead)
                totalRead += bytesRead
                val progress = totalRead.toFloat() / totalBytes.toFloat()
                onProgress(totalRead, totalBytes, progress)
            }

            outputStream.close()
            inputStream.close()
            connection.disconnect()

            val elapsed = System.currentTimeMillis() - startTime

            mapOf(
                "filePath" to destinationPath,
                "timeMs" to elapsed,
            )
        } finally {
            _cancelDownload = null
        }
    }

    /** Mutable reference to the download cancel function. */
    private var _cancelDownload: (() -> Unit)? = null

    /** Cancel an in-progress download. */
    fun cancelDownload() {
        _cancelDownload?.invoke()
    }

    // ── Warm-up ───────────────────────────────────────────────────────────────

    /**
     * Run a short warm-up inference to prime GPU/caches.
     * Called automatically after loadModel().
     */
    suspend fun warmUp(handle: Int) = withContext(Dispatchers.IO) {
        val state = sessions[handle]
        if (state == null || !state.isLoaded) return@withContext

        try {
            runLiteRtLmInference(state.interpreter, "Hello", mapOf("maxOutputTokens" to 5))
            state.warmedUp = true
            Log.i(TAG, "Warm-up complete for handle=$handle")
        } catch (e: Exception) {
            Log.w(TAG, "Warm-up failed (non-fatal)", e)
        }
    }

    // ── Token Counting ────────────────────────────────────────────────────────

    /**
     * Count the number of tokens in a text string.
     * Uses the loaded model's tokenizer.
     */
    suspend fun countTokens(handle: Int, text: String): Map<String, Any> = withContext(Dispatchers.IO) {
        val state = sessions[handle]
            ?: throw IllegalStateException("Model $handle not loaded")

        // ── LiteRT-LM Tokenizer API ───────────────────────────────────────────
        // Replace with actual LiteRT-LM tokenizer call:
        //   val count = interpreter.tokenizer.countTokens(text)

        // STUB: rough estimate (4 chars per token)
        val count = (text.length / 4).coerceAtLeast(1)

        mapOf("tokenCount" to count)
    }

    // ── Device Support ────────────────────────────────────────────────────────

    /**
     * Check run-time device support.
     *
     * LiteRT-LM requires:
     *   - Android SDK 26+ (Android 8.0)
     *   - 64-bit ARM CPU (arm64-v8a)
     *
     * GPU delegate availability depends on OpenGL ES 3.1+ / Vulkan support,
     * which is detected by LiteRT internally at init time.
     */
    fun isSupported(): Map<String, Any> {
        val sdkInt = Build.VERSION.SDK_INT
        val supported = sdkInt >= 26

        val backends = mutableListOf("cpu")
        if (supported) {
            // GPU is available on virtually all modern Android devices
            backends.add("gpu")
            // NPU is device-specific — report as available for known SoCs
            if (hasNpu()) backends.add("npu")
        }

        return mapOf(
            "supported" to supported,
            "androidVersion" to sdkInt,
            "availableBackends" to backends,
            "reason" to if (supported) "" else "Android SDK $sdkInt is below minimum 26.",
        )
    }

    /** Heuristic NPU detection — checks for known SoC features. */
    private fun hasNpu(): Boolean {
        val hardware = Build.HARDWARE.lowercase()
        val soc = Build.SOC_MODEL.lowercase()
        val board = Build.BOARD.lowercase()

        return soc.contains("snapdragon") ||
               soc.contains("dimensity") ||
               soc.contains("exynos") ||
               hardware.contains("qcom") ||
               hardware.contains("mt") ||
               board.contains("tensor")
    }

    // ── Load Model ────────────────────────────────────────────────────────────

    /**
     * Load a `.task` model file from the given path.
     *
     * This method runs synchronously on a background coroutine because
     * LiteRT-LM model creation can take several seconds for large models.
     *
     * @param modelPath Absolute path to the .task file.
     * @param maxTokens Maximum sequence length.
     * @param backend   Preferred backend identifier.
     * @return Map with handle and loadTimeMs.
     */
    suspend fun loadModel(
        modelPath: String,
        maxTokens: Int,
        backend: String,
    ): Map<String, Any> = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        val handle = modelHandle.incrementAndGet()

        Log.i(TAG, "Loading model: $modelPath (maxTokens=$maxTokens, backend=$backend, handle=$handle)")

        // ── LiteRT-LM Initialisation ─────────────────────────────────────────
        // This is where the actual LiteRT-LM native API is called.
        // The following is a template that maps to the real LiteRT-LM Java/Kotlin API.
        //
        // In production, replace `createLiteRtLmInterpreter()` with:
        //   val options = LiteRtLmOptions.builder()
        //       .setModelFilePath(modelPath)
        //       .setMaxTokens(maxTokens)
        //       .setBackend(resolveBackend(backend))
        //       .build()
        //   val interpreter = LiteRtLmInterpreter.create(options)

        val interpreter = createLiteRtLmInterpreter(modelPath, maxTokens, backend)
        val loadTime = System.currentTimeMillis() - startTime

        val state = SessionState(
            handle = handle,
            modelPath = modelPath,
            maxTokens = maxTokens,
            backend = backend,
            loadTimeMs = loadTime,
            interpreter = interpreter,
            isLoaded = true,
        )

        sessions[handle] = state

        Log.i(TAG, "Model loaded in ${loadTime}ms (handle=$handle)")

        mapOf(
            "handle" to handle,
            "loadTimeMs" to loadTime,
        )
    }

    // ── Generation (non-streaming) ────────────────────────────────────────────

    /**
     * Run text generation synchronously.
     *
     * @param handle Model handle.
     * @param prompt Input text.
     * @param configJson JSON-encoded generation parameters.
     * @return Map with text, tokenCount, timeMs, tokensPerSecond.
     */
    suspend fun generate(
        handle: Int,
        prompt: String,
        configJson: String,
    ): Map<String, Any> = withContext(Dispatchers.IO) {
        val state = sessions[handle]
            ?: throw IllegalStateException("Model $handle not loaded")

        if (!state.isLoaded) throw IllegalStateException("Model $handle released")

        state.isGenerating = true
        state.cancelled = false

        val config = parseConfig(configJson)
        val startTime = System.currentTimeMillis()

        // ── LiteRT-LM Inference ───────────────────────────────────────────────
        // Replace with actual LiteRT-LM generate() call:
        //   val result = interpreter.generate(prompt, generationConfig)
        //   val text = result.text
        //   val tokens = result.tokenCount

        val result = runLiteRtLmInference(state.interpreter, prompt, config)
        val elapsed = System.currentTimeMillis() - startTime

        state.isGenerating = false
        state.totalTokensGenerated += result.tokenCount
        state.totalGenerationTimeMs += elapsed

        val tps = if (elapsed > 0) (result.tokenCount.toFloat() / elapsed * 1000) else 0f

        mapOf(
            "text" to result.text,
            "tokenCount" to result.tokenCount,
            "timeMs" to elapsed,
            "tokensPerSecond" to tps,
        )
    }

    // ── Streaming Generation ──────────────────────────────────────────────────

    /**
     * Start streaming generation. Tokens are emitted via EventEmitter.
     *
     * @param handle     Model handle.
     * @param prompt     Input text.
     * @param configJson JSON-encoded generation parameters.
     * @param onToken    Callback for each token fragment.
     * @param onComplete Callback when generation finishes.
     * @param onError    Callback on failure.
     */
    suspend fun startStreaming(
        handle: Int,
        prompt: String,
        configJson: String,
        onToken: (String) -> Unit,
        onComplete: (Map<String, Any>) -> Unit,
        onError: (String, String) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val state = sessions[handle]
            ?: run {
                onError("model_not_loaded", "Model $handle not loaded")
                return@withContext
            }

        if (!state.isLoaded) {
            onError("model_not_loaded", "Model $handle released")
            return@withContext
        }

        state.isGenerating = true
        state.cancelled = false

        val config = parseConfig(configJson)
        val startTime = System.currentTimeMillis()
        val fullText = StringBuilder()
        var tokenCount = 0

        try {
            // ── LiteRT-LM Streaming Inference ──────────────────────────────────
            // Replace with actual LiteRT-LM streaming call:
            //   interpreter.generateStream(prompt, config) { token ->
            //       if (state.cancelled) throw CancellationException()
            //       sendToken(token)
            //   }

            val tokenList = simulateTokenGeneration(state.interpreter, prompt, config) { token ->
                if (state.cancelled) throw kotlinx.coroutines.CancellationException("CANCELLED")

                fullText.append(token)
                tokenCount++
                onToken(token)
            }

            val elapsed = System.currentTimeMillis() - startTime
            state.isGenerating = false
            state.totalTokensGenerated += tokenCount
            state.totalGenerationTimeMs += elapsed

            val tps = if (elapsed > 0) (tokenCount.toFloat() / elapsed * 1000) else 0f

            onComplete(
                mapOf(
                    "text" to fullText.toString(),
                    "tokenCount" to tokenCount,
                    "timeMs" to elapsed,
                    "tokensPerSecond" to tps,
                ),
            )

        } catch (e: kotlinx.coroutines.CancellationException) {
            state.isGenerating = false
            onError("cancelled", "Generation cancelled by user")
        } catch (e: Exception) {
            state.isGenerating = false
            Log.e(TAG, "Streaming error", e)
            onError("generation_failed", e.message ?: "Unknown error")
        }
    }

    // ── Cancel ────────────────────────────────────────────────────────────────

    /**
     * Cancel an in-progress streaming generation.
     */
    suspend fun cancelStreaming(handle: Int) = withContext(Dispatchers.IO) {
        val state = sessions[handle]
        if (state != null) {
            state.cancelled = true
            Log.i(TAG, "Cancelled generation for handle=$handle")
        }
    }

    // ── Release ───────────────────────────────────────────────────────────────

    /**
     * Release native memory for a loaded model.
     */
    suspend fun releaseModel(handle: Int) = withContext(Dispatchers.IO) {
        val state = sessions.remove(handle) ?: return@withContext

        // ── LiteRT-LM Cleanup ─────────────────────────────────────────────────
        // Replace with actual cleanup:
        //   interpreter.close()

        releaseLiteRtLmInterpreter(state.interpreter)

        Log.i(TAG, "Released model handle=$handle")
    }

    // ── Performance Stats ─────────────────────────────────────────────────────

    /**
     * List all currently loaded models.
     */
    fun getLoadedModels(): List<Map<String, Any>> {
        return sessions.values.map { state ->
            mapOf(
                "handle" to state.handle,
                "modelPath" to state.modelPath,
                "backend" to state.backend,
            )
        }
    }

    /**
     * Get aggregate performance statistics for a loaded model.
     */
    suspend fun getPerfStats(handle: Int): Map<String, Any> = withContext(Dispatchers.IO) {
        val state = sessions[handle]
            ?: return@withContext mapOf(
                "loadTimeMs" to 0L,
                "totalTokensGenerated" to 0,
                "totalGenerationTimeMs" to 0L,
            )

        mapOf(
            "loadTimeMs" to state.loadTimeMs,
            "totalTokensGenerated" to state.totalTokensGenerated,
            "totalGenerationTimeMs" to state.totalGenerationTimeMs,
            "warmedUp" to state.warmedUp,
        )
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /** Parse JSON generation config into a map. */
    private fun parseConfig(json: String): Map<String, Any> {
        return try {
            val orgJson = org.json.JSONObject(json)
            mapOf(
                "temperature" to orgJson.optDouble("temperature", 0.0),
                "topK" to orgJson.optInt("topK", 40),
                "topP" to orgJson.optDouble("topP", 0.9),
                "maxOutputTokens" to orgJson.optInt("maxOutputTokens", 256),
                "stopSequences" to orgJson.optJSONArray("stopSequences")?.let {
                    (0 until it.length()).map { idx -> it.optString(idx, "") }
                } ?: emptyList<String>(),
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse config JSON, using defaults", e)
            mapOf(
                "temperature" to 0.0,
                "topK" to 40,
                "topP" to 0.9,
                "maxOutputTokens" to 256,
                "stopSequences" to emptyList<String>(),
            )
        }
    }

    /** Resolve backend string to LiteRT delegate. */
    private fun resolveBackend(backend: String): String {
        return when (backend) {
            "gpu" -> "gpu"
            "npu" -> "npu"
            "cpu" -> "cpu"
            else -> "auto"
        }
    }

    // ── LiteRT-LM Native API Stubs ────────────────────────────────────────────
    //
    // These methods are templates for the real LiteRT-LM API calls.
    // Replace with actual Google AI Edge LiteRT library calls before release.
    //
    // The production implementation will use the following pattern:
    //
    //   import com.google.ai.edge.litert.lm.LiteRtLmInterpreter
    //   import com.google.ai.edge.litert.lm.LiteRtLmOptions
    //
    //   val options = LiteRtLmOptions.builder()
    //       .setModelFilePath(modelPath)
    //       .setMaxTokens(maxTokens)
    //       .build()
    //
    //   val interpreter = LiteRtLmInterpreter.create(options)
    //   val result = interpreter.generate(prompt)
    //   interpreter.close()

    private data class InferenceResult(val text: String, val tokenCount: Int)

    private fun createLiteRtLmInterpreter(
        modelPath: String,
        maxTokens: Int,
        backend: String,
    ): Any {
        Log.i(TAG, "[LiteRT-LM] Creating interpreter: $modelPath")
        // STUB: return a placeholder object
        return Object()
    }

    private fun runLiteRtLmInference(
        interpreter: Any?,
        prompt: String,
        config: Map<String, Any>,
    ): InferenceResult {
        val maxTokens = (config["maxOutputTokens"] as? Int) ?: 256
        // STUB: simulate generation
        val text = "STUB: LiteRT-LM inference for: ${prompt.take(50)}..."
        return InferenceResult(text, text.length / 4)
    }

    private fun releaseLiteRtLmInterpreter(interpreter: Any?) {
        Log.i(TAG, "[LiteRT-LM] Releasing interpreter")
        // STUB: no-op
    }

    private fun simulateTokenGeneration(
        interpreter: Any?,
        prompt: String,
        config: Map<String, Any>,
        onToken: (String) -> Unit,
    ): List<String> {
        val tokens = listOf("Hello", " from", " LiteRT", "-LM", "!")
        tokens.forEach { onToken(it) }
        return tokens
    }
}
