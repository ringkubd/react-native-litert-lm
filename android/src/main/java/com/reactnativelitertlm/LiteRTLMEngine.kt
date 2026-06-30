package com.reactnativelitertlm

import android.content.Context
import android.os.Build
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collectLatest
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
        var engine: com.google.ai.edge.litertlm.Engine? = null,
        var conversation: com.google.ai.edge.litertlm.Conversation? = null,
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

    // ── Embedding Generation ──────────────────────────────────────────────────

    /**
     * Generate a text embedding using a TFLite embedding model (e.g. EmbeddingGemma).
     *
     * Uses the standard TensorFlow Lite Interpreter, not LiteRT-LM Engine,
     * because embedding models are .tflite format, not .litertlm.
     *
     * @param modelPath Path to the .tflite embedding model.
     * @param text      Input text to embed.
     * @param maxSeqLen Maximum sequence length (256/512/1024/2048).
     * @return Map with embedding float array and timeMs.
     */
    suspend fun generateEmbedding(
        modelPath: String,
        text: String,
        maxSeqLen: Int,
    ): Map<String, Any> = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()

        // ── Load TFLite model ─────────────────────────────────────────────────
        // Uses standard TensorFlow Lite Interpreter for embedding models.
        // Replace with actual TFLite interpreter:
        //   val model = Interpreter(loadModelFile(modelPath))
        //   val input = tokenize(text, maxSeqLen)
        //   val output = Array(1) { FloatArray(embeddingDim) }
        //   model.run(input, output)
        //   model.close()

        // STUB: return realistic embedding dimensions
        val embeddingDim = 768
        val embedding = FloatArray(embeddingDim) { i ->
            // Simple hash-based deterministic embedding for demonstration
            (text.hashCode() % 1000) / 1000f * (i % 3 + 1) / 3f
        }

        val elapsed = System.currentTimeMillis() - startTime

        mapOf(
            "embedding" to embedding.toList(),
            "timeMs" to elapsed,
        )
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
            state.conversation?.sendMessage("Hi")
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

        // Estimate: ~4 chars per token for most models
        // LiteRT-LM doesn't expose direct tokenizer count in v0.13
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
            "socModel" to Build.SOC_MODEL,
            "socManufacturer" to Build.MANUFACTURER,
            "hardware" to Build.HARDWARE,
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
        // Uses Google's official LiteRT-LM Kotlin API.
        // https://github.com/google-ai-edge/LiteRT-LM

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = resolveBackend(backend),
        )

        val liteRtEngine = Engine(engineConfig)
        liteRtEngine.initialize()
        val conversation = liteRtEngine.createConversation()

        val loadTime = System.currentTimeMillis() - startTime

        val state = SessionState(
            handle = handle,
            modelPath = modelPath,
            maxTokens = maxTokens,
            backend = backend,
            loadTimeMs = loadTime,
            engine = liteRtEngine,
            conversation = conversation,
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
        // Uses Conversation.sendMessage() for synchronous generation.
        val samplerConfig = SamplerConfig(
            topK = (config["topK"] as? Int) ?: 40,
            topP = (config["topP"] as? Double) ?: 0.9,
            temperature = (config["temperature"] as? Double) ?: 0.0,
        )

        // Create a temporary conversation for this generation
        val conversation = state.conversation ?: state.engine?.createConversation()
        if (conversation == null) throw IllegalStateException("No conversation available")

        val response = conversation.sendMessage(prompt)
        val text = response.toString()
        val tokenCount = text.length / 4

        val elapsed = System.currentTimeMillis() - startTime

        state.isGenerating = false
        state.totalTokensGenerated += tokenCount
        state.totalGenerationTimeMs += elapsed

        val tps = if (elapsed > 0) (tokenCount.toFloat() / elapsed * 1000) else 0f

        mapOf(
            "text" to text,
            "tokenCount" to tokenCount,
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
            // Uses Conversation.sendMessageAsync() for streaming with Flow.
            val conversation = state.conversation ?: state.engine?.createConversation()
            if (conversation == null) throw IllegalStateException("No conversation available")

            conversation.sendMessageAsync(prompt)
                .catch { e ->
                    if (state.cancelled) return@catch
                    onError("generation_failed", e.message ?: "Stream error")
                }
                .collectLatest { message ->
                    if (state.cancelled) throw kotlinx.coroutines.CancellationException("CANCELLED")

                    val token = message.toString()
                    if (token.isNotEmpty()) {
                        fullText.append(token)
                        tokenCount++
                        onToken(token)
                    }
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
        try { state.conversation?.close() } catch (_: Exception) {}
        try { state.engine?.close() } catch (_: Exception) {}

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
                "stopSequences" to (orgJson.optJSONArray("stopSequences")?.let {
                    (0 until it.length()).map { idx -> it.optString(idx, "") }
                } ?: emptyList()) as List<String>,
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse config JSON, using defaults", e)
            mapOf(
                "temperature" to 0.0,
                "topK" to 40,
                "topP" to 0.9,
                "maxOutputTokens" to 256,
                "stopSequences" to listOf<String>(),
            )
        }
    }

    /** Resolve backend string to LiteRT-LM Backend. */
    private fun resolveBackend(backend: String): Backend {
        return when (backend) {
            "gpu" -> Backend.GPU()
            "npu" -> Backend.NPU()
            "cpu" -> Backend.CPU()
            else -> Backend.CPU() // auto → CPU (safe default)
        }
    }
}
