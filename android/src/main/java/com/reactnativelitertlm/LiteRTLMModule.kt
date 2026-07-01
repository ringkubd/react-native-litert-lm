package com.reactnativelitertlm

import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.facebook.react.modules.core.DeviceEventManagerModule
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * LiteRTLMModule — React Native bridge for LiteRT-LM on-device LLM inference.
 *
 * Uses React Native's native module system (not Expo Modules API) for maximum
 * compatibility with both Expo custom dev builds and React Native CLI projects.
 */
class LiteRTLMModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val TAG = "LiteRTLM"
        const val NAME = "LiteRTLM"
    }

    /** Core engine instance. */
    private val engine = LiteRTLMEngine()

    /** Coroutine scope for background work. */
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    override fun getName(): String = NAME

    // ── Event Emitter ─────────────────────────────────────────────────────────

    private fun sendEvent(eventName: String, params: WritableMap?) {
        reactApplicationContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(eventName, params)
    }

    // ── Async Functions (run on IO dispatcher) ───────────────────────────────

    @ReactMethod
    fun isSupported(promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val result = engine.isSupported()
                promise.resolve(toWritableMap(result))
            } catch (e: Exception) {
                promise.reject("IS_SUPPORTED_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun loadModel(modelPath: String, maxTokens: Int, backend: String, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val result = engine.loadModel(modelPath, maxTokens, backend)
                promise.resolve(toWritableMap(result))
            } catch (e: Exception) {
                promise.reject("LOAD_MODEL_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun generate(handle: Int, prompt: String, config: String, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val result = engine.generate(handle, prompt, config)
                promise.resolve(toWritableMap(result))
            } catch (e: Exception) {
                promise.reject("GENERATE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun startStreaming(handle: Int, prompt: String, systemPrompt: String, config: String, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                engine.startStreaming(
                    handle = handle,
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    configJson = config,
                    onToken = { token ->
                        sendEvent("onToken", Arguments.createMap().apply {
                            putString("token", token)
                        })
                    },
                    onComplete = { result ->
                        sendEvent("onComplete", toWritableMap(result))
                    },
                    onError = { code, message ->
                        sendEvent("onError", Arguments.createMap().apply {
                            putString("code", code)
                            putString("message", message)
                        })
                    },
                )
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("STREAM_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun cancelStreaming(handle: Int, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                engine.cancelStreaming(handle)
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("CANCEL_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun releaseModel(handle: Int, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                engine.releaseModel(handle)
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("RELEASE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun getPerfStats(handle: Int, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val result = engine.getPerfStats(handle)
                promise.resolve(toWritableMap(result))
            } catch (e: Exception) {
                promise.reject("PERF_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun downloadModel(url: String, destinationPath: String, expectedSizeBytes: Int, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val result = engine.downloadModel(
                    url = url,
                    destinationPath = destinationPath,
                    expectedSize = expectedSizeBytes.toLong(),
                    onProgress = { bytes, total, progress ->
                        sendEvent("onDownloadProgress", Arguments.createMap().apply {
                            putDouble("bytesDownloaded", bytes.toDouble())
                            putDouble("bytesTotal", total.toDouble())
                            putDouble("progress", progress.toDouble())
                        })
                    },
                )
                promise.resolve(toWritableMap(result))
            } catch (e: Exception) {
                promise.reject("DOWNLOAD_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun cancelDownload(promise: Promise) {
        engine.cancelDownload()
        promise.resolve(null)
    }

    @ReactMethod
    fun warmUp(handle: Int, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                engine.warmUp(handle)
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("WARMUP_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun countTokens(handle: Int, text: String, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val result = engine.countTokens(handle, text)
                promise.resolve(toWritableMap(result))
            } catch (e: Exception) {
                promise.reject("TOKEN_COUNT_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun generateEmbedding(modelPath: String, text: String, maxSeqLen: Int, promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val result = engine.generateEmbedding(modelPath, text, maxSeqLen)
                promise.resolve(toWritableMap(result))
            } catch (e: Exception) {
                promise.reject("EMBEDDING_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun getLoadedModels(promise: Promise) {
        scope.launch(Dispatchers.IO) {
            try {
                val models = engine.getLoadedModels()
                promise.resolve(toWritableArray(models))
            } catch (e: Exception) {
                promise.reject("MODELS_ERROR", e.message, e)
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private fun toWritableMap(map: Map<String, Any>): WritableMap {
        return Arguments.createMap().apply {
            for ((key, value) in map) {
                when (value) {
                    is String -> putString(key, value)
                    is Int -> putInt(key, value)
                    is Long -> putDouble(key, value.toDouble())
                    is Float -> putDouble(key, value.toDouble())
                    is Double -> putDouble(key, value)
                    is Boolean -> putBoolean(key, value)
                    is List<*> -> putArray(key, toWritableArray(value as List<Any>))
                    else -> putString(key, value.toString())
                }
            }
        }
    }

    private fun toWritableArray(list: List<Any>): WritableArray {
        return Arguments.createArray().apply {
            for (item in list) {
                when (item) {
                    is String -> pushString(item)
                    is Int -> pushInt(item)
                    is Long -> pushDouble(item.toDouble())
                    is Float -> pushDouble(item.toDouble())
                    is Double -> pushDouble(item)
                    is Boolean -> pushBoolean(item)
                    is Map<*, *> -> pushMap(toWritableMap(item as Map<String, Any>))
                    is List<*> -> pushArray(toWritableArray(item as List<Any>))
                    else -> pushString(item.toString())
                }
            }
        }
    }
}
