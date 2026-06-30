/**
 * LiteRT-LM Example App — Expo Router
 *
 * Demonstrates:
 *   1. Device support check
 *   2. Model file selection & loading
 *   3. Text generation (non-streaming)
 *   4. Streaming generation
 *   5. Cancel generation
 *   6. Performance stats
 *   7. Model release
 */

import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  Pressable,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from 'react-native';
import * as FileSystem from 'expo-file-system';
import {
  LiteRTLM,
  type LiteRTLMDeviceSupport,
  type LiteRTLMGenerationResult,
  type LiteRTLMPerfStats,
} from 'react-native-litert-lm';

type Screen = 'home' | 'loaded' | 'generating' | 'result';

export default function ExampleScreen() {
  const [screen, setScreen] = useState<Screen>('home');
  const [deviceInfo, setDeviceInfo] = useState<LiteRTLMDeviceSupport | null>(null);
  const [modelPath, setModelPath] = useState('');
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState<LiteRTLMGenerationResult | null>(null);
  const [streamText, setStreamText] = useState('');
  const [perfStats, setPerfStats] = useState<LiteRTLMPerfStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const streamRef = useRef<string>('');

  // ── Check device support ─────────────────────────────────────────────────

  const checkSupport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const info = await LiteRTLM.isSupported();
      setDeviceInfo(info);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to check device support');
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Pick model file ──────────────────────────────────────────────────────

  const pickModel = useCallback(async () => {
    // In production, use expo-document-picker or pre-bundle the model.
    // For this example, enter the path manually.
    setModelPath('');
    setError(null);

    const defaultPath = `${FileSystem.documentDirectory}gemma.task`;
    setModelPath(defaultPath);
  }, []);

  // ── Load model ───────────────────────────────────────────────────────────

  const loadModel = useCallback(async () => {
    if (!modelPath.trim()) {
      setError('Please enter a model file path.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await LiteRTLM.loadModel({
        modelPath: modelPath.trim(),
        maxTokens: 512,
        preferredBackend: 'auto',
      });
      setScreen('loaded');
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load model');
    } finally {
      setLoading(false);
    }
  }, [modelPath]);

  // ── Generate (non-streaming) ─────────────────────────────────────────────

  const generate = useCallback(async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await LiteRTLM.once(
        { modelPath: modelPath.trim(), maxTokens: 512 },
        { prompt: prompt.trim(), temperature: 0.1, maxOutputTokens: 256 },
      );
      setResult(res);
      setScreen('result');
    } catch (e: any) {
      setError(e?.message ?? 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [modelPath, prompt]);

  // ── Stream ───────────────────────────────────────────────────────────────

  const startStream = useCallback(async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt.');
      return;
    }

    setLoading(true);
    setError(null);
    setStreamText('');
    streamRef.current = '';
    setScreen('generating');

    try {
      const model = await LiteRTLM.loadModel({
        modelPath: modelPath.trim(),
        maxTokens: 512,
      });

      const sub = model.generateStream(
        { prompt: prompt.trim(), temperature: 0.1, maxOutputTokens: 256 },
        {
          onToken: (token) => {
            streamRef.current += token;
            setStreamText(streamRef.current);
          },
          onComplete: (res) => {
            setResult(res);
            setScreen('result');
            setLoading(false);
          },
          onError: (err) => {
            setError(err.message);
            setLoading(false);
          },
        },
      );

      cancelRef.current = () => sub.cancel();
    } catch (e: any) {
      setError(e?.message ?? 'Stream start failed');
      setLoading(false);
    }
  }, [modelPath, prompt]);

  // ── Cancel ───────────────────────────────────────────────────────────────

  const cancel = useCallback(() => {
    cancelRef.current?.();
    setLoading(false);
  }, []);

  // ── Get performance stats ────────────────────────────────────────────────

  const showPerfStats = useCallback(async () => {
    try {
      const stats = await (LiteRTLM as any).releaseAll();
      // Stats are internal — simplified here
    } catch {}
  }, []);

  // ── Release ──────────────────────────────────────────────────────────────

  const release = useCallback(async () => {
    setLoading(true);
    try {
      await LiteRTLM.releaseAll();
      setScreen('home');
      setResult(null);
      setStreamText('');
    } catch (e: any) {
      setError(e?.message ?? 'Release failed');
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Reset ────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    setScreen('home');
    setResult(null);
    setStreamText('');
    setError(null);
    setPrompt('');
  }, []);

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>🧠 LiteRT-LM Example</Text>
      <Text style={styles.subtitle}>On-device LLM inference for React Native</Text>

      {/* Device Support */}
      <Pressable style={styles.button} onPress={checkSupport} disabled={loading}>
        <Text style={styles.buttonText}>
          {loading ? 'Checking...' : 'Check Device Support'}
        </Text>
      </Pressable>

      {deviceInfo && (
        <View style={styles.infoBox}>
          <Text style={styles.infoText}>
            Supported: {deviceInfo.supported ? '✅ Yes' : '❌ No'}
            {'\n'}Android API: {deviceInfo.androidVersion}
            {'\n'}Backends: {deviceInfo.availableBackends.join(', ')}
            {deviceInfo.reason ? `\nReason: ${deviceInfo.reason}` : ''}
          </Text>
        </View>
      )}

      {/* Model Path */}
      <Text style={styles.label}>Model File Path</Text>
      <TextInput
        style={styles.input}
        value={modelPath}
        onChangeText={setModelPath}
        placeholder="/data/user/0/.../gemma.task"
        placeholderTextColor="#666"
        autoCapitalize="none"
      />

      {/* Load / Release */}
      <View style={styles.row}>
        <Pressable
          style={[styles.button, styles.buttonSmall]}
          onPress={loadModel}
          disabled={loading || screen === 'loaded'}
        >
          <Text style={styles.buttonText}>
            {loading ? 'Loading...' : 'Load Model'}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.button, styles.buttonSmall, styles.buttonDanger]}
          onPress={release}
          disabled={loading || screen === 'home'}
        >
          <Text style={styles.buttonText}>Release</Text>
        </Pressable>
      </View>

      {/* Prompt */}
      <Text style={styles.label}>Prompt</Text>
      <TextInput
        style={[styles.input, styles.promptInput]}
        value={prompt}
        onChangeText={setPrompt}
        multiline
        placeholder="Enter your prompt here..."
        placeholderTextColor="#666"
      />

      {/* Generate / Stream / Cancel */}
      <View style={styles.row}>
        <Pressable
          style={[styles.button, styles.buttonSmall]}
          onPress={generate}
          disabled={loading || screen === 'home'}
        >
          <Text style={styles.buttonText}>Generate</Text>
        </Pressable>
        <Pressable
          style={[styles.button, styles.buttonSmall]}
          onPress={startStream}
          disabled={loading || screen === 'home'}
        >
          <Text style={styles.buttonText}>Stream</Text>
        </Pressable>
        <Pressable
          style={[styles.button, styles.buttonSmall, styles.buttonDanger]}
          onPress={cancel}
          disabled={!loading}
        >
          <Text style={styles.buttonText}>Cancel</Text>
        </Pressable>
      </View>

      {/* Loading */}
      {loading && !streamText && (
        <ActivityIndicator color="#7C3AED" style={{ marginVertical: 16 }} />
      )}

      {/* Streaming output */}
      {streamText ? (
        <View style={styles.resultBox}>
          <Text style={styles.resultLabel}>Streaming output:</Text>
          <Text style={styles.resultText}>{streamText}</Text>
        </View>
      ) : null}

      {/* Final result */}
      {result && (
        <View style={styles.resultBox}>
          <Text style={styles.resultLabel}>Generation Result</Text>
          <Text style={styles.resultText}>{result.text}</Text>
          <Text style={styles.statsText}>
            Tokens: {result.tokenCount} | Time: {result.timeMs.toFixed(0)}ms |
            Speed: {result.tokensPerSecond.toFixed(1)} tok/s
          </Text>
        </View>
      )}

      {/* Error */}
      {error && (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>❌ {error}</Text>
        </View>
      )}

      {/* Reset */}
      <Pressable style={[styles.button, styles.buttonOutline]} onPress={reset}>
        <Text style={[styles.buttonText, { color: '#7C3AED' }]}>Reset</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f23' },
  content: { padding: 20, paddingBottom: 60 },
  title: { fontSize: 24, fontWeight: '800', color: '#FFF', marginBottom: 4 },
  subtitle: { fontSize: 13, color: '#888', marginBottom: 24 },
  label: { fontSize: 13, fontWeight: '600', color: '#AAA', marginBottom: 6, marginTop: 12 },
  input: {
    backgroundColor: '#1a1a3e',
    color: '#FFF',
    borderRadius: 10,
    padding: 14,
    fontSize: 14,
    borderWidth: 1,
    borderColor: '#333',
  },
  promptInput: { minHeight: 80, textAlignVertical: 'top' },
  row: { flexDirection: 'row', gap: 8, marginTop: 12 },
  button: {
    backgroundColor: '#7C3AED',
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 20,
    alignItems: 'center',
    marginTop: 12,
  },
  buttonSmall: { flex: 1, paddingVertical: 10, marginTop: 0 },
  buttonDanger: { backgroundColor: '#DC2626' },
  buttonOutline: { backgroundColor: 'transparent', borderWidth: 1, borderColor: '#7C3AED' },
  buttonText: { color: '#FFF', fontSize: 14, fontWeight: '600' },
  infoBox: { backgroundColor: '#1a1a3e', borderRadius: 10, padding: 14, marginTop: 8 },
  infoText: { color: '#CCC', fontSize: 13, lineHeight: 20 },
  resultBox: { backgroundColor: '#1a1a3e', borderRadius: 10, padding: 14, marginTop: 12 },
  resultLabel: { fontSize: 12, fontWeight: '600', color: '#7C3AED', marginBottom: 6 },
  resultText: { color: '#FFF', fontSize: 14, lineHeight: 20 },
  statsText: { color: '#888', fontSize: 12, marginTop: 8 },
  errorBox: { backgroundColor: '#3e1a1a', borderRadius: 10, padding: 14, marginTop: 12 },
  errorText: { color: '#FCA5A5', fontSize: 13 },
});
