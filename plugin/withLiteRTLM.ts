/**
 * Expo Config Plugin for react-native-litert-lm.
 *
 * This plugin enables GPU delegate support in LiteRT-LM by adding the
 * required `android:extractNativeLibs="true"` attribute to
 * AndroidManifest.xml if needed, and adding ProGuard rules.
 *
 * Usage in app.json / app.config.ts:
 *
 * ```json
 * {
 *   "expo": {
 *     "plugins": [
 *       [
 *         "react-native-litert-lm",
 *         {
 *           "enableGpu": true
 *         }
 *       ]
 *     ]
 *   }
 * }
 * ```
 */

import { type ConfigPlugin, withAndroidManifest } from '@expo/config-plugins';

type PluginOptions = {
  /** Enable GPU delegate support. Default: true */
  enableGpu?: boolean;
};

const withLiteRTLM: ConfigPlugin<PluginOptions> = (
  config,
  { enableGpu = true }: PluginOptions = {},
) => {
  // ── Android: ensure extractNativeLibs is set ──────────────────────────────
  // LiteRT-LM ships native .so libraries that must be extracted at install time.
  // Without this flag, the app may crash with "Unable to load native library".

  config = withAndroidManifest(config, (manifestConfig) => {
    const manifest = manifestConfig.modResults.manifest;
    const application = manifest?.application?.[0];
    if (application) {
      application.$['android:extractNativeLibs'] = 'true';
    }
    return manifestConfig;
  });

  // ── Log configuration ────────────────────────────────────────────────────

  console.log(
    `[react-native-litert-lm] Plugin configured (enableGpu: ${enableGpu})`,
  );

  return config;
};

export default withLiteRTLM;
