/**
 * Expo Config Plugin for react-native-litert-lm.
 *
 * This file is auto-detected by Expo CLI when the package is listed
 * in the `expo.plugins` array of app.json / app.config.js.
 *
 * It adds the required `android:extractNativeLibs="true"` attribute
 * to AndroidManifest.xml so that LiteRT-LM native .so libraries
 * are extracted at install time.
 *
 * Usage in app.json:
 *
 * ```json
 * {
 *   "expo": {
 *     "plugins": [
 *       ["react-native-litert-lm", { "enableGpu": true }]
 *     ]
 *   }
 * }
 * ```
 */

const { withAndroidManifest } = require('@expo/config-plugins');

/**
 * @type {import('@expo/config-plugins').ConfigPlugin}
 */
function withLiteRTLM(config, props = {}) {
  const { enableGpu = true } = props;

  config = withAndroidManifest(config, (manifestConfig) => {
    const manifest = manifestConfig.modResults.manifest;
    const application = manifest?.application?.[0];
    if (application) {
      application.$['android:extractNativeLibs'] = 'true';
    }
    return manifestConfig;
  });

  console.log(`[react-native-litert-lm] Plugin configured (enableGpu: ${enableGpu})`);

  return config;
}

module.exports = withLiteRTLM;
