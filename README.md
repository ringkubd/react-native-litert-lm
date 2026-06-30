# react-native-litert-lm

**On-device LLM inference for React Native & Expo (Android)**

Run large language models locally on Android devices using Google LiteRT-LM (AI Edge). Fully offline — no network calls, no data leaves the device.

```
npm install react-native-litert-lm
```

---

## Why this package?

| Feature | react-native-litert-lm |
|---------|------------------------|
| 🟢 Fully offline | All inference runs on-device. No internet required. |
| 🟢 Google LiteRT-LM | Uses Google's production-grade LiteRT-LM runtime (the evolution of TensorFlow Lite). |
| 🟢 GPU acceleration | Auto-detects GPU delegates (Adreno, Mali, etc.) |
| 🟢 NPU support | Automatically uses NPU on Snapdragon, Dimensity, Exynos, Tensor |
| 🟢 Expo support | Works with `expo prebuild` and EAS custom dev builds |
| 🟢 React Native CLI | Auto-linking via React Native autolinking |
| 🟢 TypeScript API | Fully typed with streaming support |
| 🟢 Privacy-first | Prompts never leave the device |

> ❌ **Expo Go is NOT supported.** This package requires native modules that Expo Go cannot provide. Use `npx expo prebuild` or EAS Build for custom dev builds.

---

## Installation

### Requirements

- **React Native** ≥ 0.76
- **Expo** ≥ 52 (if using Expo)
- **Android** SDK 26+ (Android 8.0+)
- **Android** arm64-v8a architecture (64-bit ARM)
- **Model file** in LiteRT-LM `.task` format (e.g., Gemma, CodeGemma)

### Expo Setup

```bash
npx expo install react-native-litert-lm
npx expo prebuild
npx expo run:android
```

Add to your `app.json` / `app.config.ts`:

```json
{
  "expo": {
    "plugins": [
      ["react-native-litert-lm", { "enableGpu": true }]
    ]
  }
}
```

### React Native CLI Setup

```bash
npm install react-native-litert-lm
cd android && ./gradlew clean && cd ..
npx react-native run-android
```

---

## Quick Start

```ts
import { LiteRTLM } from "react-native-litert-lm";

// 1. Load model
const model = await LiteRTLM.loadModel({
  modelPath: "/data/user/0/com.example/files/gemma.task",
  maxTokens: 512,
  preferredBackend: "auto",  // "cpu" | "gpu" | "npu" | "auto"
});

// 2. Generate text
const result = await model.generate({
  prompt: "What is the capital of France?",
  temperature: 0.2,
  maxOutputTokens: 128,
});

console.log(result.text);
// "Paris"
console.log(`Speed: ${result.tokensPerSecond.toFixed(1)} tok/s`);

// 3. Release memory
await model.release();
```

---

## Streaming API

```ts
const model = await LiteRTLM.loadModel({ modelPath: "/path/to/model.task" });

const sub = model.generateStream(
  { prompt: "Tell me a story.", temperature: 0.7, maxOutputTokens: 512 },
  {
    onToken: (token) => console.log(token),        // partial tokens
    onComplete: (result) => console.log("Done!", result),
    onError: (error) => console.error("Error:", error),
  },
);

// To cancel mid-generation:
sub.cancel();
```

---

## One-shot convenience

For quick extractions where you want load → generate → release in one call:

```ts
const result = await LiteRTLM.once(
  { modelPath: "/path/to/model.task", maxTokens: 512 },
  { prompt: "Extract expense JSON from: bought rice 500 taka", temperature: 0.1 },
);
```

---

## Use Case: Daily Expense App

Extract structured data from natural language for the Family Expense app:

```ts
const prompt = `
Return only valid JSON.
Extract expense data from this user text.

Schema:
{
  "type": "expense" | "income" | "transfer",
  "amount": number,
  "currency": "BDT",
  "category": string,
  "note": string,
  "date": string | null
}

Input:
"ajke bazar theke chal 500 taka, dim 120 taka kinlam"
`;

const result = await model.generate({
  prompt,
  temperature: 0.1,
  maxOutputTokens: 200,
});

// result.text: {"type":"expense","amount":620,"currency":"BDT","category":"Bazar","note":"chal 500, dim 120","date":null}
```

---

## Model File Placement

You must provide a `.task` model file. Options:

1. **Bundle with app** (recommended for small models):
   - Place the `.task` file in `android/app/src/main/assets/`
   - Copy to device storage on first launch using `expo-file-system`

2. **Download at runtime**:
   - Download the model file and save to `FileSystem.documentDirectory`
   - Pass the absolute path to `LiteRTLM.loadModel()`

3. **Model sources**:
   - [Gemma on Kaggle](https://www.kaggle.com/models/google/gemma)
   - [AI Edge Model Explorer](https://ai.google.dev/edge/litert/model-explorer)

---

## API Reference

### `LiteRTLM.isSupported()`

```ts
const info = await LiteRTLM.isSupported();
// { supported: true, androidVersion: 34, availableBackends: ["cpu", "gpu", "npu"] }
```

### `LiteRTLM.loadModel(config)`

```ts
const model = await LiteRTLM.loadModel({
  modelPath: string;           // required - absolute path to .task file
  maxTokens?: number;          // optional - default 512
  preferredBackend?: Backend;  // optional - "auto" | "cpu" | "gpu" | "npu"
});
```

### `model.generate(config)`

```ts
const result = await model.generate({
  prompt: string;              // required
  temperature?: number;        // optional - default 0.0
  topK?: number;               // optional - default 40
  topP?: number;               // optional - default 0.9
  maxOutputTokens?: number;    // optional - default 256
  stopSequences?: string[];    // optional - default []
});

// result: { text, tokenCount, timeMs, tokensPerSecond }
```

### `model.generateStream(config, handlers)`

```ts
const sub = model.generateStream(
  { prompt, temperature, maxOutputTokens },
  { onToken, onComplete, onError },
);

sub.cancel();
```

### `model.release()`

```ts
await model.release();
```

---

## Backend Selection

The package automatically selects the best available backend:

| Backend | Description | Availability |
|---------|-------------|-------------|
| `cpu` | CPU inference via XNNPACK | Always available |
| `gpu` | GPU delegate via OpenGL/Vulkan | Most Android 8+ devices |
| `npu` | NPU/HTP delegate via Qualcomm Hexagon, MediaPipe, etc. | Snapdragon 8+, Dimensity, Exynos, Tensor |
| `auto` | Auto-detect (GPU > CPU) | Default |

```ts
await LiteRTLM.loadModel({
  modelPath: "...",
  preferredBackend: "gpu",  // force GPU
});
```

---

## Performance Tips

| Tip | Why |
|-----|-----|
| Use GPU backend | 3-5x faster than CPU on Adreno 750 |
| Keep maxTokens low | Lower context = less RAM usage |
| Use temperature 0.0-0.2 | For structured extraction (JSON) |
| Use temperature 0.5-0.8 | For creative tasks |
| Release unused models | Free native memory |
| Bundle model in APK | Avoid download delays |

---

## Privacy & Security

- ✅ **100% on-device**: All inference runs locally
- ✅ **No network calls**: Prompts and responses never leave the device
- ✅ **No data collection**: This package does not collect any data
- ✅ **No analytics**: No tracking or telemetry

---

## Troubleshooting

### Native module not found

```
Error: Native module not found
```

**Fix:** You are using Expo Go which does not support custom native modules. Use `npx expo prebuild && npx expo run:android` or EAS Build.

### Expo Go not supported

```
Error: Native module not found. This package requires a custom dev build.
```

**Fix:** This package requires a custom development build because it includes native Kotlin code. Expo Go cannot load native modules.

### Model file not found

```
Error: model_not_loaded
```

**Fix:** Ensure the model file exists at the path you provide. Use `expo-file-system` to verify:
```ts
const info = await FileSystem.getInfoAsync(modelPath);
console.log(info.exists); // must be true
```

### Out of memory

**Fix:**
- Reduce `maxTokens` (try 256)
- Use CPU backend (uses less RAM)
- Use a smaller model
- Close other apps

### Slow CPU generation

**Fix:**
- Enable GPU: `preferredBackend: "gpu"`
- Reduce `maxOutputTokens`
- Reduce context size (`maxTokens`)

### GPU/NPU backend unavailable

The package falls back to CPU automatically. Check available backends:
```ts
const info = await LiteRTLM.isSupported();
console.log(info.availableBackends);
```

### Android build errors

```bash
# Clean build
cd android && ./gradlew clean && cd ..
npx expo run:android
```

### LiteRT-LM dependency errors

Ensure your Gradle configuration includes the necessary repositories:
```groovy
repositories {
    google()
    maven { url "https://storage.googleapis.com/download.tensorflow.org/tensorflow/lite" }
}
```

---

## Repository Structure

```
react-native-litert-lm/
├── android/
│   ├── build.gradle
│   └── src/main/java/com/reactnativelitertlm/
│       ├── LiteRTLMModule.kt    # Expo Module
│       ├── LiteRTLMPackage.kt   # RN Package
│       └── LiteRTLMEngine.kt    # Core engine
├── src/
│   ├── index.ts                  # Public API
│   ├── types.ts                  # TypeScript types
│   └── NativeLiteRTLM.ts         # Native bridge
├── plugin/
│   └── withLiteRTLM.ts           # Expo config plugin
├── example/
│   ├── app.json
│   ├── package.json
│   └── app/
│       └── index.tsx             # Example app
├── package.json
├── tsconfig.json
└── README.md
```

---

## Development

```bash
# Clone
git clone https://github.com/ringkubd/react-native-litert-lm.git
cd react-native-litert-lm

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run example app
cd example
npx expo prebuild
npx expo run:android
```

---

## License

MIT © [Arafat Hossain](https://github.com/ringkubd)

---

## Author

**Arafat Hossain** - [ringkubd@gmail.com](mailto:ringkubd@gmail.com)

- GitHub: [@ringkubd](https://github.com/ringkubd)
- Project: [Family Finance](https://github.com/ringkubd/family-finance)

---

## Links

- [Google LiteRT](https://ai.google.dev/edge/litert)
- [LiteRT-LM Guide](https://ai.google.dev/edge/litert/lm)
- [Expo Modules API](https://docs.expo.dev/modules/overview/)
- [Gemma Models](https://www.kaggle.com/models/google/gemma)
