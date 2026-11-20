# AcneApp — native Android version (Kotlin)

Native Kotlin and Jetpack Compose version of the mobile prototype for facial acne severity assessment. It reproduces the flow and look of the Flutter app (`../acne_classifier_app`) and runs the same ensemble of five ResNet-18 models with region embedding, entirely on the device.

## Flow

1. **Home:** buttons to start an analysis and open the history.
2. **Capture:** camera (front by default, switchable to the rear one), an oval face guide, and the option to pick a photo from the gallery.
3. **Analysis:**
   - ML Kit detects the face;
   - the five anatomical crops (forehead, chin, nose, left and right cheeks) are derived from the facial contours;
   - each crop is classified by the five ONNX models, and their probabilities are averaged.
4. **Result:**
   - overall grade, given by the most affected region, on the Clear/Mild/Moderate/Severe scale;
   - grade and confidence for each region;
   - general recommendations and a notice that the app does not replace a dermatologist;
   - sharing.
5. **History and About:** analyses are stored only on the device, with a private copy of the photo.

No image or result is sent to a server.

## Model

| Item | Value |
|---|---|
| Files | `../mobile_models/region_model_seed{42..46}.onnx` (packaged as assets by Gradle, with no duplicate copy) |
| Inputs | `image` float32 `[N, 3, 224, 224]`, RGB in 0..1 (no mean/std normalization, matching the training `ToTensor`); `region_idx` int64 `[N]` |
| Output | `probs` float32 `[N, 4]` (softmax already applied in the graph) |
| Region indices | `0=forehead`, `1=chin`, `2=nose`, `3=left_cheek`, `4=right_cheek` |

The five crops of a photo are sent to each model as a single batch. The ONNX sessions are loaded once and reused in later analyses.

## Structure

```
app/src/main/java/br/edu/ifce/acneapp/
  MainActivity.kt                 Navigation (tabs, capture, analysis, result)
  ml/FaceRegion.kt                Regions and types
  ml/FaceRegionExtractor.kt       ML Kit + geometry of the five crops
  ml/OnnxClassifier.kt            ONNX Runtime ensemble
  data/AnalysisRecord.kt          History data model
  data/HistoryRepository.kt       Local persistence (SharedPreferences + private photos)
  data/Recommendations.kt         Recommendation texts
  ui/theme/Theme.kt               Colors and typography
  ui/components/Components.kt     Card, top bar, severity scale
  ui/screens/                     Screens
```

## Requirements

- Android Studio Ladybug (2024.2) or later, with JDK 17
- Android SDK 35
- Device or emulator running Android 8.0 (API 26) or later

## Building

The project needs the `mobile_models/` folder at the repository root, with the five `.onnx` files.

**With Android Studio:** open the `acne_classifier_android/` folder, wait for the Gradle sync and run the app (Run ▶).

**From the command line** (requires JDK 17 and Android SDK 35; the SDK path goes in `local.properties`, e.g. `sdk.dir=C:/Users/<user>/AppData/Local/Android/Sdk`):

```bash
cd acne_classifier_android
./gradlew assembleDebug        # Linux/macOS
gradlew.bat assembleDebug      # Windows
```

The APK is written to `app/build/outputs/apk/debug/app-debug.apk`. The `.github/workflows/android-kotlin.yml` workflow builds the same APK on GitHub Actions on every push that changes this folder and publishes it as an artifact.

## Timing

The analysis stages are logged to logcat with the `BENCH` tag:

- `model_load_ms`
- `face_extract_ms`
- `inference_ms`
- `total_pipeline_ms`

```bash
adb logcat -s BENCH
```

## Differences from the Flutter app

- **Crop resizing:** bilinear, like the torchvision `Resize` used in training. Flutter used nearest neighbor.
- **Model loading:** done once per app process, not on every analysis.
- **Inference:** each model receives the five crops in a single batch (5 runs per analysis instead of 25).
- **Photos:** the analyzed photo is copied to the app's private storage, so the history keeps showing the image.

## Disclaimer

Research prototype. The results measure agreement with the dataset labels and do not replace an assessment by a dermatologist.
