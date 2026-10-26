import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import 'face_region_extractor.dart';

class RegionPrediction {
  final FaceRegion region;
  final int severityLevel; // 0=Clear .. 3=Severe (Mild/Moderate labels start at 1)
  final double confidence;
  final List<double> probabilities;

  const RegionPrediction({
    required this.region,
    required this.severityLevel,
    required this.confidence,
    required this.probabilities,
  });
}

/// Runs the real, trained region-aware ResNet-18 ensemble (5 independently
/// trained seeds, exported to ONNX — see training/utils/export_onnx.py) for
/// each face region crop, entirely on-device. No network call is made.
///
/// This is the same architecture and weights reported in the paper, run
/// without test-time augmentation (TTA is skipped on-device to keep
/// inference fast — each analysis already means 5 seeds x 5 regions = 25
/// forward passes).
class OnnxInferenceService {
  static const _seeds = [42, 43, 44, 45, 46];
  final List<OrtSession> _sessions = [];
  bool _ready = false;

  Future<void> ensureLoaded() async {
    if (_ready) return;
    OrtEnv.instance.init();
    final options = OrtSessionOptions();
    for (final seed in _seeds) {
      final bytes = await rootBundle.load('assets/models/region_model_seed$seed.onnx');
      final session = OrtSession.fromBuffer(bytes.buffer.asUint8List(), options);
      _sessions.add(session);
    }
    _ready = true;
  }

  Float32List _toChwFloat(img.Image image) {
    final data = Float32List(3 * 224 * 224);
    var i = 0;
    // CHW layout: all R, then all G, then all B — matches torchvision's
    // ToTensor() output layout for a [C, H, W] tensor.
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < 224; y++) {
        for (var x = 0; x < 224; x++) {
          final p = image.getPixel(x, y);
          final v = c == 0 ? p.r : (c == 1 ? p.g : p.b);
          data[i++] = v / 255.0;
        }
      }
    }
    return data;
  }

  Future<RegionPrediction> predict(ExtractedRegion extracted) async {
    await ensureLoaded();
    final chw = _toChwFloat(extracted.crop);
    final regionIdx = Int64List.fromList([extracted.region.modelIndex]);

    final probsSum = List<double>.filled(4, 0.0);
    final runOptions = OrtRunOptions();

    for (final session in _sessions) {
      final imageTensor = OrtValueTensor.createTensorWithDataList(chw, [1, 3, 224, 224]);
      final regionTensor = OrtValueTensor.createTensorWithDataList(regionIdx, [1]);
      final outputs = await session.runAsync(runOptions, {
        'image': imageTensor,
        'region_idx': regionTensor,
      });
      final raw = outputs?[0]?.value;
      imageTensor.release();
      regionTensor.release();
      outputs?.forEach((o) => o?.release());

      final row = (raw as List)[0] as List;
      for (var k = 0; k < 4; k++) {
        probsSum[k] += (row[k] as num).toDouble();
      }
    }
    runOptions.release();

    final probs = probsSum.map((v) => v / _sessions.length).toList();
    var bestIdx = 0;
    for (var k = 1; k < 4; k++) {
      if (probs[k] > probs[bestIdx]) bestIdx = k;
    }

    return RegionPrediction(
      region: extracted.region,
      severityLevel: bestIdx,
      confidence: probs[bestIdx],
      probabilities: probs,
    );
  }

  void dispose() {
    for (final s in _sessions) {
      s.release();
    }
    _sessions.clear();
    _ready = false;
  }
}
