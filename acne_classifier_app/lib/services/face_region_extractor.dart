import 'dart:io';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

/// The five anatomical regions the model was trained on, in the fixed order
/// the ONNX models expect as `region_idx` (0..4) — must match
/// training/run_ablation.py REGIONS exactly.
enum FaceRegion { forehead, chin, nose, leftCheek, rightCheek }

extension FaceRegionLabel on FaceRegion {
  String get label {
    switch (this) {
      case FaceRegion.forehead:
        return 'Forehead';
      case FaceRegion.chin:
        return 'Chin';
      case FaceRegion.nose:
        return 'Nose';
      case FaceRegion.leftCheek:
        return 'Left Cheek';
      case FaceRegion.rightCheek:
        return 'Right Cheek';
    }
  }

  int get modelIndex => index; // forehead=0, chin=1, nose=2, left=3, right=4
}

class ExtractedRegion {
  final FaceRegion region;
  final img.Image crop;
  const ExtractedRegion(this.region, this.crop);
}

class NoFaceDetectedException implements Exception {
  final String message;
  const NoFaceDetectedException([this.message = 'No face was detected in this photo.']);
  @override
  String toString() => message;
}

/// Finds a face with Apple's on-device Vision framework (native Swift, see
/// ios/Runner/FaceRegionDetector.swift) and derives five anatomical region
/// crops from the returned landmark anchors. This mirrors, as closely as
/// Vision's landmark set allows, the region geometry used to build the
/// training crops (training/utils/segment_faces_mediapipe.py) — proportional
/// boxes anchored on eyebrow, nose and lip positions rather than a fixed
/// grid.
///
/// Vision (not google_mlkit_face_detection) is used because, at the time of
/// writing, Google ML Kit's vision pods ship no arm64 iOS Simulator slice,
/// which made the app unbuildable in the Simulator on Apple Silicon. Vision
/// is part of iOS itself, so this gap does not exist.
class FaceRegionExtractor {
  static const _channel = MethodChannel('acne_app/face_region_detector');

  Future<List<ExtractedRegion>> extract(String imagePath) async {
    final Map<Object?, Object?>? raw;
    try {
      raw = await _channel.invokeMethod<Map<Object?, Object?>>('detectFace', {'path': imagePath});
    } on PlatformException catch (e) {
      throw NoFaceDetectedException(e.message ?? 'Face detection failed.');
    }
    if (raw == null) {
      throw const NoFaceDetectedException();
    }
    final anchors = raw.map((k, v) => MapEntry(k as String, (v as num).toDouble()));

    final bytes = await File(imagePath).readAsBytes();
    var decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw const NoFaceDetectedException('Could not decode this image.');
    }
    decoded = img.bakeOrientation(decoded);

    final boxes = _regionBoxes(anchors, decoded.width, decoded.height);
    final crops = <ExtractedRegion>[];
    for (final entry in boxes.entries) {
      final r = entry.value;
      if (r.width < 8 || r.height < 8) continue;
      final cropped = img.copyCrop(
        decoded,
        x: r.left.round(),
        y: r.top.round(),
        width: r.width.round(),
        height: r.height.round(),
      );
      final resized = img.copyResize(cropped, width: 224, height: 224);
      crops.add(ExtractedRegion(entry.key, resized));
    }
    return crops;
  }

  Map<FaceRegion, Rect> _regionBoxes(Map<String, double> a, int imgWidth, int imgHeight) {
    final faceLeft = a['faceLeft']!;
    final faceTop = a['faceTop']!;
    final faceW = a['faceWidth']!;
    final faceH = a['faceHeight']!;
    final faceRight = faceLeft + faceW;
    final faceBottom = faceTop + faceH;

    final browY = a['browY'] ?? (faceTop + faceH * 0.36);
    final noseCenterX = a['noseCenterX'] ?? (faceLeft + faceW / 2);
    final noseTopY = (a['noseTopY'] ?? (faceTop + faceH * 0.38)) - faceH * 0.02;
    final noseBottomY = (a['noseBottomY'] ?? (faceTop + faceH * 0.62)) + faceH * 0.03;
    final lipBottomY = a['lipBottomY'] ?? (faceTop + faceH * 0.82);

    Rect clamp(Rect r) => Rect.fromLTRB(
          r.left.clamp(0, imgWidth.toDouble()),
          r.top.clamp(0, imgHeight.toDouble()),
          r.right.clamp(0, imgWidth.toDouble()),
          r.bottom.clamp(0, imgHeight.toDouble()),
        );

    final forehead = clamp(Rect.fromLTRB(
      faceLeft + faceW * 0.14,
      faceTop - faceH * 0.04,
      faceRight - faceW * 0.14,
      browY - faceH * 0.02,
    ));

    final chin = clamp(Rect.fromLTRB(
      faceLeft + faceW * 0.3,
      lipBottomY + faceH * 0.02,
      faceRight - faceW * 0.3,
      faceBottom + faceH * 0.05,
    ));

    final nose = clamp(Rect.fromLTRB(
      noseCenterX - faceW * 0.14,
      noseTopY,
      noseCenterX + faceW * 0.14,
      noseBottomY,
    ));

    final leftCheek = clamp(Rect.fromCenter(
      center: Offset(faceLeft + faceW * 0.22, faceTop + faceH * 0.6),
      width: faceW * 0.3,
      height: faceH * 0.3,
    ));

    final rightCheek = clamp(Rect.fromCenter(
      center: Offset(faceRight - faceW * 0.22, faceTop + faceH * 0.6),
      width: faceW * 0.3,
      height: faceH * 0.3,
    ));

    return {
      FaceRegion.forehead: forehead,
      FaceRegion.chin: chin,
      FaceRegion.nose: nose,
      FaceRegion.leftCheek: leftCheek,
      FaceRegion.rightCheek: rightCheek,
    };
  }

  void dispose() {}
}
