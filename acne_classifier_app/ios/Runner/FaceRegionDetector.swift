import Flutter
import UIKit
import Vision

/// Native face-landmark detection using Apple's Vision framework (built into
/// iOS, no third-party dependency). Returns pixel-space anchor points the
/// Dart side uses to derive the same five anatomical region boxes that used
/// to come from ML Kit's contours — this replaces google_mlkit_face_detection,
/// which currently ships no arm64 Simulator slice and could not run here.
class FaceRegionDetector: NSObject, FlutterPlugin {
  static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(
      name: "acne_app/face_region_detector",
      binaryMessenger: registrar.messenger()
    )
    let instance = FaceRegionDetector()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "detectFace" else {
      result(FlutterMethodNotImplemented)
      return
    }
    guard let args = call.arguments as? [String: Any],
          let path = args["path"] as? String,
          let image = UIImage(contentsOfFile: path),
          let cgImage = image.cgImage
    else {
      result(FlutterError(code: "bad_image", message: "Could not load image", details: nil))
      return
    }

    let width = CGFloat(cgImage.width)
    let height = CGFloat(cgImage.height)
    let orientation = cgOrientation(image.imageOrientation)

    func payload(for face: VNFaceObservation) -> [String: CGFloat] {
      // Vision's boundingBox is normalized with origin at bottom-left;
      // flip to image (top-left origin) pixel space.
      let faceRect = CGRect(
        x: face.boundingBox.minX * width,
        y: (1 - face.boundingBox.maxY) * height,
        width: face.boundingBox.width * width,
        height: face.boundingBox.height * height
      )

      func toImagePoints(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
        guard let region = region else { return [] }
        return region.normalizedPoints.map { p in
          CGPoint(
            x: faceRect.minX + CGFloat(p.x) * faceRect.width,
            y: faceRect.minY + (1 - CGFloat(p.y)) * faceRect.height
          )
        }
      }

      func avgY(_ points: [CGPoint]) -> CGFloat? {
        guard !points.isEmpty else { return nil }
        return points.reduce(0) { $0 + $1.y } / CGFloat(points.count)
      }

      let landmarks = face.landmarks
      let browPoints = toImagePoints(landmarks?.leftEyebrow) + toImagePoints(landmarks?.rightEyebrow)
      let nosePoints = toImagePoints(landmarks?.nose)
      let lipPoints = toImagePoints(landmarks?.outerLips)

      let browY = avgY(browPoints) ?? (faceRect.minY + faceRect.height * 0.36)
      let noseXs = nosePoints.map { $0.x }
      let noseYs = nosePoints.map { $0.y }
      let noseCenterX = noseXs.isEmpty ? faceRect.midX : noseXs.reduce(0, +) / CGFloat(noseXs.count)
      let noseTopY = noseYs.min() ?? (faceRect.minY + faceRect.height * 0.38)
      let noseBottomY = noseYs.max() ?? (faceRect.minY + faceRect.height * 0.62)
      let lipBottomY = lipPoints.map { $0.y }.max() ?? (faceRect.minY + faceRect.height * 0.82)

      return [
        "faceLeft": faceRect.minX,
        "faceTop": faceRect.minY,
        "faceWidth": faceRect.width,
        "faceHeight": faceRect.height,
        "browY": browY,
        "noseCenterX": noseCenterX,
        "noseTopY": noseTopY,
        "noseBottomY": noseBottomY,
        "lipBottomY": lipBottomY,
        "imageWidth": width,
        "imageHeight": height,
      ]
    }

    // Bounding-box-only detection: a lighter fallback for devices/OS builds
    // where the full landmarks model can't initialize (region boxes then
    // fall back to the proportional defaults in payload(for:)).
    func detectFaceRectanglesOnly() {
      let request = VNDetectFaceRectanglesRequest { request, error in
        guard error == nil,
              let faces = request.results as? [VNFaceObservation],
              let face = faces.max(by: { $0.boundingBox.width < $1.boundingBox.width })
        else {
          DispatchQueue.main.async { result(nil) }
          return
        }
        DispatchQueue.main.async { result(payload(for: face)) }
      }
      let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation)
      do {
        try handler.perform([request])
      } catch {
        DispatchQueue.main.async { result(nil) }
      }
    }

    let landmarksRequest = VNDetectFaceLandmarksRequest { request, error in
      if error != nil {
        detectFaceRectanglesOnly()
        return
      }
      guard let faces = request.results as? [VNFaceObservation],
            let face = faces.max(by: { $0.boundingBox.width < $1.boundingBox.width })
      else {
        DispatchQueue.main.async { result(nil) }
        return
      }
      DispatchQueue.main.async { result(payload(for: face)) }
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation)
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        try handler.perform([landmarksRequest])
      } catch {
        detectFaceRectanglesOnly()
      }
    }
  }

  private func cgOrientation(_ ui: UIImage.Orientation) -> CGImagePropertyOrientation {
    switch ui {
    case .up: return .up
    case .down: return .down
    case .left: return .left
    case .right: return .right
    case .upMirrored: return .upMirrored
    case .downMirrored: return .downMirrored
    case .leftMirrored: return .leftMirrored
    case .rightMirrored: return .rightMirrored
    @unknown default: return .up
    }
  }
}
