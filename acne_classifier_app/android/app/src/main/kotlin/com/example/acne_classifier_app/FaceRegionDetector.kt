package com.example.acne_classifier_app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * Android counterpart to ios/Runner/FaceRegionDetector.swift: detects a face
 * with Google ML Kit (on-device) and returns the same pixel-space anchor
 * payload, so the Dart side (face_region_extractor.dart) derives identical
 * region boxes regardless of platform.
 */
class FaceRegionDetector : FlutterPlugin, MethodChannel.MethodCallHandler {
    private var channel: MethodChannel? = null
    private var context: Context? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        context = binding.applicationContext
        channel = MethodChannel(binding.binaryMessenger, "acne_app/face_region_detector")
        channel?.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel?.setMethodCallHandler(null)
        channel = null
        context = null
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "detectFace") {
            result.notImplemented()
            return
        }
        val path = call.argument<String>("path")
        if (path == null) {
            result.error("bad_image", "No path provided", null)
            return
        }

        val bitmap = loadUprightBitmap(path)
        if (bitmap == null) {
            result.error("bad_image", "Could not load image", null)
            return
        }

        val detector = FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .build()
        )

        detector.process(InputImage.fromBitmap(bitmap, 0))
            .addOnSuccessListener { faces ->
                val face = faces.maxByOrNull { it.boundingBox.width() }
                if (face == null) {
                    result.success(null)
                    return@addOnSuccessListener
                }

                val box = face.boundingBox
                val browPoints = (contourPoints(face.getContour(FaceContour.LEFT_EYEBROW_TOP)) +
                    contourPoints(face.getContour(FaceContour.RIGHT_EYEBROW_TOP)))
                val nosePoints = (contourPoints(face.getContour(FaceContour.NOSE_BRIDGE)) +
                    contourPoints(face.getContour(FaceContour.NOSE_BOTTOM)))
                val lipPoints = contourPoints(face.getContour(FaceContour.LOWER_LIP_BOTTOM))

                val faceHeight = box.height().toDouble()
                val browY = browPoints.map { it.second }.average0(box.top + faceHeight * 0.36)
                val noseCenterX = nosePoints.map { it.first }.average0(box.exactCenterX().toDouble())
                val noseTopY = nosePoints.map { it.second }.minOrNullD(box.top + faceHeight * 0.38)
                val noseBottomY = nosePoints.map { it.second }.maxOrNullD(box.top + faceHeight * 0.62)
                val lipBottomY = lipPoints.map { it.second }.maxOrNullD(box.top + faceHeight * 0.82)

                val payload = mapOf(
                    "faceLeft" to box.left.toDouble(),
                    "faceTop" to box.top.toDouble(),
                    "faceWidth" to box.width().toDouble(),
                    "faceHeight" to box.height().toDouble(),
                    "browY" to browY,
                    "noseCenterX" to noseCenterX,
                    "noseTopY" to noseTopY,
                    "noseBottomY" to noseBottomY,
                    "lipBottomY" to lipBottomY,
                    "imageWidth" to bitmap.width.toDouble(),
                    "imageHeight" to bitmap.height.toDouble(),
                )
                result.success(payload)
            }
            .addOnFailureListener { e ->
                result.error("vision_failed", e.localizedMessage, null)
            }
    }

    private fun contourPoints(contour: FaceContour?): List<Pair<Double, Double>> {
        if (contour == null) return emptyList()
        return contour.points.map { it.x.toDouble() to it.y.toDouble() }
    }

    private fun List<Double>.average0(fallback: Double): Double =
        if (isEmpty()) fallback else average()

    private fun List<Double>.minOrNullD(fallback: Double): Double = minOrNull() ?: fallback

    private fun List<Double>.maxOrNullD(fallback: Double): Double = maxOrNull() ?: fallback

    /** Decodes the file and bakes in EXIF rotation, mirroring img.bakeOrientation on the Dart side. */
    private fun loadUprightBitmap(path: String): Bitmap? {
        val original = BitmapFactory.decodeFile(path) ?: return null
        val orientation = try {
            ExifInterface(path).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return original
        }
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
    }
}
