package br.edu.ifce.acneapp.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.RectF
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.tasks.await
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Detects a face with Google ML Kit (on-device) and derives five anatomical region
 * crops from its contour anchors. The geometry is the same as in the Flutter
 * prototype (lib/services/face_region_extractor.dart): proportional boxes anchored
 * on the eyebrows, nose and lower lip, approximating the landmark-based regions
 * used to build the training crops (training/utils/segment_faces_mediapipe.py).
 */
class FaceRegionExtractor {

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()
    )

    suspend fun extract(imagePath: String): List<ExtractedRegion> {
        val bitmap = loadUprightBitmap(imagePath)
            ?: throw NoFaceDetectedException("Could not decode this image.")

        val faces = try {
            detector.process(InputImage.fromBitmap(bitmap, 0)).await()
        } catch (e: Exception) {
            throw NoFaceDetectedException(e.localizedMessage ?: "Face detection failed.")
        }
        val face = faces.maxByOrNull { it.boundingBox.width() } ?: throw NoFaceDetectedException()

        val boxes = regionBoxes(anchorsOf(face), bitmap.width, bitmap.height)
        return boxes.mapNotNull { (region, r) ->
            if (r.width() < 8f || r.height() < 8f) return@mapNotNull null
            val left = r.left.roundToInt()
            val top = r.top.roundToInt()
            val width = min(r.width().roundToInt(), bitmap.width - left)
            val height = min(r.height().roundToInt(), bitmap.height - top)
            if (width < 8 || height < 8) return@mapNotNull null
            val cropped = Bitmap.createBitmap(bitmap, left, top, width, height)
            // Bilinear resize, as torchvision's Resize((224, 224)) used in training.
            ExtractedRegion(region, Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true))
        }
    }

    fun close() = detector.close()

    private class Anchors(
        val faceLeft: Float, val faceTop: Float, val faceWidth: Float, val faceHeight: Float,
        val browY: Float, val noseCenterX: Float, val noseTopY: Float, val noseBottomY: Float,
        val lipBottomY: Float,
    )

    private fun anchorsOf(face: Face): Anchors {
        val box = face.boundingBox
        val h = box.height().toFloat()
        fun points(type: Int) = face.getContour(type)?.points.orEmpty()
        val brow = points(FaceContour.LEFT_EYEBROW_TOP) + points(FaceContour.RIGHT_EYEBROW_TOP)
        val nose = points(FaceContour.NOSE_BRIDGE) + points(FaceContour.NOSE_BOTTOM)
        val lip = points(FaceContour.LOWER_LIP_BOTTOM)
        return Anchors(
            faceLeft = box.left.toFloat(),
            faceTop = box.top.toFloat(),
            faceWidth = box.width().toFloat(),
            faceHeight = h,
            browY = if (brow.isEmpty()) box.top + h * 0.36f else brow.map { it.y }.average().toFloat(),
            noseCenterX = if (nose.isEmpty()) box.exactCenterX() else nose.map { it.x }.average().toFloat(),
            noseTopY = nose.minOfOrNull { it.y } ?: (box.top + h * 0.38f),
            noseBottomY = nose.maxOfOrNull { it.y } ?: (box.top + h * 0.62f),
            lipBottomY = lip.maxOfOrNull { it.y } ?: (box.top + h * 0.82f),
        )
    }

    private fun regionBoxes(a: Anchors, imgWidth: Int, imgHeight: Int): List<Pair<FaceRegion, RectF>> {
        val w = a.faceWidth
        val h = a.faceHeight
        val right = a.faceLeft + w
        val bottom = a.faceTop + h
        val noseTop = a.noseTopY - h * 0.02f
        val noseBottom = a.noseBottomY + h * 0.03f

        fun clamp(l: Float, t: Float, r: Float, b: Float) = RectF(
            l.coerceIn(0f, imgWidth.toFloat()),
            t.coerceIn(0f, imgHeight.toFloat()),
            r.coerceIn(0f, imgWidth.toFloat()),
            b.coerceIn(0f, imgHeight.toFloat()),
        )

        fun centered(cx: Float, cy: Float, bw: Float, bh: Float) =
            clamp(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

        return listOf(
            FaceRegion.FOREHEAD to clamp(a.faceLeft + w * 0.14f, a.faceTop - h * 0.04f, right - w * 0.14f, a.browY - h * 0.02f),
            FaceRegion.CHIN to clamp(a.faceLeft + w * 0.3f, a.lipBottomY + h * 0.02f, right - w * 0.3f, bottom + h * 0.05f),
            FaceRegion.NOSE to clamp(a.noseCenterX - w * 0.14f, noseTop, a.noseCenterX + w * 0.14f, noseBottom),
            FaceRegion.LEFT_CHEEK to centered(a.faceLeft + w * 0.22f, a.faceTop + h * 0.6f, w * 0.3f, h * 0.3f),
            FaceRegion.RIGHT_CHEEK to centered(right - w * 0.22f, a.faceTop + h * 0.6f, w * 0.3f, h * 0.3f),
        )
    }

    companion object {
        const val INPUT_SIZE = 224
        private const val MAX_SIDE = 2048

        /** Decodes the file (downsampled to at most [MAX_SIDE] px) and applies its EXIF rotation. */
        fun loadUprightBitmap(path: String): Bitmap? {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(path, bounds)
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
            var sample = 1
            while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MAX_SIDE) sample *= 2
            val original = BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
                ?: return null

            val orientation = try {
                ExifInterface(path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
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
                ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.postRotate(90f); matrix.postScale(-1f, 1f) }
                ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.postRotate(270f); matrix.postScale(-1f, 1f) }
                else -> return original
            }
            return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
        }
    }
}
