package br.edu.ifce.acneapp.ml

import android.graphics.Bitmap

/**
 * The five anatomical regions, in the order the ONNX models expect as `region_idx`.
 * Must match REGIONS in training/run_ablation.py: forehead=0, chin=1, nose=2,
 * left_cheek=3, right_cheek=4.
 */
enum class FaceRegion(val label: String) {
    FOREHEAD("Forehead"),
    CHIN("Chin"),
    NOSE("Nose"),
    LEFT_CHEEK("Left Cheek"),
    RIGHT_CHEEK("Right Cheek");

    val modelIndex: Int get() = ordinal
}

/** A 224x224 crop of one anatomical region, ready for the classifier. */
class ExtractedRegion(val region: FaceRegion, val crop: Bitmap)

class NoFaceDetectedException(message: String = "No face was detected in this photo.") : Exception(message)
