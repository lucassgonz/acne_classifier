package br.edu.ifce.acneapp.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.nio.LongBuffer

class RegionPrediction(
    val region: FaceRegion,
    val severityLevel: Int, // 0=Clear, 1=Mild, 2=Moderate, 3=Severe
    val confidence: Float,
    val probabilities: FloatArray,
)

/**
 * Runs the trained Region-Aware ResNet-18 ensemble (5 seeds, exported to ONNX by
 * training/utils/export_onnx.py) entirely on-device. Each model takes
 * `image` [N, 3, 224, 224] (RGB scaled to 0..1, no mean/std normalization, as
 * torchvision's ToTensor in training) and `region_idx` [N] (int64), and returns
 * softmax probabilities `probs` [N, 4]. The ensemble output is the mean of the
 * five models' probabilities.
 *
 * Sessions are created once per process and reused across analyses.
 */
object OnnxClassifier {
    private val SEEDS = listOf(42, 43, 44, 45, 46)
    private const val NUM_CLASSES = 4

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val sessions = mutableListOf<OrtSession>()
    private val lock = Mutex()

    val isLoaded: Boolean get() = sessions.size == SEEDS.size

    suspend fun ensureLoaded(context: Context) = lock.withLock {
        if (isLoaded) return@withLock
        withContext(Dispatchers.IO) {
            val options = OrtSession.SessionOptions()
            for (seed in SEEDS) {
                val bytes = context.assets.open("region_model_seed$seed.onnx").use { it.readBytes() }
                sessions += env.createSession(bytes, options)
            }
        }
    }

    suspend fun predict(context: Context, regions: List<ExtractedRegion>): List<RegionPrediction> {
        if (regions.isEmpty()) return emptyList()
        ensureLoaded(context)
        return withContext(Dispatchers.Default) {
            val n = regions.size
            val pixels = FloatBuffer.allocate(n * 3 * FaceRegionExtractor.INPUT_SIZE * FaceRegionExtractor.INPUT_SIZE)
            regions.forEach { writeChw(it, pixels) }
            pixels.rewind()
            val regionIdx = LongBuffer.wrap(LongArray(n) { regions[it].region.modelIndex.toLong() })

            val sums = Array(n) { FloatArray(NUM_CLASSES) }
            val size = FaceRegionExtractor.INPUT_SIZE.toLong()
            OnnxTensor.createTensor(env, pixels, longArrayOf(n.toLong(), 3, size, size)).use { image ->
                OnnxTensor.createTensor(env, regionIdx, longArrayOf(n.toLong())).use { idx ->
                    for (session in sessions) {
                        session.run(mapOf("image" to image, "region_idx" to idx)).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            val probs = result[0].value as Array<FloatArray>
                            for (i in 0 until n) for (k in 0 until NUM_CLASSES) sums[i][k] += probs[i][k]
                        }
                    }
                }
            }

            regions.mapIndexed { i, r ->
                val probs = FloatArray(NUM_CLASSES) { k -> sums[i][k] / sessions.size }
                val best = probs.indices.maxBy { probs[it] }
                RegionPrediction(r.region, best, probs[best], probs)
            }
        }
    }

    /** Appends the crop in CHW order (all R, then G, then B), scaled to 0..1. */
    private fun writeChw(region: ExtractedRegion, out: FloatBuffer) {
        val s = FaceRegionExtractor.INPUT_SIZE
        val argb = IntArray(s * s)
        region.crop.getPixels(argb, 0, s, 0, 0, s, s)
        for (c in 0 until 3) {
            val shift = 16 - 8 * c // R=16, G=8, B=0
            for (p in argb) out.put(((p shr shift) and 0xFF) / 255f)
        }
    }
}
