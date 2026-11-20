package br.edu.ifce.acneapp.ui.screens

import android.os.SystemClock
import android.util.Log
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.RadioButtonUnchecked
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.edu.ifce.acneapp.data.AnalysisRecord
import br.edu.ifce.acneapp.data.HistoryRepository
import br.edu.ifce.acneapp.data.RegionRecord
import br.edu.ifce.acneapp.ml.FaceRegionExtractor
import br.edu.ifce.acneapp.ml.NoFaceDetectedException
import br.edu.ifce.acneapp.ml.OnnxClassifier
import br.edu.ifce.acneapp.ui.components.AppTopBar
import br.edu.ifce.acneapp.ui.theme.AppColors
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

private val STEPS = listOf("Preprocessing", "Extracting regions", "Classification", "Post-processing")
private const val TAG = "BENCH"

/**
 * Runs the full on-device pipeline (model loading, face/region extraction,
 * ensemble classification), saves the result to the history and hands its id on.
 * Stage timings are written to logcat under the tag BENCH.
 */
@Composable
fun AnalyzingScreen(photo: String, onBack: () -> Unit, onDone: (String) -> Unit) {
    val context = LocalContext.current
    var step by remember { mutableIntStateOf(0) }
    var error by remember { mutableStateOf<String?>(null) }
    val progress = remember { Animatable(0f) }

    LaunchedEffect(photo) {
        launch { progress.animateTo(1f, tween(durationMillis = 8000, easing = LinearEasing)) }
        val extractor = FaceRegionExtractor()
        try {
            val t0 = SystemClock.elapsedRealtime()
            step = 0
            OnnxClassifier.ensureLoaded(context)
            Log.d(TAG, "model_load_ms=${SystemClock.elapsedRealtime() - t0}")

            step = 1
            val t1 = SystemClock.elapsedRealtime()
            val regions = extractor.extract(photo)
            Log.d(TAG, "face_extract_ms=${SystemClock.elapsedRealtime() - t1} regions=${regions.size}")
            if (regions.isEmpty()) {
                throw NoFaceDetectedException(
                    "A face was found, but no region could be cropped from this photo. " +
                        "Try a straight-on photo with the whole face visible.",
                )
            }

            step = 2
            val t2 = SystemClock.elapsedRealtime()
            val predictions = OnnxClassifier.predict(context, regions)
            Log.d(TAG, "inference_ms=${SystemClock.elapsedRealtime() - t2} regions=${regions.size}")
            Log.d(TAG, "total_pipeline_ms=${SystemClock.elapsedRealtime() - t0}")

            step = 3
            val record = AnalysisRecord(
                id = File(photo).nameWithoutExtension,
                imagePath = photo,
                regions = predictions.map { RegionRecord(it.region.label, it.severityLevel, it.confidence) },
                createdAtMillis = System.currentTimeMillis(),
            )
            HistoryRepository(context).add(record)
            delay(300)
            onDone(record.id)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "analysis failed", e)
            error = e.message ?: e.toString()
        } finally {
            extractor.close()
        }
    }

    Column(Modifier.fillMaxSize()) {
        AppTopBar(
            title = "Analyzing…",
            navigationIcon = {
                IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = "Back") }
            },
        )
        val message = error
        if (message != null) {
            Column(
                Modifier.fillMaxSize().padding(28.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Icon(Icons.Rounded.ErrorOutline, contentDescription = null, tint = AppColors.Severe, modifier = Modifier.size(48.dp))
                Spacer(Modifier.height(16.dp))
                Text("Could not complete the analysis", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                Text(message, style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Center)
                Spacer(Modifier.height(32.dp))
                WideButton("Try Again", null, AppColors.Primary, androidx.compose.ui.graphics.Color.White, onBack)
            }
            return@Column
        }

        Column(
            Modifier.fillMaxSize().padding(horizontal = 32.dp, vertical = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(Modifier.height(24.dp))
            Box(Modifier.size(180.dp), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(
                    progress = { progress.value },
                    modifier = Modifier.size(180.dp),
                    strokeWidth = 10.dp,
                    color = AppColors.Primary,
                    trackColor = AppColors.Outline,
                )
                Text(
                    "${(progress.value * 100).toInt().coerceIn(0, 99)}%",
                    fontSize = 34.sp,
                    fontWeight = FontWeight.Bold,
                    color = AppColors.Primary,
                )
            }
            Spacer(Modifier.height(32.dp))
            Text("Analyzing acne severity", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))
            Text("Running the on-device model across all regions…", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(32.dp))
            STEPS.forEachIndexed { i, label ->
                val done = i < step
                val active = i == step
                Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        label,
                        modifier = Modifier.weight(1f),
                        fontSize = 15.sp,
                        fontWeight = if (active) FontWeight.SemiBold else FontWeight.Normal,
                        color = if (done || active) AppColors.TextPrimary else AppColors.TextSecondary,
                    )
                    when {
                        done -> Icon(Icons.Rounded.CheckCircle, contentDescription = null, tint = AppColors.Primary, modifier = Modifier.size(22.dp))
                        active -> CircularProgressIndicator(strokeWidth = 2.2.dp, color = AppColors.PrimaryLight, modifier = Modifier.size(18.dp))
                        else -> Icon(Icons.Rounded.RadioButtonUnchecked, contentDescription = null, tint = AppColors.Outline, modifier = Modifier.size(20.dp))
                    }
                }
            }
        }
    }
}
