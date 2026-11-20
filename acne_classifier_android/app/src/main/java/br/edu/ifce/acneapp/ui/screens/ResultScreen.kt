package br.edu.ifce.acneapp.ui.screens

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.edu.ifce.acneapp.data.AnalysisRecord
import br.edu.ifce.acneapp.data.HistoryRepository
import br.edu.ifce.acneapp.data.PROFESSIONAL_DISCLAIMER
import br.edu.ifce.acneapp.data.RECOMMENDATIONS_BY_LEVEL
import br.edu.ifce.acneapp.data.RegionRecord
import br.edu.ifce.acneapp.data.SEVERITY_LABELS
import br.edu.ifce.acneapp.ui.components.AppCard
import br.edu.ifce.acneapp.ui.components.AppTopBar
import br.edu.ifce.acneapp.ui.components.SeverityMarker
import br.edu.ifce.acneapp.ui.components.SeverityScale
import br.edu.ifce.acneapp.ui.theme.AppColors
import br.edu.ifce.acneapp.ui.theme.AppRadius
import coil.compose.AsyncImage
import java.io.File
import java.util.Locale

@Composable
fun ResultScreen(id: String, onClose: () -> Unit) {
    val context = LocalContext.current
    var record by remember { mutableStateOf<AnalysisRecord?>(null) }
    var missing by remember { mutableStateOf(false) }
    LaunchedEffect(id) {
        record = HistoryRepository(context).find(id)
        missing = record == null
    }

    Column(Modifier.fillMaxSize()) {
        val current = record
        AppTopBar(
            title = "Result",
            navigationIcon = { IconButton(onClick = onClose) { Icon(Icons.Rounded.Close, contentDescription = "Close") } },
            actions = {
                if (current != null) {
                    IconButton(onClick = { share(context, current) }) { Icon(Icons.Rounded.Share, contentDescription = "Share") }
                }
            },
        )
        when {
            current != null -> ResultContent(current, onClose)
            missing -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("This analysis is no longer available.", style = MaterialTheme.typography.bodyMedium)
            }
            else -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = AppColors.Primary)
            }
        }
    }
}

@Composable
private fun ResultContent(record: AnalysisRecord, onClose: () -> Unit) {
    val overall = record.overallSeverity
    val color = AppColors.forSeverity(overall)
    val recommendation = RECOMMENDATIONS_BY_LEVEL[overall]

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp)
            .navigationBarsPadding(),
    ) {
        Row {
            AsyncImage(
                model = File(record.imagePath),
                contentDescription = "Analyzed photo",
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .size(112.dp)
                    .clip(RoundedCornerShape(AppRadius.Md))
                    .background(AppColors.Outline),
            )
            Spacer(Modifier.width(16.dp))
            Column(Modifier.weight(1f)) {
                Text("Overall Acne Severity", fontSize = 13.sp, color = AppColors.TextSecondary)
                Text(SEVERITY_LABELS[overall], fontSize = 26.sp, fontWeight = FontWeight.Bold, color = color)
                Spacer(Modifier.height(4.dp))
                Text(
                    "Based on the most affected of ${record.regions.size} analyzed regions",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
        Spacer(Modifier.height(24.dp))
        SeverityScale(overall)
        Spacer(Modifier.height(32.dp))
        Text("By region", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            record.regions.forEach { RegionResultCard(it) }
        }
        Spacer(Modifier.height(16.dp))
        AppCard {
            Row {
                Icon(Icons.Outlined.Description, contentDescription = null, tint = AppColors.Primary)
                Spacer(Modifier.width(12.dp))
                Column {
                    Text("Recommendations", fontWeight = FontWeight.Bold, fontSize = 15.5.sp, color = AppColors.TextPrimary)
                    Spacer(Modifier.height(6.dp))
                    Text(recommendation.summary, style = MaterialTheme.typography.bodyMedium)
                    Spacer(Modifier.height(8.dp))
                    recommendation.suggestions.forEach { s ->
                        Row(Modifier.padding(bottom = 4.dp)) {
                            Text("•  ", style = MaterialTheme.typography.bodyMedium)
                            Text(s, style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }
        }
        Spacer(Modifier.height(16.dp))
        Row(
            Modifier
                .fillMaxWidth()
                .background(AppColors.Outline.copy(alpha = 0.4f), RoundedCornerShape(AppRadius.Md))
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Rounded.Info, contentDescription = null, tint = AppColors.TextSecondary, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(10.dp))
            Text(PROFESSIONAL_DISCLAIMER, fontSize = 12.sp, color = AppColors.TextSecondary)
        }
        Spacer(Modifier.height(24.dp))
        OutlinedButton(
            onClick = onClose,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(AppRadius.Md),
            border = androidx.compose.foundation.BorderStroke(1.4.dp, AppColors.Outline),
        ) {
            Text("Back to Home", color = AppColors.Primary, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun RegionResultCard(region: RegionRecord) {
    val color = AppColors.forSeverity(region.severityLevel)
    AppCard(padding = androidx.compose.foundation.layout.PaddingValues(horizontal = 16.dp, vertical = 14.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SeverityMarker(region.severityLevel)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(region.region, fontWeight = FontWeight.SemiBold, fontSize = 14.5.sp, color = AppColors.TextPrimary)
                Text(SEVERITY_LABELS[region.severityLevel], color = color, fontWeight = FontWeight.Bold, fontSize = 15.sp)
            }
            Text(percent(region.confidence), fontWeight = FontWeight.Bold, color = AppColors.TextSecondary, fontSize = 13.sp)
        }
    }
}

private fun percent(value: Float) = String.format(Locale.US, "%.1f%%", value * 100)

private fun share(context: Context, record: AnalysisRecord) {
    val lines = record.regions.joinToString("\n") {
        "${it.region}: ${SEVERITY_LABELS[it.severityLevel]} (${percent(it.confidence)})"
    }
    val text = "Acne severity assessment\nOverall: ${SEVERITY_LABELS[record.overallSeverity]}\n\n$lines"
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_TEXT, text)
    }
    context.startActivity(Intent.createChooser(intent, "Share result"))
}
