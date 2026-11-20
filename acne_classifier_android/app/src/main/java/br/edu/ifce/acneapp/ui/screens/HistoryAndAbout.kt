package br.edu.ifce.acneapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.PrivacyTip
import androidx.compose.material.icons.outlined.Science
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Face
import androidx.compose.material.icons.rounded.History
import androidx.compose.material.icons.rounded.WarningAmber
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.edu.ifce.acneapp.data.AnalysisRecord
import br.edu.ifce.acneapp.data.HistoryRepository
import br.edu.ifce.acneapp.data.SEVERITY_LABELS
import br.edu.ifce.acneapp.ui.components.AppCard
import br.edu.ifce.acneapp.ui.components.AppTopBar
import br.edu.ifce.acneapp.ui.theme.AppColors
import br.edu.ifce.acneapp.ui.theme.AppRadius
import coil.compose.AsyncImage
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryScreen(onOpenRecord: (String) -> Unit) {
    val context = LocalContext.current
    var records by remember { mutableStateOf<List<AnalysisRecord>?>(null) }
    LaunchedEffect(Unit) { records = HistoryRepository(context).load() }

    Column(Modifier.fillMaxSize()) {
        AppTopBar("History")
        val list = records
        when {
            list == null -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = AppColors.Primary)
            }
            list.isEmpty() -> EmptyHistory()
            else -> LazyColumn(
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                items(list, key = { it.id }) { record -> HistoryTile(record) { onOpenRecord(record.id) } }
            }
        }
    }
}

private val DATE_FORMAT = SimpleDateFormat("MMM d, yyyy  HH:mm", Locale.US)

@Composable
private fun HistoryTile(record: AnalysisRecord, onClick: () -> Unit) {
    val overall = record.overallSeverity
    val color = AppColors.forSeverity(overall)
    AppCard(
        modifier = Modifier.clip(RoundedCornerShape(AppRadius.Md)).clickable(onClick = onClick),
        padding = PaddingValues(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            AsyncImage(
                model = File(record.imagePath),
                contentDescription = null,
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(AppRadius.Sm))
                    .background(AppColors.Outline),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("${record.regions.size} regions analyzed", color = AppColors.TextSecondary, fontSize = 12.5.sp)
                Text(SEVERITY_LABELS[overall], color = color, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                Text(DATE_FORMAT.format(Date(record.createdAtMillis)), color = AppColors.TextSecondary, fontSize = 12.sp)
            }
            Box(
                Modifier
                    .background(color.copy(alpha = 0.12f), RoundedCornerShape(50))
                    .padding(horizontal = 10.dp, vertical = 5.dp),
            ) {
                Text(
                    String.format(Locale.US, "%.1f%%", record.overallConfidence * 100),
                    color = color,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.5.sp,
                )
            }
            Spacer(Modifier.width(6.dp))
            Icon(Icons.Rounded.ChevronRight, contentDescription = null, tint = AppColors.TextSecondary)
        }
    }
}

@Composable
private fun EmptyHistory() {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(Icons.Rounded.History, contentDescription = null, tint = AppColors.TextSecondary, modifier = Modifier.size(48.dp))
        Spacer(Modifier.height(16.dp))
        Text("No analyses yet", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(6.dp))
        Text(
            "Results you analyze will show up here.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
fun AboutScreen() {
    Column(Modifier.fillMaxSize()) {
        AppTopBar("About")
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Box(
                Modifier.size(72.dp).background(AppColors.Primary, CircleShape),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Rounded.Face, contentDescription = null, tint = Color.White, modifier = Modifier.size(32.dp))
            }
            Spacer(Modifier.height(16.dp))
            Text("AcneApp", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(4.dp))
            Text("Research prototype", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(32.dp))
            InfoCard(
                Icons.Outlined.Science,
                "About this prototype",
                "This application runs a region-aware ResNet-18 ensemble entirely on-device: a photo is " +
                    "captured, the face and five anatomical regions are located automatically, and each region " +
                    "is scored on the four-level Hayashi scale (Clear, Mild, Moderate, Severe), then combined " +
                    "into an overall result.",
            )
            Spacer(Modifier.height(16.dp))
            InfoCard(
                Icons.Rounded.WarningAmber,
                "Not a medical device",
                "Results are informational only and do not replace a diagnosis from a qualified " +
                    "dermatologist. Always seek professional advice for treatment decisions.",
            )
            Spacer(Modifier.height(16.dp))
            InfoCard(
                Icons.Outlined.PrivacyTip,
                "Your data",
                "Photos and results stay on this device. Nothing is uploaded to a server by this app.",
            )
        }
    }
}

@Composable
internal fun InfoCard(icon: ImageVector, title: String, body: String) {
    AppCard {
        Row {
            Icon(icon, contentDescription = null, tint = AppColors.Primary)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.fillMaxWidth()) {
                Text(title, fontWeight = FontWeight.Bold, fontSize = 15.5.sp, color = AppColors.TextPrimary)
                Spacer(Modifier.height(6.dp))
                Text(body, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}
