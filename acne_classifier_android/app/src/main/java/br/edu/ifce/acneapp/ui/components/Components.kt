package br.edu.ifce.acneapp.ui.components

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.edu.ifce.acneapp.data.SEVERITY_LABELS
import br.edu.ifce.acneapp.ui.theme.AppColors
import br.edu.ifce.acneapp.ui.theme.AppRadius

/** Flat outlined card used across the app. */
@Composable
fun AppCard(
    modifier: Modifier = Modifier,
    padding: PaddingValues = PaddingValues(18.dp),
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(AppColors.Surface, RoundedCornerShape(AppRadius.Md))
            .border(1.dp, AppColors.Outline, RoundedCornerShape(AppRadius.Md))
            .padding(padding),
        content = content,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppTopBar(
    title: String,
    navigationIcon: @Composable () -> Unit = {},
    actions: @Composable androidx.compose.foundation.layout.RowScope.() -> Unit = {},
) {
    CenterAlignedTopAppBar(
        title = { Text(title, fontSize = 18.sp, fontWeight = FontWeight.SemiBold) },
        navigationIcon = navigationIcon,
        actions = actions,
        colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
            containerColor = AppColors.Primary,
            titleContentColor = Color.White,
            navigationIconContentColor = Color.White,
            actionIconContentColor = Color.White,
        ),
    )
}

/** Four-segment Clear/Mild/Moderate/Severe bar with a pointer over the active segment. */
@Composable
fun SeverityScale(level: Int, modifier: Modifier = Modifier) {
    BoxWithConstraints(modifier.fillMaxWidth()) {
        val segment = maxWidth / 4
        val pointerX by animateDpAsState(segment * level + segment / 2 - 7.dp, label = "pointer")
        val active = AppColors.forSeverity(level)
        Column {
            Box(Modifier.fillMaxWidth().height(10.dp)) {
                Canvas(Modifier.offset { IntOffset(pointerX.roundToPx(), 0) }.size(14.dp, 8.dp)) {
                    val path = Path().apply {
                        moveTo(size.width / 2, size.height)
                        lineTo(0f, 0f)
                        lineTo(size.width, 0f)
                        close()
                    }
                    drawPath(path, active)
                }
            }
            Spacer(Modifier.height(4.dp))
            Row(Modifier.fillMaxWidth()) {
                for (i in 0 until 4) {
                    Box(
                        Modifier
                            .weight(1f)
                            .padding(end = if (i == 3) 0.dp else 4.dp)
                            .height(8.dp)
                            .background(
                                AppColors.forSeverity(i).copy(alpha = if (i == level) 1f else 0.28f),
                                RoundedCornerShape(50),
                            ),
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth()) {
                for (i in 0 until 4) {
                    Text(
                        SEVERITY_LABELS[i],
                        modifier = Modifier.weight(1f),
                        textAlign = TextAlign.Center,
                        fontSize = 12.5.sp,
                        fontWeight = if (i == level) FontWeight.Bold else FontWeight.Medium,
                        color = if (i == level) AppColors.forSeverity(i) else AppColors.TextSecondary,
                    )
                }
            }
        }
    }
}

/** Thin colored bar used as a severity marker in lists. */
@Composable
fun SeverityMarker(level: Int) {
    Box(
        Modifier
            .width(8.dp)
            .height(36.dp)
            .background(AppColors.forSeverity(level), RoundedCornerShape(50)),
    )
}
