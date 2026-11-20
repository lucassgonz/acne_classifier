package br.edu.ifce.acneapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.Face
import androidx.compose.material.icons.rounded.History
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Schedule
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.edu.ifce.acneapp.ui.theme.AppColors
import br.edu.ifce.acneapp.ui.theme.AppRadius

private data class Tab(val label: String, val icon: ImageVector)

private val TABS = listOf(
    Tab("Home", Icons.Rounded.Home),
    Tab("History", Icons.Rounded.History),
    Tab("About", Icons.Outlined.Info),
)

@Composable
fun HomeShell(onStartAnalysis: () -> Unit, onOpenRecord: (String) -> Unit) {
    var index by rememberSaveable { mutableIntStateOf(0) }
    Scaffold(
        containerColor = AppColors.Background,
        // Each tab handles the status bar itself (gradient or top app bar).
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        bottomBar = {
            Column {
                HorizontalDivider(color = AppColors.Outline)
                NavigationBar(containerColor = AppColors.Surface, tonalElevation = 0.dp) {
                    TABS.forEachIndexed { i, tab ->
                        NavigationBarItem(
                            selected = index == i,
                            onClick = { index = i },
                            icon = { Icon(tab.icon, contentDescription = tab.label) },
                            label = { Text(tab.label, fontSize = 11.5.sp) },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = AppColors.Primary,
                                selectedTextColor = AppColors.Primary,
                                unselectedIconColor = AppColors.TextSecondary,
                                unselectedTextColor = AppColors.TextSecondary,
                                indicatorColor = AppColors.Primary.copy(alpha = 0.10f),
                            ),
                        )
                    }
                }
            }
        },
    ) { padding ->
        Box(Modifier.padding(padding)) {
            when (index) {
                0 -> HomeScreen(onStartAnalysis = onStartAnalysis, onViewHistory = { index = 1 })
                1 -> HistoryScreen(onOpenRecord = onOpenRecord)
                else -> AboutScreen()
            }
        }
    }
}

@Composable
private fun HomeScreen(onStartAnalysis: () -> Unit, onViewHistory: () -> Unit) {
    Column(
        Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(AppColors.PrimaryDark, AppColors.PrimaryLight)))
            .statusBarsPadding()
            .padding(start = 28.dp, end = 28.dp, bottom = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.weight(3f))
        Box(
            Modifier
                .size(92.dp)
                .border(1.4.dp, Color.White.copy(alpha = 0.55f), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Rounded.Face, contentDescription = null, tint = Color.White, modifier = Modifier.size(42.dp))
        }
        Spacer(Modifier.height(24.dp))
        Text("AcneApp", color = Color.White, fontSize = 32.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(4.dp))
        Text(
            "Automatic evaluation of facial\nacne severity using artificial intelligence",
            color = Color.White.copy(alpha = 0.85f),
            fontSize = 15.sp,
            lineHeight = 21.sp,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.weight(4f))
        WideButton("Start Analysis", Icons.Rounded.CameraAlt, AppColors.PrimaryDark, Color.White, onStartAnalysis)
        Spacer(Modifier.height(16.dp))
        WideButton("History", Icons.Rounded.Schedule, Color.White, AppColors.PrimaryDark, onViewHistory)
        Spacer(Modifier.height(24.dp))
        Text(
            "This is a research prototype and does not replace\nprofessional dermatological advice.",
            color = Color.White.copy(alpha = 0.72f),
            fontSize = 12.5.sp,
            lineHeight = 17.sp,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
internal fun WideButton(
    text: String,
    icon: ImageVector?,
    container: Color,
    content: Color,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth().height(56.dp),
        shape = RoundedCornerShape(AppRadius.Md),
        colors = ButtonDefaults.buttonColors(containerColor = container, contentColor = content),
        elevation = null,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.Center) {
            if (icon != null) {
                Icon(icon, contentDescription = null, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
            }
            Text(text, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}
