package br.edu.ifce.acneapp.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Design tokens shared by every screen (same values as the Flutter prototype). */
object AppColors {
    val PrimaryDark = Color(0xFF0B4F47)
    val Primary = Color(0xFF0F6B5F)
    val PrimaryLight = Color(0xFF1C8C77)

    val Background = Color(0xFFF5F9F7)
    val Surface = Color.White
    val Outline = Color(0xFFE1EAE6)

    val TextPrimary = Color(0xFF17241F)
    val TextSecondary = Color(0xFF647470)

    val Clear = Color(0xFF4CAF50)
    val Mild = Color(0xFFE9A824)
    val Moderate = Color(0xFFF2994A)
    val Severe = Color(0xFFE5484D)

    fun forSeverity(level: Int): Color = when (level) {
        0 -> Clear
        1 -> Mild
        2 -> Moderate
        else -> Severe
    }
}

object AppRadius {
    val Sm = 10.dp
    val Md = 16.dp
    val Lg = 24.dp
}

private val colorScheme = lightColorScheme(
    primary = AppColors.Primary,
    onPrimary = Color.White,
    secondary = AppColors.PrimaryLight,
    background = AppColors.Background,
    surface = AppColors.Surface,
    onSurface = AppColors.TextPrimary,
    outline = AppColors.Outline,
    error = AppColors.Severe,
)

private val typography = Typography(
    headlineSmall = TextStyle(fontSize = 26.sp, fontWeight = FontWeight.Bold, color = AppColors.TextPrimary),
    titleLarge = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.Bold, color = AppColors.TextPrimary),
    titleMedium = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = AppColors.TextPrimary),
    bodyLarge = TextStyle(fontSize = 16.sp, color = AppColors.TextPrimary, lineHeight = 22.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, color = AppColors.TextSecondary, lineHeight = 20.sp),
    labelLarge = TextStyle(fontSize = 15.sp, fontWeight = FontWeight.SemiBold),
)

@Composable
fun AcneAppTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = colorScheme, typography = typography, content = content)
}
