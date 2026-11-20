package br.edu.ifce.acneapp

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import br.edu.ifce.acneapp.ui.screens.AnalyzingScreen
import br.edu.ifce.acneapp.ui.screens.CaptureScreen
import br.edu.ifce.acneapp.ui.screens.HomeShell
import br.edu.ifce.acneapp.ui.screens.ResultScreen
import br.edu.ifce.acneapp.ui.theme.AcneAppTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // White status-bar icons over the green headers; dark navigation-bar icons over white.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.light(android.graphics.Color.WHITE, android.graphics.Color.WHITE),
        )
        super.onCreate(savedInstanceState)
        setContent {
            AcneAppTheme {
                AppNavigation()
            }
        }
    }
}

private object Routes {
    const val HOME = "home"
    const val CAPTURE = "capture"
    const val ANALYZING = "analyzing?photo={photo}"
    const val RESULT = "result/{id}"

    fun analyzing(photo: String) = "analyzing?photo=${Uri.encode(photo)}"
    fun result(id: String) = "result/$id"
}

/**
 * Tabs (Home / History / About) form the start destination; the capture,
 * analysis and result screens are pushed on top with their own back navigation.
 */
@Composable
private fun AppNavigation() {
    val nav = rememberNavController()
    NavHost(navController = nav, startDestination = Routes.HOME) {
        composable(Routes.HOME) {
            HomeShell(
                onStartAnalysis = { nav.navigate(Routes.CAPTURE) },
                onOpenRecord = { id -> nav.navigate(Routes.result(id)) },
            )
        }
        composable(Routes.CAPTURE) {
            CaptureScreen(
                onBack = { nav.popBackStack() },
                onPhotoReady = { photo -> nav.navigate(Routes.analyzing(photo)) },
            )
        }
        composable(
            Routes.ANALYZING,
            arguments = listOf(navArgument("photo") { type = NavType.StringType }),
        ) { entry ->
            AnalyzingScreen(
                photo = requireNotNull(entry.arguments?.getString("photo")),
                onBack = { nav.popBackStack() },
                onDone = { id ->
                    nav.navigate(Routes.result(id)) {
                        popUpTo(Routes.HOME)
                    }
                },
            )
        }
        composable(
            Routes.RESULT,
            arguments = listOf(navArgument("id") { type = NavType.StringType }),
        ) { entry ->
            ResultScreen(
                id = requireNotNull(entry.arguments?.getString("id")),
                onClose = { nav.popBackStack(Routes.HOME, inclusive = false) },
            )
        }
    }
}
