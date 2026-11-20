package br.edu.ifce.acneapp.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.Cameraswitch
import androidx.compose.material.icons.rounded.PhotoLibrary
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import br.edu.ifce.acneapp.data.HistoryRepository
import br.edu.ifce.acneapp.ui.components.AppTopBar
import br.edu.ifce.acneapp.ui.theme.AppColors
import br.edu.ifce.acneapp.ui.theme.AppRadius
import kotlinx.coroutines.launch

/**
 * Camera preview with a face guide, capture button, gallery picker and camera flip.
 * The chosen photo is stored in the app's private storage and its path is passed on.
 */
@Composable
fun CaptureScreen(onBack: () -> Unit, onPhotoReady: (String) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val repo = remember { HistoryRepository(context) }

    var hasPermission by remember { mutableStateOf(hasCameraPermission(context)) }
    var useFront by remember { mutableStateOf(true) }
    var capturing by remember { mutableStateOf(false) }
    var showTips by remember { mutableStateOf(false) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        hasPermission = it
    }
    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            scope.launch {
                runCatching { repo.importPhoto(uri, newId()) }
                    .onSuccess(onPhotoReady)
                    .onFailure { Toast.makeText(context, "Could not open the image: ${it.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }
    val pickFromGallery = {
        galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    LaunchedEffect(Unit) { if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA) }

    val imageCapture = remember { ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build() }
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }

    if (hasPermission) {
        DisposableEffect(useFront) {
            val providerFuture = ProcessCameraProvider.getInstance(context)
            providerFuture.addListener({
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                val wanted = if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                val selector = if (provider.hasCamera(wanted)) wanted
                else if (provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) CameraSelector.DEFAULT_BACK_CAMERA
                else null
                if (selector == null) {
                    cameraError = "No camera available on this device."
                    return@addListener
                }
                try {
                    provider.unbindAll()
                    provider.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture)
                    cameraError = null
                } catch (e: Exception) {
                    Log.e("CaptureScreen", "Camera bind failed", e)
                    cameraError = "Could not start the camera: ${e.message}"
                }
            }, ContextCompat.getMainExecutor(context))
            onDispose { runCatching { providerFuture.get().unbindAll() } }
        }
    }

    fun takePicture() {
        if (capturing) return
        capturing = true
        val id = newId()
        val file = repo.newPhotoFile(id)
        imageCapture.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    capturing = false
                    onPhotoReady(file.absolutePath)
                }

                override fun onError(exception: ImageCaptureException) {
                    capturing = false
                    Toast.makeText(context, "Could not capture the photo: ${exception.message}", Toast.LENGTH_LONG).show()
                }
            },
        )
    }

    Column(Modifier.fillMaxSize().background(Color.Black)) {
        AppTopBar(
            title = "Capture Image",
            navigationIcon = {
                IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = "Back") }
            },
            actions = {
                IconButton(onClick = { showTips = true }) { Icon(Icons.Outlined.Info, contentDescription = "Tips") }
            },
        )

        val error = cameraError ?: if (!hasPermission) "Camera permission is needed to take a photo." else null
        if (error != null) {
            Column(
                Modifier.fillMaxSize().padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(error, color = Color.White, textAlign = TextAlign.Center)
                Spacer(Modifier.height(20.dp))
                WideButton("Choose from Gallery", Icons.Rounded.PhotoLibrary, AppColors.Primary, Color.White) { pickFromGallery() }
                if (!hasPermission) {
                    Spacer(Modifier.height(12.dp))
                    TextButton(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                        Text("Allow camera", color = Color.White)
                    }
                }
            }
        } else {
            Box(Modifier.weight(1f).fillMaxWidth()) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                Box(
                    Modifier
                        .align(Alignment.Center)
                        .size(240.dp, 310.dp)
                        .border(2.2.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(130.dp)),
                )
                Text(
                    "Position your face within the guide\nand ensure good lighting.",
                    modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 20.dp),
                    textAlign = TextAlign.Center,
                    style = TextStyle(
                        color = Color.White.copy(alpha = 0.9f),
                        fontSize = 14.5.sp,
                        shadow = Shadow(Color.Black, blurRadius = 6f),
                    ),
                )
            }
            Row(
                Modifier.fillMaxWidth().background(Color.Black).navigationBarsPadding().padding(vertical = 22.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SquareIconButton(Icons.Rounded.PhotoLibrary, "Gallery") { pickFromGallery() }
                Box(
                    Modifier
                        .size(76.dp)
                        .background(AppColors.PrimaryLight, CircleShape)
                        .border(4.dp, Color.White, CircleShape)
                        .clickable(enabled = !capturing) { takePicture() },
                    contentAlignment = Alignment.Center,
                ) {
                    if (capturing) {
                        CircularProgressIndicator(color = Color.White, strokeWidth = 2.4.dp, modifier = Modifier.size(30.dp))
                    } else {
                        Icon(Icons.Rounded.CameraAlt, contentDescription = "Take photo", tint = Color.White, modifier = Modifier.size(30.dp))
                    }
                }
                SquareIconButton(Icons.Rounded.Cameraswitch, "Flip") { useFront = !useFront }
            }
        }
    }

    if (showTips) {
        AlertDialog(
            onDismissRequest = { showTips = false },
            confirmButton = { TextButton(onClick = { showTips = false }) { Text("OK") } },
            title = { Text("Tips for a good photo", fontWeight = FontWeight.Bold) },
            text = {
                Text(
                    "• Use even, natural lighting\n• Keep the face clean of makeup\n" +
                        "• Look straight at the camera\n• Fill the oval guide with your face",
                )
            },
        )
    }
}

@Composable
private fun SquareIconButton(icon: ImageVector, label: String, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clickable(onClick = onClick)) {
        Box(
            Modifier.size(48.dp).background(Color.White.copy(alpha = 0.12f), RoundedCornerShape(AppRadius.Sm)),
            contentAlignment = Alignment.Center,
        ) {
            Icon(icon, contentDescription = label, tint = Color.White, modifier = Modifier.size(22.dp))
        }
        Spacer(Modifier.height(4.dp))
        Text(label, color = Color.White.copy(alpha = 0.7f), fontSize = 11.5.sp)
    }
}

private fun hasCameraPermission(context: Context) =
    ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

internal fun newId(): String = System.currentTimeMillis().toString()
