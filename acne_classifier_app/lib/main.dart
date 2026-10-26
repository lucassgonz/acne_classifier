import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'screens/home_shell.dart';
import 'theme/app_theme.dart';

List<CameraDescription> _cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    _cameras = await availableCameras();
  } catch (_) {
    // No camera available (e.g. running on a simulator without one
    // configured) — the capture screen handles an empty list gracefully.
    _cameras = [];
  }
  runApp(const AcneClassifierApp());
}

class AcneClassifierApp extends StatelessWidget {
  const AcneClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AcneApp',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      home: HomeShell(cameras: _cameras),
    );
  }
}
