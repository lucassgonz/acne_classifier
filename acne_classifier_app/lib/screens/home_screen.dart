import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../theme/app_theme.dart';
import 'capture_screen.dart';

class HomeScreen extends StatelessWidget {
  final List<CameraDescription> cameras;
  final VoidCallback onViewHistory;

  const HomeScreen({super.key, required this.cameras, required this.onViewHistory});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: AppColors.homeGradient,
        ),
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(28, 0, 28, 24),
          child: Column(
            children: [
              const Spacer(flex: 3),
              Container(
                width: 92,
                height: 92,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white.withValues(alpha: 0.55), width: 1.4),
                ),
                child: const Icon(Icons.face_retouching_natural, color: Colors.white, size: 42),
              ),
              const SizedBox(height: AppSpacing.lg),
              const Text(
                'AcneApp',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 32,
                  fontWeight: FontWeight.w700,
                  height: 1.25,
                ),
              ),
              const SizedBox(height: AppSpacing.xs),
              Text(
                'Automatic evaluation of facial\nacne severity using artificial intelligence',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.85),
                  fontSize: 15,
                  height: 1.4,
                ),
              ),
              const Spacer(flex: 4),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => CaptureScreen(cameras: cameras)),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primaryDark,
                    foregroundColor: Colors.white,
                    elevation: 0,
                  ),
                  icon: const Icon(Icons.camera_alt_rounded, size: 20),
                  label: const Text('Start Analysis'),
                ),
              ),
              const SizedBox(height: AppSpacing.md),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: onViewHistory,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: AppColors.primaryDark,
                    elevation: 0,
                  ),
                  icon: const Icon(Icons.schedule_rounded, size: 20),
                  label: const Text('History'),
                ),
              ),
              const SizedBox(height: AppSpacing.lg),
              Text(
                'This is a research prototype and does not replace\nprofessional dermatological advice.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white.withValues(alpha: 0.72), fontSize: 12.5, height: 1.4),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
