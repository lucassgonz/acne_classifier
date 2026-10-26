import 'package:flutter/material.dart';

import '../theme/app_theme.dart';
import '../widgets/app_card.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('About')),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          Center(
            child: Container(
              width: 72,
              height: 72,
              decoration: const BoxDecoration(color: AppColors.primary, shape: BoxShape.circle),
              child: const Icon(Icons.face_retouching_natural, color: Colors.white, size: 32),
            ),
          ),
          const SizedBox(height: AppSpacing.md),
          Text(
            'AcneApp',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 4),
          Text(
            'Research prototype',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: AppSpacing.xl),
          const _InfoCard(
            icon: Icons.science_outlined,
            title: 'About this prototype',
            body: 'This application runs a region-aware ResNet-18 ensemble entirely on-device: a photo '
                'is captured, the face and five anatomical regions are located automatically, and each '
                'region is scored on the four-level Hayashi scale (Clear, Mild, Moderate, Severe), '
                'then combined into an overall result.',
          ),
          const SizedBox(height: AppSpacing.md),
          const _InfoCard(
            icon: Icons.warning_amber_rounded,
            title: 'Not a medical device',
            body: 'Results are informational only and do not replace a diagnosis from a qualified '
                'dermatologist. Always seek professional advice for treatment decisions.',
          ),
          const SizedBox(height: AppSpacing.md),
          const _InfoCard(
            icon: Icons.privacy_tip_outlined,
            title: 'Your data',
            body: 'Photos and results stay on this device. Nothing is uploaded to a server by this app.',
          ),
        ],
      ),
    );
  }
}

class _InfoCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String body;

  const _InfoCard({required this.icon, required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    return AppCard(
      child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: AppColors.primary),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15.5)),
                  const SizedBox(height: 6),
                  Text(body, style: Theme.of(context).textTheme.bodyMedium),
                ],
              ),
            ),
          ],
        ),
    );
  }
}
