import 'dart:io';

import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';

import '../models/analysis_record.dart';
import '../services/recommendations.dart';
import '../theme/app_theme.dart';
import '../widgets/app_card.dart';
import '../widgets/severity_scale.dart';

class ResultScreen extends StatelessWidget {
  final AnalysisRecord record;

  const ResultScreen({super.key, required this.record});

  @override
  Widget build(BuildContext context) {
    final overall = record.overallSeverity;
    final color = AppColors.forSeverity(overall);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Result'),
        leading: IconButton(
          icon: const Icon(Icons.close_rounded),
          onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.ios_share_rounded),
            onPressed: () => _share(),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(AppRadius.md),
                child: Image.file(File(record.imagePath), width: 112, height: 112, fit: BoxFit.cover),
              ),
              const SizedBox(width: AppSpacing.md),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Overall Acne Severity', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
                    Text(
                      kSeverityLabels[overall],
                      style: TextStyle(fontSize: 26, fontWeight: FontWeight.w700, color: color),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Based on the most affected of ${record.regions.length} analyzed regions',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: AppSpacing.lg),
          SeverityScale(level: overall),
          const SizedBox(height: AppSpacing.xl),
          Text('By region', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.sm),
          ...record.regions.map((r) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: _RegionResultCard(region: r),
              )),
          const SizedBox(height: AppSpacing.md),
          AppCard(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.description_outlined, color: AppColors.primary),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Recommendations', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15.5)),
                      const SizedBox(height: 6),
                      Text(kRecommendationsByLevel[overall].summary, style: Theme.of(context).textTheme.bodyMedium),
                      const SizedBox(height: 8),
                      ...kRecommendationsByLevel[overall].suggestions.map(
                            (s) => Padding(
                              padding: const EdgeInsets.only(bottom: 4),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text('•  '),
                                  Expanded(child: Text(s, style: Theme.of(context).textTheme.bodyMedium)),
                                ],
                              ),
                            ),
                          ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.md),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppColors.outline.withValues(alpha: 0.4),
              borderRadius: BorderRadius.circular(AppRadius.md),
            ),
            child: const Row(
              children: [
                Icon(Icons.info_outline_rounded, size: 18, color: AppColors.textSecondary),
                SizedBox(width: 10),
                Expanded(
                  child: Text(kProfessionalDisclaimer, style: TextStyle(fontSize: 12, color: AppColors.textSecondary)),
                ),
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.lg),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
              child: const Text('Back to Home'),
            ),
          ),
        ],
      ),
    );
  }

  void _share() {
    final lines = record.regions
        .map((r) => '${r.region}: ${kSeverityLabels[r.severityLevel]} (${(r.confidence * 100).toStringAsFixed(1)}%)')
        .join('\n');
    Share.share(
      'Acne severity assessment\n'
      'Overall: ${kSeverityLabels[record.overallSeverity]}\n\n$lines',
    );
  }
}

class _RegionResultCard extends StatelessWidget {
  final RegionRecord region;
  const _RegionResultCard({required this.region});

  @override
  Widget build(BuildContext context) {
    final color = AppColors.forSeverity(region.severityLevel);
    return AppCard(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 36,
            decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(AppRadius.pill)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(region.region, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14.5)),
                Text(
                  kSeverityLabels[region.severityLevel],
                  style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 15),
                ),
              ],
            ),
          ),
          Text(
            '${(region.confidence * 100).toStringAsFixed(1)}%',
            style: const TextStyle(fontWeight: FontWeight.w700, color: AppColors.textSecondary, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
