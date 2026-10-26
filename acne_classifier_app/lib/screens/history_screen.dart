import 'dart:io';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/analysis_record.dart';
import '../services/history_service.dart';
import '../theme/app_theme.dart';
import '../widgets/app_card.dart';
import 'result_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => HistoryScreenState();
}

class HistoryScreenState extends State<HistoryScreen> {
  final _service = HistoryService();
  late Future<List<AnalysisRecord>> _future;

  @override 
  void initState() {
    super.initState();
    _future = _service.load();
  }

  /// Called by the bottom-nav shell when this tab becomes visible again,
  /// so a result just saved from the flow above shows up immediately.
  void refresh() => setState(() {
        _future = _service.load();
      });

  Future<void> _openDetail(AnalysisRecord record) async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => ResultScreen(record: record)),
    );
    refresh();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: FutureBuilder<List<AnalysisRecord>>(
        future: _future,
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final records = snapshot.data!;
          if (records.isEmpty) {
            return const _EmptyState();
          }
          return RefreshIndicator(
            onRefresh: () async => refresh(),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: records.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (context, i) => _HistoryTile(
                record: records[i],
                onTap: () => _openDetail(records[i]),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  final AnalysisRecord record;
  final VoidCallback onTap;
  const _HistoryTile({required this.record, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final overall = record.overallSeverity;
    final color = AppColors.forSeverity(overall);
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppRadius.md),
      child: AppCard(
        padding: const EdgeInsets.all(10),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(AppRadius.sm),
              child: Image.file(
                File(record.imagePath),
                width: 56,
                height: 56,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Container(
                  width: 56,
                  height: 56,
                  color: AppColors.outline,
                  child: const Icon(Icons.image_not_supported_outlined, color: AppColors.textSecondary),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('${record.regions.length} regions analyzed',
                      style: const TextStyle(color: AppColors.textSecondary, fontSize: 12.5)),
                  Text(
                    kSeverityLabels[overall],
                    style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 16),
                  ),
                  Text(
                    DateFormat('MMM d, y  HH:mm').format(record.createdAt),
                    style: const TextStyle(color: AppColors.textSecondary, fontSize: 12),
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(AppRadius.pill),
              ),
              child: Text(
                '${(record.overallConfidence * 100).toStringAsFixed(1)}%',
                style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 12.5),
              ),
            ),
            const SizedBox(width: 6),
            const Icon(Icons.chevron_right_rounded, color: AppColors.textSecondary),
          ],
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.history_rounded, size: 48, color: AppColors.textSecondary),
            const SizedBox(height: AppSpacing.md),
            Text('No analyses yet', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 6),
            Text(
              'Results you analyze will show up here.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }
}
