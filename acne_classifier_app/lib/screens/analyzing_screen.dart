import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/analysis_record.dart';
import '../services/face_region_extractor.dart';
import '../services/history_service.dart';
import '../services/onnx_inference_service.dart';
import '../theme/app_theme.dart';
import 'result_screen.dart';

class AnalyzingScreen extends StatefulWidget {
  final String imagePath;

  const AnalyzingScreen({super.key, required this.imagePath});

  @override
  State<AnalyzingScreen> createState() => _AnalyzingScreenState();
}

class _AnalyzingScreenState extends State<AnalyzingScreen> with SingleTickerProviderStateMixin {
  static const _steps = ['Preprocessing', 'Extracting regions', 'Classification', 'Post-processing'];

  late final AnimationController _controller;
  int _stepIndex = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(seconds: 8))..forward();
    _controller.addListener(() => setState(() {}));
    _run();
  }

  Future<void> _run() async {
    final extractor = FaceRegionExtractor();
    final inference = OnnxInferenceService();
    final total = Stopwatch()..start();
    try {
      setState(() => _stepIndex = 0);
      final tLoad = Stopwatch()..start();
      await inference.ensureLoaded();
      debugPrint('BENCH model_load_ms=${tLoad.elapsedMilliseconds}');

      setState(() => _stepIndex = 1);
      final tExtract = Stopwatch()..start();
      final regions = await extractor.extract(widget.imagePath);
      debugPrint('BENCH face_extract_ms=${tExtract.elapsedMilliseconds} regions=${regions.length}');
      if (regions.isEmpty) {
        throw const NoFaceDetectedException(
          'A face was found, but no region could be cropped from this photo. '
          'Try a straight-on photo with the whole face visible.',
        );
      }

      setState(() => _stepIndex = 2);
      final predictions = <RegionRecord>[];
      for (final r in regions) {
        final tInfer = Stopwatch()..start();
        final pred = await inference.predict(r);
        debugPrint('BENCH inference_ms=${tInfer.elapsedMilliseconds} region=${r.region.label}');
        predictions.add(RegionRecord(
          region: pred.region.label,
          severityLevel: pred.severityLevel,
          confidence: pred.confidence,
        ));
      }
      debugPrint('BENCH total_pipeline_ms=${total.elapsedMilliseconds}');

      setState(() => _stepIndex = 3);
      final record = AnalysisRecord(
        id: '${DateTime.now().microsecondsSinceEpoch}',
        imagePath: widget.imagePath,
        regions: predictions,
        createdAt: DateTime.now(),
      );
      await HistoryService().add(record);

      await Future.delayed(const Duration(milliseconds: 300));
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => ResultScreen(record: record)),
      );
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      extractor.dispose();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Analyzing…')),
        body: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline_rounded, color: AppColors.severe, size: 48),
              const SizedBox(height: AppSpacing.md),
              Text('Could not complete the analysis', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(_error!, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: AppSpacing.xl),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('Try Again'),
                ),
              ),
            ],
          ),
        ),
      );
    }

    final percent = (_controller.value * 100).clamp(0, 99).round();
    return Scaffold(
      appBar: AppBar(title: const Text('Analyzing…')),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 24),
        child: Column(
          children: [
            const SizedBox(height: 24),
            SizedBox(
              width: 180,
              height: 180,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 180,
                    height: 180,
                    child: CircularProgressIndicator(
                      value: _controller.value,
                      strokeWidth: 10,
                      backgroundColor: AppColors.outline,
                      valueColor: const AlwaysStoppedAnimation(AppColors.primary),
                    ),
                  ),
                  Text(
                    '$percent%',
                    style: const TextStyle(fontSize: 34, fontWeight: FontWeight.w700, color: AppColors.primary),
                  ),
                ],
              ),
            ),
            const SizedBox(height: AppSpacing.xl),
            Text('Analyzing acne severity', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text('Running the on-device model across all regions…', style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(height: AppSpacing.xl),
            Column(
              children: List.generate(_steps.length, (i) {
                final done = i < _stepIndex;
                final active = i == _stepIndex;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          _steps[i],
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: active ? FontWeight.w600 : FontWeight.w400,
                            color: done || active ? AppColors.textPrimary : AppColors.textSecondary,
                          ),
                        ),
                      ),
                      if (done)
                        const Icon(Icons.check_circle_rounded, color: AppColors.primary, size: 22)
                      else if (active)
                        const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2.2, color: AppColors.primaryLight),
                        )
                      else
                        const Icon(Icons.radio_button_unchecked_rounded, color: AppColors.outline, size: 20),
                    ],
                  ),
                );
              }),
            ),
          ],
        ),
      ),
    );
  }
}
