/// Severity result for a single anatomical region within one analysis.
class RegionRecord {
  final String region; // display label, e.g. "Forehead"
  final int severityLevel; // 0=Clear .. 3=Severe
  final double confidence; // 0..1

  const RegionRecord({
    required this.region,
    required this.severityLevel,
    required this.confidence,
  });

  Map<String, dynamic> toJson() => {
        'region': region,
        'severityLevel': severityLevel,
        'confidence': confidence,
      };

  factory RegionRecord.fromJson(Map<String, dynamic> json) => RegionRecord(
        region: json['region'] as String,
        severityLevel: json['severityLevel'] as int,
        confidence: (json['confidence'] as num).toDouble(),
      );
}

/// One full analysis: the captured photo plus a result for every detected
/// region, persisted locally so History has something to show (and tap
/// into) across app launches.
class AnalysisRecord {
  final String id;
  final String imagePath;
  final List<RegionRecord> regions;
  final DateTime createdAt;

  const AnalysisRecord({
    required this.id,
    required this.imagePath,
    required this.regions,
    required this.createdAt,
  });

  /// Overall severity follows the most affected region — the convention
  /// used when summarizing a multi-site Hayashi-style grading into one
  /// headline figure.
  int get overallSeverity =>
      regions.isEmpty ? 0 : regions.map((r) => r.severityLevel).reduce((a, b) => a > b ? a : b);

  double get overallConfidence {
    if (regions.isEmpty) return 0;
    final worst = overallSeverity;
    final matching = regions.where((r) => r.severityLevel == worst);
    return matching.map((r) => r.confidence).reduce((a, b) => a + b) / matching.length;
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'imagePath': imagePath,
        'regions': regions.map((r) => r.toJson()).toList(),
        'createdAt': createdAt.toIso8601String(),
      };

  factory AnalysisRecord.fromJson(Map<String, dynamic> json) => AnalysisRecord(
        id: json['id'] as String,
        imagePath: json['imagePath'] as String,
        regions: (json['regions'] as List)
            .map((e) => RegionRecord.fromJson(e as Map<String, dynamic>))
            .toList(),
        createdAt: DateTime.parse(json['createdAt'] as String),
      );
}
