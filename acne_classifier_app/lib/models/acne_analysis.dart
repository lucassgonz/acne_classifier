
class AcneAnalysis {
  final String id;
  final String userId;
  final String frontImageUrl;
  final String rightSideImageUrl;
  final String leftSideImageUrl;
  final Map<String, dynamic> classificationResult;
  final String severityLevel;
  final double? confidenceScore;
  final String analysisStatus;
  final int? processingTimeSeconds;
  final String modelVersion;
  final DateTime createdAt;
  final DateTime? completedAt;

  AcneAnalysis({
    required this.id,
    required this.userId,
    required this.frontImageUrl,
    required this.rightSideImageUrl,
    required this.leftSideImageUrl,
    required this.classificationResult,
    required this.severityLevel,
    this.confidenceScore,
    required this.analysisStatus,
    this.processingTimeSeconds,
    required this.modelVersion,
    required this.createdAt,
    this.completedAt,
  });

  factory AcneAnalysis.fromJson(Map<String, dynamic> json) {
    return AcneAnalysis(
      id: json['id'],
      userId: json['user_id'],
      frontImageUrl: json['front_image_url'],
      rightSideImageUrl: json['right_side_image_url'],
      leftSideImageUrl: json['left_side_image_url'],
      classificationResult: json['classification_result'],
      severityLevel: json['severity_level'],
      confidenceScore: json['confidence_score']?.toDouble(),
      analysisStatus: json['analysis_status'],
      processingTimeSeconds: json['processing_time_seconds'],
      modelVersion: json['model_version'],
      createdAt: DateTime.parse(json['created_at']),
      completedAt: json['completed_at'] != null 
          ? DateTime.parse(json['completed_at']) 
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'user_id': userId,
      'front_image_url': frontImageUrl,
      'right_side_image_url': rightSideImageUrl,
      'left_side_image_url': leftSideImageUrl,
      'classification_result': classificationResult,
      'severity_level': severityLevel,
      'confidence_score': confidenceScore,
      'analysis_status': analysisStatus,
      'processing_time_seconds': processingTimeSeconds,
      'model_version': modelVersion,
      'completed_at': completedAt?.toIso8601String(),
    };
  }
}