import 'dart:io';

import 'package:acne_classifier_app/models/acne_analysis.dart';
import 'package:acne_classifier_app/services/supabase_service.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class AnalysisService {
  final SupabaseClient _client = SupabaseService.instance.client;

  Future<List<String>?> uploadAnalysisImages({
    required String userId,
    required List<File> images,
  }) async {
    try {
      List<String> imageUrls = [];
      final timestamp = DateTime.now().millisecondsSinceEpoch;

      for (int i = 0; i < images.length; i++) {
        final fileName = 'analysis_${userId}_${timestamp}_${i}.jpg';

        await _client.storage
            .from('analysis-images')
            .upload(fileName, images[i]);

        final publicUrl = _client.storage
            .from('analysis-images')
            .getPublicUrl(fileName);

        imageUrls.add(publicUrl);
      }

      return imageUrls;
    } catch (e) {
      print('Erro ao fazer upload das imagens: $e');
      return null;
    }
  }

  Future<String?> createAnalysis({
    required String userId,
    required List<String> imageUrls,
  }) async {
    try {
      final response = await _client
          .from('acne_analyses')
          .insert({
            'user_id': userId,
            'front_image_url': imageUrls[0],
            'right_side_image_url': imageUrls[1],
            'left_side_image_url': imageUrls[2],
            'classification_result': {},
            'severity_level': 'none',
            'analysis_status': 'pending',
            'model_version': 'v1.0',
          })
          .select('id')
          .single();

      return response['id'];
    } catch (e) {
      print('Erro ao criar análise: $e');
      return null;
    }
  }

  Future<bool> updateAnalysisResult({
    required String analysisId,
    required Map<String, dynamic> result,
    required String severityLevel,
    required double confidenceScore,
  }) async {
    try {
      await _client
          .from('acne_analyses')
          .update({
            'classification_result': result,
            'severity_level': severityLevel,
            'confidence_score': confidenceScore,
            'analysis_status': 'completed',
            'completed_at': DateTime.now().toIso8601String(),
          })
          .eq('id', analysisId);

      return true;
    } catch (e) {
      print('Erro ao atualizar análise: $e');
      return false;
    }
  }

  Future<List<AcneAnalysis>> getUserAnalyses(String userId) async {
    try {
      final response = await _client
          .from('acne_analyses')
          .select()
          .eq('user_id', userId)
          .order('created_at', ascending: false);

      return response.map((json) => AcneAnalysis.fromJson(json)).toList();
    } catch (e) {
      print('Erro ao buscar análises: $e');
      return [];
    }
  }

  Future<AcneAnalysis?> getAnalysis(String analysisId) async {
    try {
      final response = await _client
          .from('acne_analyses')
          .select()
          .eq('id', analysisId)
          .single();

      return AcneAnalysis.fromJson(response);
    } catch (e) {
      print('Erro ao buscar análise: $e');
      return null;
    }
  }
}
