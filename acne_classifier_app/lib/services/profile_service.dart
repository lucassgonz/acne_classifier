import 'dart:io';

import 'package:acne_classifier_app/models/user_profile.dart';
import 'package:acne_classifier_app/services/supabase_service.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class ProfileService {
  final SupabaseClient _client = SupabaseService.instance.client;

  Future<UserProfile?> getUserProfile(String userId) async {
    try {
      final response = await _client
          .from('profiles')
          .select()
          .eq('id', userId)
          .single();

      return UserProfile.fromJson(response);
    } catch (e) {
      print('Erro ao buscar perfil: $e');
      return null;
    }
  }

  Future<bool> updateUserProfile(UserProfile profile) async {
    try {
      await _client
          .from('profiles')
          .update(profile.toJson())
          .eq('id', profile.id);

      return true;
    } catch (e) {
      print('Erro ao atualizar perfil: $e');
      return false;
    }
  }

  Future<String?> uploadAvatar(String userId, File imageFile) async {
    try {
      final fileName = 'avatar_${userId}_${DateTime.now().millisecondsSinceEpoch}.jpg';

      await _client.storage
          .from('avatars')
          .upload(fileName, imageFile);

      final publicUrl = _client.storage
          .from('avatars')
          .getPublicUrl(fileName);

      return publicUrl;
    } catch (e) {
      print('Erro ao fazer upload do avatar: $e');
      return null;
    }
  }
}
