import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../theme/app_theme.dart';
import 'analyzing_screen.dart';

class CaptureScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CaptureScreen({super.key, required this.cameras});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  CameraController? _controller;
  Future<void>? _initFuture;
  int _cameraIndex = 0;
  bool _isCapturing = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    if (widget.cameras.isEmpty) {
      _error = 'No camera available on this device.';
      return;
    }
    _cameraIndex = widget.cameras.indexWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
    );
    if (_cameraIndex < 0) _cameraIndex = 0;
    _startCamera(_cameraIndex);
  }

  void _startCamera(int index) {
    final controller = CameraController(
      widget.cameras[index],
      ResolutionPreset.high,
      enableAudio: false,
    );
    setState(() {
      _controller = controller;
      _initFuture = controller.initialize().catchError((Object e) {
        setState(() => _error = 'Could not start the camera: $e');
      });
    });
  }

  Future<void> _flipCamera() async {
    if (widget.cameras.length < 2) return;
    final next = (_cameraIndex + 1) % widget.cameras.length;
    await _controller?.dispose();
    _cameraIndex = next;
    _startCamera(next);
  }

  Future<void> _takePicture() async {
    final controller = _controller;
    if (controller == null || _isCapturing) return;
    setState(() => _isCapturing = true);
    try {
      await _initFuture;
      final file = await controller.takePicture();
      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => AnalyzingScreen(imagePath: file.path)),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not capture the photo: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  Future<void> _pickFromGallery() async {
    final picked = await ImagePicker().pickImage(source: ImageSource.gallery, imageQuality: 92);
    if (picked == null || !mounted) return;
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => AnalyzingScreen(imagePath: picked.path)),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: AppColors.primary,
        title: const Text('Capture Image'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => _showTip(context),
          ),
        ],
      ),
      body: _error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(_error!, style: const TextStyle(color: Colors.white), textAlign: TextAlign.center),
                    const SizedBox(height: 20),
                    ElevatedButton.icon(
                      onPressed: _pickFromGallery,
                      icon: const Icon(Icons.photo_library_rounded),
                      label: const Text('Choose from Gallery'),
                    ),
                  ],
                ),
              ),
            )
          : FutureBuilder<void>(
              future: _initFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState != ConnectionState.done || _controller == null) {
                  return const Center(child: CircularProgressIndicator(color: Colors.white));
                }
                return Column(
                  children: [
                    Expanded(
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(_controller!),
                          const _FaceGuideOverlay(),
                          Positioned(
                            left: 0,
                            right: 0,
                            bottom: 20,
                            child: Text(
                              'Position your face within the guide\nand ensure good lighting.',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.white.withValues(alpha: 0.9),
                                fontSize: 14.5,
                                shadows: const [Shadow(blurRadius: 6, color: Colors.black)],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(vertical: 22),
                      color: Colors.black,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          _SquareIconButton(icon: Icons.photo_library_rounded, label: 'Gallery', onTap: _pickFromGallery),
                          GestureDetector(
                            onTap: _isCapturing ? null : _takePicture,
                            child: Container(
                              width: 76,
                              height: 76,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: AppColors.primaryLight,
                                border: Border.all(color: Colors.white, width: 4),
                              ),
                              child: _isCapturing
                                  ? const Padding(
                                      padding: EdgeInsets.all(22),
                                      child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.4),
                                    )
                                  : const Icon(Icons.camera_alt_rounded, color: Colors.white, size: 30),
                            ),
                          ),
                          _SquareIconButton(icon: Icons.flip_camera_ios_rounded, label: 'Flip', onTap: _flipCamera),
                        ],
                      ),
                    ),
                  ],
                );
              },
            ),
    );
  }

  void _showTip(BuildContext context) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(AppRadius.lg)),
      ),
      builder: (_) => const Padding(
        padding: EdgeInsets.all(28),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Tips for a good photo', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
            SizedBox(height: 12),
            Text('• Use even, natural lighting\n• Keep the face clean of makeup\n• Look straight at the camera\n• Fill the oval guide with your face'),
          ],
        ),
      ),
    );
  }
}

class _FaceGuideOverlay extends StatelessWidget {
  const _FaceGuideOverlay();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        width: 240,
        height: 310,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.white.withValues(alpha: 0.85), width: 2.2),
          borderRadius: BorderRadius.circular(130),
        ),
      ),
    );
  }
}

class _SquareIconButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _SquareIconButton({required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(AppRadius.sm),
            ),
            child: Icon(icon, color: Colors.white, size: 22),
          ),
          const SizedBox(height: 4),
          Text(label, style: const TextStyle(color: Colors.white70, fontSize: 11.5)),
        ],
      ),
    );
  }
}
