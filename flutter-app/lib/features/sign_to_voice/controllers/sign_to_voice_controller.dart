import 'package:get/get.dart';
import 'package:logger/logger.dart';

/// Controller for Sign-to-Voice feature
/// Manages detection results from Native Camera
class SignToVoiceController extends GetxController {
  final Logger _logger = Logger();

  // Observable state
  final RxBool isInitialized = true.obs; // Native view initializes itself
  final RxBool isRecording = false.obs;
  final RxBool isProcessing = false.obs;
  final RxString currentWord = ''.obs;
  final RxDouble confidence = 0.0.obs;
  final RxList<String> detectedWords = <String>[].obs;
  final RxString errorMessage = ''.obs;

  // Debounce/Logic control
  DateTime? _lastWordTime;
  static const wordDebounce = Duration(seconds: 1);

  @override
  void onInit() {
    super.onInit();
    _logger.i('Sign-to-Voice controller initialized');
  }

  /// Start recording (listening to results)
  void startRecording() {
    detectedWords.clear();
    currentWord.value = '';
    isRecording.value = true;
    _logger.i('Recording started');
  }

  /// Stop recording
  void stopRecording() {
    isRecording.value = false;
    currentWord.value = '';
    _logger.i('Recording stopped');
  }

  /// Handle result from Native Camera
  void onResult(String label, double score) {
    if (!isRecording.value) return;

    // Update UI for current detection
    currentWord.value = label;
    confidence.value = score;
    isProcessing.value = true; // Just to show activity

    // Add to detected words if debounce passed and score is high enough
    // Logic: if label stays the same for X frames?
    // Or just simply add if it's different from last added?
    // Let's implement a simple logic: if score > 0.8 and (different from last OR time passed)

    if (score > 0.6) {
      final now = DateTime.now();
      if (_lastWordTime == null ||
          now.difference(_lastWordTime!) > wordDebounce) {
        if (detectedWords.isEmpty || detectedWords.last != label) {
          detectedWords.add(label);
          _lastWordTime = now;
          _logger.i('Detected: $label (${(score * 100).toStringAsFixed(1)}%)');
        }
      }
    }

    // Reset processing flag shortly
    Future.delayed(const Duration(milliseconds: 100), () {
      if (!isClosed) isProcessing.value = false;
    });
  }

  /// Clear detected words
  void clearWords() {
    detectedWords.clear();
    currentWord.value = '';
    confidence.value = 0.0;
  }

  /// Get sentence from detected words
  String getSentence() {
    return detectedWords.join(' ');
  }
}
