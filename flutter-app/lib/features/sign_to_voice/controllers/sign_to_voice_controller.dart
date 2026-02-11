import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:logger/logger.dart';
import 'package:ai_voice_to_hand_signs_project/data/services/database_service.dart';
import 'package:ai_voice_to_hand_signs_project/data/models/session.dart';
import 'package:ai_voice_to_hand_signs_project/data/models/transcript_entry.dart';
import 'package:ai_voice_to_hand_signs_project/data/models/emotion_data.dart';

/// Controller for Sign-to-Voice feature
/// Manages detection results from Native Camera and integrates with RTDB
class SignToVoiceController extends GetxController {
  final Logger _logger = Logger();
  final DatabaseService _dbService = Get.find<DatabaseService>();

  // Observable state
  final RxBool isInitialized = true.obs; // Native view initializes itself
  final RxBool isRecording = false.obs;
  final RxBool isProcessing = false.obs;
  final RxString currentWord = ''.obs;
  final RxDouble confidence = 0.0.obs;
  final RxList<String> detectedWords = <String>[].obs;
  final RxString errorMessage = ''.obs;

  // Emotion detection state
  final Rx<EmotionData?> currentEmotion = Rx<EmotionData?>(null);
  final RxList<EmotionData> emotionHistory = <EmotionData>[].obs;

  // Session management
  String? _currentSessionId;
  String? get currentSessionId => _currentSessionId;

  // Debounce/Logic control
  DateTime? _lastWordTime;
  DateTime? _lastEmotionTime;
  static const wordDebounce = Duration(seconds: 1);
  static const emotionDebounce = Duration(seconds: 3);

  @override
  void onInit() {
    super.onInit();
    _logger.i('Sign-to-Voice controller initialized');
  }

  @override
  void onClose() {
    // End session if still active
    if (_currentSessionId != null && isRecording.value) {
      stopRecording();
    }
    super.onClose();
  }

  /// Start recording (listening to results) and create a session in RTDB
  Future<void> startRecording() async {
    detectedWords.clear();
    currentWord.value = '';
    emotionHistory.clear();
    currentEmotion.value = null;
    isRecording.value = true;

    try {
      // Create a new session in RTDB
      final user = FirebaseAuth.instance.currentUser;
      if (user != null) {
        _currentSessionId = await _dbService.createSession(
          user.uid,
          SessionType.signToVoice,
        );
        _logger.i('Recording started - Session: $_currentSessionId');
      } else {
        _logger.w('Recording started without authenticated user');
      }
    } catch (e) {
      _logger.e('Failed to create session: $e');
      errorMessage.value = 'Failed to start session';
    }
  }

  /// Stop recording and end the session
  Future<void> stopRecording() async {
    isRecording.value = false;
    currentWord.value = '';

    try {
      // End session in RTDB
      if (_currentSessionId != null) {
        final user = FirebaseAuth.instance.currentUser;
        if (user != null) {
          await _dbService.endSession(user.uid, _currentSessionId!);
          _logger.i('Recording stopped - Session ended: $_currentSessionId');
        }
        _currentSessionId = null;
      }
    } catch (e) {
      _logger.e('Failed to end session: $e');
    }
  }

  /// Handle sign detection result from Native Camera
  void onResult(String label, double score) {
    if (!isRecording.value) return;

    // Update UI for current detection
    currentWord.value = label;
    confidence.value = score;
    isProcessing.value = true;

    // Add to detected words if debounce passed and score is high enough
    if (score > 0.6) {
      final now = DateTime.now();
      if (_lastWordTime == null ||
          now.difference(_lastWordTime!) > wordDebounce) {
        if (detectedWords.isEmpty || detectedWords.last != label) {
          detectedWords.add(label);
          _lastWordTime = now;
          _logger.i('Detected: $label (${(score * 100).toStringAsFixed(1)}%)');

          // Save to RTDB as transcript entry
          _saveDetectedWord(label, score);
        }
      }
    }

    // Reset processing flag shortly
    Future.delayed(const Duration(milliseconds: 100), () {
      if (!isClosed) isProcessing.value = false;
    });
  }

  /// Handle emotion detection result (to be called from camera/ML module)
  void onEmotionDetected(EmotionType emotion, double confidence) {
    if (!isRecording.value) return;

    final now = DateTime.now();

    // Debounce emotion updates
    if (_lastEmotionTime == null ||
        now.difference(_lastEmotionTime!) > emotionDebounce) {
      final emotionData = EmotionData(
        emotion: emotion,
        confidence: confidence,
        timestamp: now.millisecondsSinceEpoch,
      );

      currentEmotion.value = emotionData;
      emotionHistory.add(emotionData);
      _lastEmotionTime = now;

      _logger.i(
        'Emotion detected: ${emotionData.label} (${(confidence * 100).toStringAsFixed(1)}%)',
      );

      // Save emotion to RTDB
      _saveEmotionData(emotionData);
    }
  }

  /// Save detected word to RTDB as transcript entry
  Future<void> _saveDetectedWord(String word, double confidence) async {
    if (_currentSessionId == null) return;

    try {
      final entry = TranscriptEntry(
        type: TranscriptType.rawGloss,
        content: word,
        timestamp: DateTime.now().millisecondsSinceEpoch,
        speakerRole: SpeakerRole.deafUser,
      );

      await _dbService.addTranscriptEntry(_currentSessionId!, entry);
      _logger.d('Saved word to RTDB: $word');
    } catch (e) {
      _logger.e('Failed to save word to RTDB: $e');
    }
  }

  /// Save emotion data to RTDB as transcript entry
  Future<void> _saveEmotionData(EmotionData emotionData) async {
    if (_currentSessionId == null) return;

    try {
      final entry = TranscriptEntry(
        type: TranscriptType.rawGloss,
        content:
            'Emotion: ${emotionData.label} ${emotionData.emoji} (${(emotionData.confidence * 100).toStringAsFixed(0)}%)',
        timestamp: emotionData.timestamp,
        speakerRole: SpeakerRole.deafUser,
      );

      await _dbService.addTranscriptEntry(_currentSessionId!, entry);
      _logger.d('Saved emotion to RTDB: ${emotionData.label}');
    } catch (e) {
      _logger.e('Failed to save emotion to RTDB: $e');
    }
  }

  /// Clear detected words and emotion history
  void clearWords() {
    detectedWords.clear();
    currentWord.value = '';
    confidence.value = 0.0;
    emotionHistory.clear();
    currentEmotion.value = null;
  }

  /// Get sentence from detected words
  String getSentence() {
    return detectedWords.join(' ');
  }

  /// Get dominant emotion from history
  EmotionType? getDominantEmotion() {
    if (emotionHistory.isEmpty) return null;

    // Count occurrences of each emotion
    final counts = <EmotionType, int>{};
    for (final emotion in emotionHistory) {
      counts[emotion.emotion] = (counts[emotion.emotion] ?? 0) + 1;
    }

    // Find the most frequent
    EmotionType? dominant;
    int maxCount = 0;
    counts.forEach((emotion, count) {
      if (count > maxCount) {
        maxCount = count;
        dominant = emotion;
      }
    });

    return dominant;
  }

  /// Get average confidence of detected emotions
  double getAverageEmotionConfidence() {
    if (emotionHistory.isEmpty) return 0.0;

    final sum = emotionHistory.fold<double>(
      0.0,
      (sum, emotion) => sum + emotion.confidence,
    );

    return sum / emotionHistory.length;
  }
}
