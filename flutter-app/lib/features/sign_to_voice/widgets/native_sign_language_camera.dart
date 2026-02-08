import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';

class NativeSignLanguageCamera extends StatefulWidget {
  final Function(String label, double score) onResult;

  const NativeSignLanguageCamera({Key? key, required this.onResult})
    : super(key: key);

  @override
  State<NativeSignLanguageCamera> createState() =>
      _NativeSignLanguageCameraState();
}

class _NativeSignLanguageCameraState extends State<NativeSignLanguageCamera> {
  // Unique channel names per view could be managed if needed, but for singleton camera use case, fixed name pattern is okay-ish if viewId is used.
  // The Kotlin side expects "com.example.sign_language/camera_$viewId"

  MethodChannel? _methodChannel;
  EventChannel? _eventChannel;

  @override
  Widget build(BuildContext context) {
    // Check for Android
    if (defaultTargetPlatform != TargetPlatform.android) {
      return const Center(child: Text("Only Android is supported"));
    }

    const String viewType = 'camera_view';
    final Map<String, dynamic> creationParams = <String, dynamic>{};

    return AndroidView(
      viewType: viewType,
      layoutDirection: TextDirection.ltr,
      creationParams: creationParams,
      creationParamsCodec: const StandardMessageCodec(),
      onPlatformViewCreated: _onPlatformViewCreated,
    );
  }

  void _onPlatformViewCreated(int id) {
    _methodChannel = MethodChannel('com.example.sign_language/camera_$id');
    _eventChannel = EventChannel('com.example.sign_language/camera_events_$id');

    _eventChannel?.receiveBroadcastStream().listen(
      (event) {
        if (mounted) {
          if (event is Map) {
            final label = event['label']?.toString();
            final score = event['score'] is num
                ? (event['score'] as num).toDouble()
                : 0.0;

            if (label != null) {
              widget.onResult(label, score);
            }
          }
        }
      },
      onError: (error) {
        debugPrint("Error in native camera stream: $error");
      },
    );
  }

  @override
  void dispose() {
    _methodChannel?.invokeMethod('stop');
    super.dispose();
  }
}
