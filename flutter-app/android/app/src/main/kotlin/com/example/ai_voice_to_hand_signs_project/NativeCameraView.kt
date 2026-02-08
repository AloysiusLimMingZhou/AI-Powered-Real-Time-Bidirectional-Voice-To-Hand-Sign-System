package com.example.ai_voice_to_hand_signs_project

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView
import java.util.concurrent.Executors

class NativeCameraView(
    private val context: Context,
    private val messenger: BinaryMessenger,
    viewId: Int,
    args: Any?
) : PlatformView, MethodChannel.MethodCallHandler {

    private val previewView: PreviewView = PreviewView(context)
    private val methodChannel: MethodChannel = MethodChannel(messenger, "com.example.sign_language/camera_$viewId")
    private val eventChannel: EventChannel = EventChannel(messenger, "com.example.sign_language/camera_events_$viewId")
    
    private var eventSink: EventChannel.EventSink? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    
    private var analyzer: SignLanguageAnalyzer? = null

    init {
        methodChannel.setMethodCallHandler(this)
        eventChannel.setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                eventSink = events
            }

            override fun onCancel(arguments: Any?) {
                eventSink = null
            }
        })
        
        startCamera()
    }

    override fun getView(): View {
        return previewView
    }

    override fun dispose() {
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        executor.shutdown()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return
        
        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Setup Analyzer
        analyzer = SignLanguageAnalyzer(context) { label, score ->
            mainHandler.post {
                eventSink?.success(mapOf(
                    "label" to label,
                    "score" to score
                ))
            }
        }
        
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // MediaPipe prefers RGBA or Bitmap
            .build()
            .also {
                it.setAnalyzer(executor, analyzer!!)
            }

        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

        try {
            cameraProvider.unbindAll()
             // Note: We need a LifecycleOwner. 
             // PlatformView doesn't provide it directly in older Flutter versions.
             // We cast context to LifecycleOwner if possible (Activity), 
             // but 'context' passed to PlatformView might be a wrapper.
             // 
             // The workaround is accessing the Activity from MainActivity or
             // passing the lifecycle owner. Use `FlutterActivity` which is a `LifecycleOwner`.
             // But context here is `ContextThemeWrapper` usually.
             
             // WE NEED TO PASS LIFECYCLE.
             // Best way: Use `ProcessLifecycleOwner` or get Activity.
             // Since we are in an Activity, let's try to cast context or find it.
             
             val lifecycleOwner = (context as? LifecycleOwner) 
                 ?: (context as? android.content.ContextWrapper)?.baseContext as? LifecycleOwner
                 // This is tricky.
                 // Ideally, we bind to the Activity's lifecycle.
                 
             if (lifecycleOwner != null) {
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
             } else {
                 Log.e("NativeCameraView", "Could not find LifecycleOwner")
             }
            
        } catch (exc: Exception) {
            Log.e("NativeCameraView", "Use case binding failed", exc)
        }
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "stop" -> {
                cameraProvider?.unbindAll()
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }
}
