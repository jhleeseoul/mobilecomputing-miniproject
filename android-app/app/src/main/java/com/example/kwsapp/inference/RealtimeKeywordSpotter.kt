package com.example.kwsapp.inference

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlin.math.max

data class StreamingInferenceState(
    val isListening: Boolean = false,
    val waveform: FloatArray = FloatArray(160),
    val prediction: PredictionResult? = null,
    val predictionTimestampMs: Long = 0L,
    val errorMessage: String? = null,
)

class RealtimeKeywordSpotter(
    private val spotter: KeywordSpotter,
    private val labels: List<String>,
    private val sampleRate: Int = 16000,
    private val windowSamples: Int = 16000,
    private val inferenceIntervalMs: Long = 320L,
    private val waveformPoints: Int = 160,
    private val smoothingWindow: Int = 3,
) {
    private val _state = MutableStateFlow(StreamingInferenceState())
    val state: StateFlow<StreamingInferenceState> = _state.asStateFlow()

    private val lifecycleMutex = Mutex()
    private var readJob: Job? = null
    private var recorder: AudioRecord? = null

    fun start(scope: CoroutineScope) {
        if (readJob?.isActive == true) return
        readJob = scope.launch(Dispatchers.Default) {
            try {
                lifecycleMutex.withLock {
                    if (recorder != null) return@withLock
                    val audio = createAudioRecord()
                    recorder = audio
                    audio.startRecording()
                    if (audio.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                        audio.release()
                        recorder = null
                        _state.update {
                            it.copy(
                                isListening = false,
                                errorMessage = "Microphone start failed",
                            )
                        }
                        return@withLock
                    }
                    _state.update { it.copy(isListening = true, errorMessage = null) }
                }

                val activeRecorder = recorder ?: return@launch
                runCaptureLoop(activeRecorder)
            } catch (t: Throwable) {
                _state.update {
                    it.copy(
                        isListening = false,
                        errorMessage = t.message ?: "Microphone initialization failed",
                    )
                }
                releaseRecorder()
            }
        }.also { job ->
            job.invokeOnCompletion {
                if (readJob === job) {
                    readJob = null
                }
            }
        }
    }

    fun stop() {
        readJob?.cancel()
        readJob = null
        releaseRecorder()
        _state.update { it.copy(isListening = false) }
    }

    private fun createAudioRecord(): AudioRecord {
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        require(minBufferSize > 0) { "Invalid min buffer size: $minBufferSize" }

        val bufferSize = max(minBufferSize, 4096)
        val audio = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        )
        require(audio.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord init failed" }
        return audio
    }

    private suspend fun runCaptureLoop(audioRecord: AudioRecord) {
        val rolling = FloatArray(windowSamples)
        val chunkShort = ShortArray(1024)
        val chunkFloat = FloatArray(1024)
        var filled = 0
        var lastInferenceAtMs = 0L
        val smoothQueue = ArrayDeque<FloatArray>()

        try {
            while (kotlinx.coroutines.currentCoroutineContext().isActive) {
                val read = audioRecord.read(chunkShort, 0, chunkShort.size)
                if (read <= 0) continue

                for (i in 0 until read) {
                    chunkFloat[i] = chunkShort[i] / 32768f
                }
                filled = appendToRolling(rolling, filled, chunkFloat, read)

                val waveform = downsample(rolling, if (filled == windowSamples) windowSamples else filled)
                val nowMs = SystemClock.elapsedRealtime()

                if (filled == windowSamples && nowMs - lastInferenceAtMs >= inferenceIntervalMs) {
                    val rawResult = spotter.predict(rolling.copyOf())
                    val smoothedScores = smoothScores(smoothQueue, rawResult.scores)
                    val topIdx = argmax(smoothedScores)
                    val topScore = if (topIdx >= 0) smoothedScores[topIdx] else 0f
                    val topLabel = labels.getOrElse(topIdx.coerceAtLeast(0)) { "class_$topIdx" }

                    val smoothedResult = rawResult.copy(
                        topLabel = topLabel,
                        topScore = topScore,
                        scores = smoothedScores,
                    )

                    _state.update {
                        it.copy(
                            waveform = waveform,
                            prediction = smoothedResult,
                            predictionTimestampMs = System.currentTimeMillis(),
                            errorMessage = null,
                        )
                    }
                    lastInferenceAtMs = nowMs
                } else {
                    _state.update {
                        it.copy(
                            waveform = waveform,
                            errorMessage = null,
                        )
                    }
                }
            }
        } catch (t: Throwable) {
            _state.update {
                it.copy(
                    isListening = false,
                    errorMessage = t.message ?: "Realtime capture failed",
                )
            }
        } finally {
            releaseRecorder()
            _state.update { it.copy(isListening = false) }
        }
    }

    private fun appendToRolling(
        rolling: FloatArray,
        filled: Int,
        chunk: FloatArray,
        read: Int,
    ): Int {
        if (read >= rolling.size) {
            val start = read - rolling.size
            System.arraycopy(chunk, start, rolling, 0, rolling.size)
            return rolling.size
        }

        if (filled < rolling.size) {
            val writable = minOf(rolling.size - filled, read)
            System.arraycopy(chunk, 0, rolling, filled, writable)
            val nextFilled = filled + writable
            if (writable == read) return nextFilled

            val remain = read - writable
            System.arraycopy(rolling, remain, rolling, 0, rolling.size - remain)
            System.arraycopy(chunk, writable, rolling, rolling.size - remain, remain)
            return rolling.size
        }

        System.arraycopy(rolling, read, rolling, 0, rolling.size - read)
        System.arraycopy(chunk, 0, rolling, rolling.size - read, read)
        return rolling.size
    }

    private fun downsample(rolling: FloatArray, validSize: Int): FloatArray {
        if (validSize <= 0) return FloatArray(waveformPoints)
        val out = FloatArray(waveformPoints)
        val step = validSize.toFloat() / waveformPoints.toFloat()
        for (i in 0 until waveformPoints) {
            val idx = (i * step).toInt().coerceIn(0, validSize - 1)
            out[i] = rolling[idx]
        }
        return out
    }

    private fun smoothScores(queue: ArrayDeque<FloatArray>, scores: FloatArray): FloatArray {
        queue.addLast(scores.copyOf())
        while (queue.size > smoothingWindow) {
            queue.removeFirst()
        }

        val out = FloatArray(scores.size)
        for (item in queue) {
            for (i in out.indices) {
                out[i] += item[i]
            }
        }
        for (i in out.indices) {
            out[i] /= queue.size.toFloat()
        }
        return out
    }

    private fun argmax(values: FloatArray): Int {
        if (values.isEmpty()) return -1
        var bestIdx = 0
        var best = values[0]
        for (i in 1 until values.size) {
            if (values[i] > best) {
                best = values[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    private fun releaseRecorder() {
        val local = recorder
        recorder = null
        if (local != null) {
            runCatching { local.stop() }
            runCatching { local.release() }
        }
    }
}
