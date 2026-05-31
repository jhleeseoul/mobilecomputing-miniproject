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
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

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
    private val inferenceIntervalMs: Long = 200L,
    private val waveformPoints: Int = 160,
    private val emaAlpha: Float = 0.6f,
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

    fun close() {
        val activeJob = readJob
        stop()
        if (activeJob != null) {
            runBlocking {
                withTimeoutOrNull(1500L) {
                    activeJob.join()
                }
            }
        }
        spotter.close()
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
        var emaScores: FloatArray? = null
        var rmsEma = 0f
        var agcGain = 1f

        try {
            while (kotlinx.coroutines.currentCoroutineContext().isActive) {
                val read = audioRecord.read(chunkShort, 0, chunkShort.size)
                if (read <= 0) continue

                for (i in 0 until read) {
                    chunkFloat[i] = chunkShort[i] / 32768f
                }
                agcGain = applyAgcInPlace(chunkFloat, read, previousGain = agcGain, rmsEmaSeed = rmsEma)
                rmsEma = updateRmsEma(chunkFloat, read, previous = rmsEma)
                filled = appendToRolling(rolling, filled, chunkFloat, read)

                val waveform = downsample(rolling, if (filled == windowSamples) windowSamples else filled)
                val nowMs = SystemClock.elapsedRealtime()

                if (filled == windowSamples && nowMs - lastInferenceAtMs >= inferenceIntervalMs) {
                    val rawResult = spotter.predict(rolling)
                    val stabilizedScores = emaBlend(emaScores, rawResult.scores, emaAlpha)
                    emaScores = stabilizedScores
                    val topIdx = argmax(stabilizedScores)
                    val topScore = if (topIdx >= 0) stabilizedScores[topIdx] else 0f
                    val topLabel = labels.getOrElse(topIdx.coerceAtLeast(0)) { "class_$topIdx" }

                    val stabilizedResult = rawResult.copy(
                        topLabel = topLabel,
                        topScore = topScore,
                        scores = stabilizedScores,
                    )

                    _state.update {
                        it.copy(
                            waveform = waveform,
                            prediction = stabilizedResult,
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

    private fun emaBlend(previous: FloatArray?, current: FloatArray, alpha: Float): FloatArray {
        if (previous == null || previous.size != current.size) {
            return current.copyOf()
        }
        val out = FloatArray(current.size)
        val a = alpha.coerceIn(0.0f, 1.0f)
        val b = 1.0f - a
        for (i in current.indices) {
            out[i] = current[i] * a + previous[i] * b
        }
        return out
    }

    private fun applyAgcInPlace(
        chunk: FloatArray,
        valid: Int,
        previousGain: Float,
        rmsEmaSeed: Float,
    ): Float {
        if (valid <= 0) return previousGain
        var sumSq = 0.0
        var clippedCount = 0
        for (i in 0 until valid) {
            val v = chunk[i]
            sumSq += (v * v).toDouble()
            if (abs(v) >= 0.98f) {
                clippedCount++
            }
        }
        val chunkRms = sqrt((sumSq / valid.toDouble()).coerceAtLeast(0.0)).toFloat()
        val rmsEma = if (rmsEmaSeed <= 0f) chunkRms else 0.92f * rmsEmaSeed + 0.08f * chunkRms

        val targetRms = 0.06f
        val minRms = 1e-3f
        val noiseFloor = 0.004f
        val desiredGain = if (rmsEma < noiseFloor) {
            1.0f
        } else {
            (targetRms / rmsEma.coerceAtLeast(minRms)).coerceIn(1.0f, 4.0f)
        }
        var gain = (0.8f * previousGain + 0.2f * desiredGain).coerceIn(1.0f, 4.0f)

        val clippedRatio = clippedCount.toFloat() / valid.toFloat()
        if (clippedRatio > 0.01f) {
            gain = (gain * 0.85f).coerceAtLeast(1.0f)
        }

        for (i in 0 until valid) {
            chunk[i] = (chunk[i] * gain).coerceIn(-1f, 1f)
        }
        return gain
    }

    private fun updateRmsEma(chunk: FloatArray, valid: Int, previous: Float): Float {
        if (valid <= 0) return previous
        var sumSq = 0.0
        for (i in 0 until valid) {
            val v = chunk[i]
            sumSq += (v * v).toDouble()
        }
        val rms = sqrt((sumSq / valid.toDouble()).coerceAtLeast(0.0)).toFloat()
        return if (previous <= 0f) rms else 0.92f * previous + 0.08f * rms
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
