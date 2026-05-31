package com.example.kwsapp.inference

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.abs

class KeywordSpotter(
    private val context: Context,
    private val modelAssetPath: String = "model_int8.tflite",
    private val labels: List<String> = listOf("yes", "no", "up", "down", "left", "right", "stop", "go", "unknown", "silence"),
) : Closeable {
    private val mfccExtractor = MfccExtractor()

    private val interpreterDelegate = lazy {
        Interpreter(loadModelFile(context, modelAssetPath), Interpreter.Options().apply { setNumThreads(2) })
    }
    private val interpreter: Interpreter get() = interpreterDelegate.value

    data class ModelIO(
        val inputShape: IntArray,
        val outputShape: IntArray,
        val inputScale: Float,
        val inputZeroPoint: Int,
        val outputScale: Float,
        val outputZeroPoint: Int,
        val inputIsInt8: Boolean,
        val outputIsInt8: Boolean,
    )

    private val io: ModelIO by lazy {
        val inputDetails = interpreter.getInputTensor(0)
        val outputDetails = interpreter.getOutputTensor(0)

        val inputQuant = inputDetails.quantizationParams()
        val outputQuant = outputDetails.quantizationParams()

        ModelIO(
            inputShape = inputDetails.shape(),
            outputShape = outputDetails.shape(),
            inputScale = inputQuant.scale,
            inputZeroPoint = inputQuant.zeroPoint,
            outputScale = outputQuant.scale,
            outputZeroPoint = outputQuant.zeroPoint,
            inputIsInt8 = inputDetails.dataType().name == "INT8",
            outputIsInt8 = outputDetails.dataType().name == "INT8",
        )
    }

    private val inputElementCount: Int by lazy { io.inputShape.drop(1).reduce { acc, value -> acc * value } }
    private val outputElementCount: Int by lazy { io.outputShape.last() }

    private val inputBuffer: ByteBuffer by lazy {
        val bytes = if (io.inputIsInt8) inputElementCount else inputElementCount * 4
        ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
    }
    private val outputBufferInt8: Array<ByteArray> by lazy { Array(1) { ByteArray(outputElementCount) } }
    private val outputBufferFloat: Array<FloatArray> by lazy { Array(1) { FloatArray(outputElementCount) } }
    private val outputScratch: FloatArray by lazy { FloatArray(outputElementCount) }

    fun predict(featureOrAudio: FloatArray): PredictionResult {
        val inputVector = toModelInput(featureOrAudio)
        val startNs = System.nanoTime()

        if (io.outputIsInt8) {
            interpreter.run(writeInputToBuffer(inputVector), outputBufferInt8)
            dequantizeOutputInto(outputBufferInt8[0], outputScratch)
        } else {
            interpreter.run(writeInputToBuffer(inputVector), outputBufferFloat)
            System.arraycopy(outputBufferFloat[0], 0, outputScratch, 0, outputElementCount)
        }

        val scores = outputScratch.copyOf()
        if (!isProbabilityDistribution(scores)) {
            softmaxInPlace(scores)
        }

        val latencyMs = (System.nanoTime() - startNs) / 1_000_000f
        val topIndex = scores.indices.maxByOrNull { scores[it] } ?: 0

        return PredictionResult(
            topLabel = labels.getOrElse(topIndex) { "class_$topIndex" },
            topScore = scores[topIndex],
            scores = scores,
            latencyMs = latencyMs,
        )
    }

    private fun toModelInput(featureOrAudio: FloatArray): FloatArray {
        return when {
            featureOrAudio.size == inputElementCount -> featureOrAudio
            featureOrAudio.size >= 8000 -> {
                val frames = mfccExtractor.extractFromAudio(featureOrAudio, audioSampleRate = 16000)
                flatten(frames)
            }
            else -> throw IllegalArgumentException(
                "Input length ${featureOrAudio.size} does not match model feature size $inputElementCount"
            )
        }
    }

    private fun flatten(features: Array<FloatArray>): FloatArray {
        val inputShape = io.inputShape
        val expectedFrames = inputShape[1]
        val expectedBins = inputShape[2]
        val flattened = FloatArray(expectedFrames * expectedBins)

        for (f in 0 until expectedFrames) {
            for (b in 0 until expectedBins) {
                val value = if (f < features.size && b < features[f].size) features[f][b] else 0f
                flattened[f * expectedBins + b] = value
            }
        }
        return flattened
    }

    private fun writeInputToBuffer(flattenedFeatures: FloatArray): ByteBuffer {
        val buffer = inputBuffer
        buffer.clear()
        if (io.inputIsInt8) {
            require(io.inputScale != 0f) { "Invalid int8 input quantization scale" }
            for (v in flattenedFeatures) {
                val q = (v / io.inputScale + io.inputZeroPoint).toInt().coerceIn(-128, 127)
                buffer.put(q.toByte())
            }
        } else {
            for (v in flattenedFeatures) {
                buffer.putFloat(v)
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun dequantizeOutputInto(raw: ByteArray, out: FloatArray) {
        require(io.outputScale != 0f) { "Invalid int8 output quantization scale" }
        for (i in raw.indices) {
            out[i] = (raw[i].toInt() - io.outputZeroPoint) * io.outputScale
        }
    }

    private fun isProbabilityDistribution(values: FloatArray): Boolean {
        if (values.isEmpty()) return false
        var sum = 0f
        for (value in values) {
            if (!value.isFinite()) return false
            if (value < -1e-3f || value > 1.001f) return false
            sum += value
        }
        return abs(sum - 1f) <= 0.05f
    }

    private fun softmaxInPlace(logits: FloatArray) {
        val max = logits.maxOrNull() ?: 0f
        var sum = 0.0
        for (i in logits.indices) {
            logits[i] = exp((logits[i] - max).toDouble()).toFloat()
            sum += logits[i]
        }
        if (sum == 0.0) return

        for (i in logits.indices) {
            logits[i] = (logits[i] / sum).toFloat()
        }
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(assetPath)
        FileInputStream(descriptor.fileDescriptor).use { inputStream ->
            val channel = inputStream.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
        }
    }

    override fun close() {
        if (interpreterDelegate.isInitialized()) {
            interpreterDelegate.value.close()
        }
    }
}
