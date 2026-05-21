package com.example.kwsapp.inference

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class KeywordSpotter(
    private val context: Context,
    private val modelAssetPath: String = "model_int8.tflite",
    private val labels: List<String> = listOf("yes", "no", "up", "down", "left", "right", "stop", "go", "unknown", "silence"),
) {
    private val mfccExtractor = MfccExtractor()

    private val interpreter: Interpreter by lazy {
        Interpreter(loadModelFile(context, modelAssetPath), Interpreter.Options().apply { setNumThreads(2) })
    }

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

    fun predict(featureOrAudio: FloatArray): PredictionResult {
        val inputVector = toModelInput(featureOrAudio)
        val startNs = System.nanoTime()

        val output = if (io.outputIsInt8) {
            val out = Array(1) { ByteArray(io.outputShape.last()) }
            interpreter.run(buildInputBuffer(inputVector), out)
            dequantizeOutput(out[0])
        } else {
            val out = Array(1) { FloatArray(io.outputShape.last()) }
            interpreter.run(buildInputBuffer(inputVector), out)
            out[0]
        }

        val latencyMs = (System.nanoTime() - startNs) / 1_000_000f
        val probs = softmax(output)
        val topIndex = probs.indices.maxByOrNull { probs[it] } ?: 0

        return PredictionResult(
            topLabel = labels.getOrElse(topIndex) { "class_$topIndex" },
            topScore = probs[topIndex],
            scores = probs,
            latencyMs = latencyMs,
        )
    }

    private fun toModelInput(featureOrAudio: FloatArray): FloatArray {
        val featureElementCount = io.inputShape.drop(1).reduce { acc, value -> acc * value }

        return when {
            featureOrAudio.size == featureElementCount -> featureOrAudio
            featureOrAudio.size >= 8000 -> {
                val frames = mfccExtractor.extractFromAudio(featureOrAudio, audioSampleRate = 16000)
                flatten(frames)
            }
            else -> throw IllegalArgumentException(
                "Input length ${featureOrAudio.size} does not match model feature size $featureElementCount"
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

    private fun buildInputBuffer(flattenedFeatures: FloatArray): ByteBuffer {
        val buffer = if (io.inputIsInt8) {
            ByteBuffer.allocateDirect(flattenedFeatures.size)
        } else {
            ByteBuffer.allocateDirect(flattenedFeatures.size * 4)
        }
        buffer.order(ByteOrder.nativeOrder())

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

    private fun dequantizeOutput(raw: ByteArray): FloatArray {
        require(io.outputScale != 0f) { "Invalid int8 output quantization scale" }
        return FloatArray(raw.size) { idx -> (raw[idx].toInt() - io.outputZeroPoint) * io.outputScale }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size)
        var sum = 0.0
        for (i in logits.indices) {
            exps[i] = exp((logits[i] - max).toDouble()).toFloat()
            sum += exps[i]
        }
        if (sum == 0.0) return exps

        return FloatArray(logits.size) { i -> (exps[i] / sum).toFloat() }
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(assetPath)
        FileInputStream(descriptor.fileDescriptor).use { inputStream ->
            val channel = inputStream.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
        }
    }
}
