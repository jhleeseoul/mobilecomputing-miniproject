package com.example.kwsapp.inference

import java.util.Arrays
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.pow

class MfccExtractor(
    private val sampleRate: Int = 16000,
    private val frameLength: Int = 640,
    private val frameStep: Int = 320,
    private val fftLength: Int = 1024,
    private val numMelBins: Int = 40,
    private val numMfcc: Int = 13,
    private val normalize: Boolean = true,
) {
    private val analysisWindow: FloatArray by lazy { hann(frameLength) }
    private val melFilters: Array<FloatArray> by lazy { melFilterBank(sampleRate, fftLength, numMelBins) }
    private val dctBasis: Array<FloatArray> by lazy { buildDctBasis(numMelBins, numMfcc) }
    private val spectrumBins: Int = fftLength / 2 + 1

    fun extractFromAudio(audio: FloatArray, audioSampleRate: Int): Array<FloatArray> {
        val resampled = if (audioSampleRate == sampleRate) audio else resampleLinear(audio, audioSampleRate, sampleRate)
        val oneSecond = padOrTrim(resampled, sampleRate)
        return extractFromWaveform(oneSecond)
    }

    fun extractFromWaveform(waveform: FloatArray): Array<FloatArray> {
        val frames = frameSignal(waveform, frameLength, frameStep)
        val realScratch = DoubleArray(fftLength)
        val imagScratch = DoubleArray(fftLength)
        val powerScratch = FloatArray(spectrumBins)

        val features = Array(frames.size) { FloatArray(numMfcc) }
        for (i in frames.indices) {
            powerSpectrumInPlace(frames[i], realScratch, imagScratch, powerScratch)
            val mel = FloatArray(numMelBins)

            for (m in 0 until numMelBins) {
                var sum = 0.0
                for (k in 0 until spectrumBins) {
                    sum += powerScratch[k] * melFilters[m][k]
                }
                mel[m] = ln(sum.toFloat() + 1e-6f)
            }

            val mfcc = dctType2(mel, numMfcc)
            features[i] = mfcc
        }

        if (normalize) {
            normalizeInPlace(features)
        }

        return features
    }

    private fun frameSignal(signal: FloatArray, length: Int, step: Int): Array<FloatArray> {
        val numFrames = 1 + (signal.size - length).coerceAtLeast(0) / step
        val window = if (length == frameLength) analysisWindow else hann(length)
        val frames = Array(numFrames) { FloatArray(length) }

        for (i in 0 until numFrames) {
            val start = i * step
            for (j in 0 until length) {
                val idx = start + j
                val sample = if (idx < signal.size) signal[idx] else 0f
                frames[i][j] = sample * window[j]
            }
        }
        return frames
    }

    private fun hann(length: Int): FloatArray {
        val out = FloatArray(length)
        if (length <= 0) return out
        if (length == 1) {
            out[0] = 1f
            return out
        }
        for (i in 0 until length) {
            out[i] = (0.5 - 0.5 * cos(2.0 * PI * i / (length - 1))).toFloat()
        }
        return out
    }

    private fun powerSpectrumInPlace(
        frame: FloatArray,
        real: DoubleArray,
        imag: DoubleArray,
        power: FloatArray,
    ) {
        Arrays.fill(real, 0.0)
        Arrays.fill(imag, 0.0)
        for (i in frame.indices) {
            real[i] = frame[i].toDouble()
        }

        fftInPlace(real, imag)
        for (k in power.indices) {
            val r = real[k]
            val im = imag[k]
            power[k] = ((r * r + im * im) / fftLength).toFloat()
        }
    }

    // Iterative radix-2 Cooley-Tukey FFT.
    private fun fftInPlace(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]
                real[i] = real[j]
                real[j] = tr
                val ti = imag[i]
                imag[i] = imag[j]
                imag[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wlenR = cos(ang)
            val wlenI = kotlin.math.sin(ang)
            var i = 0
            while (i < n) {
                var wr = 1.0
                var wi = 0.0
                for (k in 0 until len / 2) {
                    val uR = real[i + k]
                    val uI = imag[i + k]
                    val vR = real[i + k + len / 2] * wr - imag[i + k + len / 2] * wi
                    val vI = real[i + k + len / 2] * wi + imag[i + k + len / 2] * wr

                    real[i + k] = uR + vR
                    imag[i + k] = uI + vI
                    real[i + k + len / 2] = uR - vR
                    imag[i + k + len / 2] = uI - vI

                    val nextWr = wr * wlenR - wi * wlenI
                    wi = wr * wlenI + wi * wlenR
                    wr = nextWr
                }
                i += len
            }
            len = len shl 1
        }
    }

    private fun melFilterBank(sr: Int, fftSize: Int, numMel: Int): Array<FloatArray> {
        val numBins = fftSize / 2 + 1
        val melMin = hzToMel(20.0)
        val melMax = hzToMel(sr / 2.0)

        val melPoints = DoubleArray(numMel + 2) { idx ->
            melMin + (melMax - melMin) * idx / (numMel + 1)
        }
        val hzPoints = melPoints.map { melToHz(it) }
        val binPoints = hzPoints.map { ((fftSize + 1) * it / sr).toInt().coerceIn(0, numBins - 1) }

        val filters = Array(numMel) { FloatArray(numBins) }
        for (m in 1..numMel) {
            val left = binPoints[m - 1]
            val center = binPoints[m]
            val right = binPoints[m + 1]

            for (k in left until center) {
                val denom = (center - left).coerceAtLeast(1)
                filters[m - 1][k] = ((k - left).toFloat() / denom).coerceAtLeast(0f)
            }
            for (k in center until right) {
                val denom = (right - center).coerceAtLeast(1)
                filters[m - 1][k] = ((right - k).toFloat() / denom).coerceAtLeast(0f)
            }
        }
        return filters
    }

    private fun dctType2(values: FloatArray, outDim: Int): FloatArray {
        val n = values.size
        val out = FloatArray(outDim)
        val canUseBasis = outDim == dctBasis.size && n == dctBasis.firstOrNull()?.size
        for (k in 0 until outDim) {
            var sum = 0.0
            for (i in 0 until n) {
                val basis = if (canUseBasis) dctBasis[k][i] else cos(PI * k * (2 * i + 1) / (2.0 * n)).toFloat()
                sum += values[i] * basis
            }
            out[k] = sum.toFloat()
        }
        return out
    }

    private fun buildDctBasis(inputDim: Int, outputDim: Int): Array<FloatArray> {
        return Array(outputDim) { k ->
            FloatArray(inputDim) { i ->
                cos(PI * k * (2 * i + 1) / (2.0 * inputDim)).toFloat()
            }
        }
    }

    private fun normalizeInPlace(features: Array<FloatArray>) {
        var sum = 0.0
        var sqSum = 0.0
        var count = 0

        for (frame in features) {
            for (v in frame) {
                val d = v.toDouble()
                sum += d
                sqSum += d * d
                count++
            }
        }

        if (count == 0) return
        val mean = sum / count
        val variance = (sqSum / count) - mean.pow(2)
        val std = kotlin.math.sqrt(variance.coerceAtLeast(1e-12))

        for (i in features.indices) {
            for (j in features[i].indices) {
                features[i][j] = ((features[i][j] - mean) / (std + 1e-6)).toFloat()
            }
        }
    }

    private fun padOrTrim(audio: FloatArray, target: Int): FloatArray {
        if (audio.size == target) return audio
        if (audio.size > target) return audio.copyOfRange(0, target)
        val out = FloatArray(target)
        System.arraycopy(audio, 0, out, 0, audio.size)
        return out
    }

    private fun resampleLinear(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (input.isEmpty()) return FloatArray(0)
        val ratio = dstRate.toDouble() / srcRate.toDouble()
        val outLength = (input.size * ratio).toInt().coerceAtLeast(1)
        val out = FloatArray(outLength)

        for (i in 0 until outLength) {
            val srcPos = i / ratio
            val x0 = srcPos.toInt().coerceIn(0, input.size - 1)
            val x1 = (x0 + 1).coerceAtMost(input.size - 1)
            val t = (srcPos - x0).toFloat()
            out[i] = input[x0] * (1 - t) + input[x1] * t
        }
        return out
    }

    private fun hzToMel(hz: Double): Double = 2595.0 * kotlin.math.log10(1.0 + hz / 700.0)
    private fun melToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
}
