package com.example.kwsapp.inference

import android.content.Context
import java.io.BufferedInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object WavReader {
    data class WavData(val sampleRate: Int, val pcm: FloatArray)

    fun readAssetPcm16(context: Context, assetPath: String): WavData {
        context.assets.open(assetPath).use { input ->
            val bytes = BufferedInputStream(input).readBytes()
            require(bytes.size > 44) { "Invalid wav file: too small" }

            val header = ByteBuffer.wrap(bytes, 0, 44).order(ByteOrder.LITTLE_ENDIAN)
            val riff = String(bytes, 0, 4)
            val wave = String(bytes, 8, 4)
            require(riff == "RIFF" && wave == "WAVE") { "Invalid WAV header" }

            val channels = header.getShort(22).toInt()
            val sampleRate = header.getInt(24)
            val bitsPerSample = header.getShort(34).toInt()
            require(bitsPerSample == 16) { "Only 16-bit PCM WAV is supported" }

            val dataStart = 44
            val sampleCountTotal = (bytes.size - dataStart) / 2
            val frameCount = sampleCountTotal / channels

            val pcm = FloatArray(frameCount)
            var readIndex = dataStart
            var frame = 0
            while (frame < frameCount) {
                var mixed = 0f
                for (ch in 0 until channels) {
                    val low = bytes[readIndex].toInt() and 0xff
                    val high = bytes[readIndex + 1].toInt()
                    val sample = (high shl 8) or low
                    mixed += sample.toShort() / 32768f
                    readIndex += 2
                }
                pcm[frame] = mixed / channels
                frame++
            }

            return WavData(sampleRate = sampleRate, pcm = pcm)
        }
    }
}
