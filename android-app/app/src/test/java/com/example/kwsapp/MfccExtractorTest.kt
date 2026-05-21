package com.example.kwsapp

import com.example.kwsapp.inference.MfccExtractor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MfccExtractorTest {
    @Test
    fun extractorReturnsExpectedShape() {
        val extractor = MfccExtractor()
        val audio = FloatArray(16000) { 0f }

        val mfcc = extractor.extractFromAudio(audio, 16000)

        assertEquals(49, mfcc.size)
        assertEquals(13, mfcc[0].size)
        assertTrue(mfcc.all { frame -> frame.all { it.isFinite() } })
    }
}
