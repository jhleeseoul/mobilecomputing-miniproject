package com.example.kwsapp.inference

data class PredictionResult(
    val topLabel: String,
    val topScore: Float,
    val scores: FloatArray,
    val latencyMs: Float,
)
