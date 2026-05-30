package com.example.kwsapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.kwsapp.inference.KeywordSpotter
import com.example.kwsapp.inference.PredictionResult
import com.example.kwsapp.inference.RealtimeKeywordSpotter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Week4RealtimeScreen()
        }
    }
}

@Composable
private fun Week4RealtimeScreen() {
    val context = LocalContext.current
    val labels = remember {
        listOf("yes", "no", "up", "down", "left", "right", "stop", "go", "unknown", "silence")
    }
    val scope = rememberCoroutineScope()
    val spotter = remember { KeywordSpotter(context, modelAssetPath = "model_int8.tflite", labels = labels) }
    val realtime = remember { RealtimeKeywordSpotter(spotter = spotter, labels = labels) }
    val streamState by realtime.state.collectAsState()
    val timeFormatter = remember { SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault()) }

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted ->
        hasPermission = granted
        if (granted) {
            realtime.start(scope)
        }
    }

    var cursorX by remember { mutableIntStateOf(0) }
    var cursorY by remember { mutableIntStateOf(0) }
    var panelActive by remember { mutableStateOf(false) }
    var panelMessage by remember { mutableStateOf("Say 'go' to activate command panel") }
    var lastAcceptedCommand by remember { mutableStateOf("none") }
    var lastAcceptedAtMs by remember { mutableStateOf(0L) }
    var selectedPage by rememberSaveable { mutableIntStateOf(0) }

    LaunchedEffect(streamState.prediction) {
        val prediction = streamState.prediction ?: return@LaunchedEffect
        if (prediction.topScore < 0.72f) return@LaunchedEffect
        if (prediction.topLabel == "unknown" || prediction.topLabel == "silence") return@LaunchedEffect

        val now = System.currentTimeMillis()
        if (now - lastAcceptedAtMs < 350L) return@LaunchedEffect

        when (prediction.topLabel) {
            "go" -> {
                panelActive = true
                panelMessage = "ACTIVE"
            }

            "stop" -> {
                panelActive = false
                panelMessage = "PAUSED"
            }

            "up" -> if (panelActive) cursorY = (cursorY - 1).coerceAtLeast(-5)
            "down" -> if (panelActive) cursorY = (cursorY + 1).coerceAtMost(5)
            "left" -> if (panelActive) cursorX = (cursorX - 1).coerceAtLeast(-5)
            "right" -> if (panelActive) cursorX = (cursorX + 1).coerceAtMost(5)
            "yes" -> panelMessage = "Confirmed"
            "no" -> panelMessage = "Cancelled"
        }

        lastAcceptedCommand = prediction.topLabel
        lastAcceptedAtMs = now
    }

    DisposableEffect(Unit) {
        onDispose { realtime.stop() }
    }

    MaterialTheme {
        Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text("Week4 Real-Time KWS App", style = MaterialTheme.typography.headlineSmall)
                Text(
                    if (streamState.isListening) "Mic: Listening" else "Mic: Stopped",
                    fontWeight = FontWeight.SemiBold,
                )
                streamState.errorMessage?.let { Text("Error: $it", color = Color(0xFFB00020)) }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = {
                        if (!hasPermission) {
                            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                        } else if (streamState.isListening) {
                            realtime.stop()
                        } else {
                            realtime.start(scope)
                        }
                    }) {
                        Text(
                            when {
                                !hasPermission -> "Grant Mic Permission"
                                streamState.isListening -> "Stop Listening"
                                else -> "Start Listening"
                            }
                        )
                    }

                    Button(onClick = {
                        cursorX = 0
                        cursorY = 0
                        panelMessage = if (panelActive) "ACTIVE" else "PAUSED"
                    }) {
                        Text("Reset Panel")
                    }
                }

                TabRow(selectedTabIndex = selectedPage) {
                    Tab(
                        selected = selectedPage == 0,
                        onClick = { selectedPage = 0 },
                        text = { Text("Waveform") },
                    )
                    Tab(
                        selected = selectedPage == 1,
                        onClick = { selectedPage = 1 },
                        text = { Text("Command") },
                    )
                }

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                ) {
                    if (selectedPage == 0) {
                        WaveformCard(samples = streamState.waveform)
                    } else {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .verticalScroll(rememberScrollState()),
                            verticalArrangement = Arrangement.spacedBy(10.dp),
                        ) {
                            PredictionCard(
                                prediction = streamState.prediction,
                                labels = labels,
                                predictionTimestampMs = streamState.predictionTimestampMs,
                                formatter = timeFormatter,
                            )

                            CommandPanelCard(
                                x = cursorX,
                                y = cursorY,
                                panelActive = panelActive,
                                panelMessage = panelMessage,
                                lastAcceptedCommand = lastAcceptedCommand,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun WaveformCard(samples: FloatArray) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Waveform (latest 1s buffer)", fontWeight = FontWeight.SemiBold)
            WaveformView(
                samples = samples,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp)
                    .background(Color(0xFFF4F7FA))
                    .border(1.dp, Color(0xFFCFD8DC)),
            )
            Text("Open the Command tab to see prediction and voice command panel.")
        }
    }
}

@Composable
private fun PredictionCard(
    prediction: PredictionResult?,
    labels: List<String>,
    predictionTimestampMs: Long,
    formatter: SimpleDateFormat,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Realtime Prediction", fontWeight = FontWeight.SemiBold)
            if (prediction == null) {
                Text("No inference yet")
                return@Column
            }

            val ts = if (predictionTimestampMs > 0) {
                formatter.format(Date(predictionTimestampMs))
            } else {
                "-"
            }
            Text("Top Label: ${prediction.topLabel}")
            Text("Top Score: ${"%.3f".format(prediction.topScore)}")
            Text("Latency: ${"%.2f".format(prediction.latencyMs)} ms")
            Text("Updated At: $ts")

            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                prediction.scores.forEachIndexed { idx, score ->
                    Text("${labels[idx]}: ${"%.3f".format(score)}")
                    LinearProgressIndicator(
                        progress = { score.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }
}

@Composable
private fun CommandPanelCard(
    x: Int,
    y: Int,
    panelActive: Boolean,
    panelMessage: String,
    lastAcceptedCommand: String,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Voice Command Panel", fontWeight = FontWeight.SemiBold)
            Text("State: ${if (panelActive) "ACTIVE" else "PAUSED"}")
            Text("Message: $panelMessage")
            Text("Last Command: $lastAcceptedCommand")
            Text("Cursor: ($x, $y)")

            Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                Canvas(
                    modifier = Modifier
                        .size(180.dp)
                        .background(Color(0xFFFDF7E3))
                        .border(1.dp, Color(0xFFBCAAA4)),
                ) {
                    val w = size.width
                    val h = size.height
                    drawLine(
                        color = Color(0xFF9E9E9E),
                        start = Offset(w / 2f, 0f),
                        end = Offset(w / 2f, h),
                        strokeWidth = 2f,
                    )
                    drawLine(
                        color = Color(0xFF9E9E9E),
                        start = Offset(0f, h / 2f),
                        end = Offset(w, h / 2f),
                        strokeWidth = 2f,
                    )

                    val px = ((x + 5f) / 10f).coerceIn(0f, 1f) * w
                    val py = ((y + 5f) / 10f).coerceIn(0f, 1f) * h
                    drawCircle(
                        color = if (panelActive) Color(0xFF2E7D32) else Color(0xFFB71C1C),
                        radius = 10f,
                        center = Offset(px, py),
                    )
                }
            }
        }
    }
}

@Composable
private fun WaveformView(samples: FloatArray, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        if (samples.isEmpty()) return@Canvas
        val midY = size.height / 2f
        drawLine(
            color = Color(0xFF90A4AE),
            start = Offset(0f, midY),
            end = Offset(size.width, midY),
            strokeWidth = 1f,
        )

        val last = (samples.size - 1).coerceAtLeast(1)
        var prevX = 0f
        var prevY = (midY - samples[0] * midY).coerceIn(0f, size.height)
        for (i in 1..last) {
            val x = (i.toFloat() / last.toFloat()) * size.width
            val y = (midY - samples[i] * midY).coerceIn(0f, size.height)
            drawLine(
                color = Color(0xFF1E88E5),
                start = Offset(prevX, prevY),
                end = Offset(x, y),
                strokeWidth = 2f,
            )
            prevX = x
            prevY = y
        }
    }
}
