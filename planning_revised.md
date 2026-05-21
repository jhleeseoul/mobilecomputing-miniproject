# Real-Time Speech Command Recognition on Android with Optimized On-Device DNN

## 1. Goal & Usage Scenario

The goal of this project is to build a real-time Android application that recognizes short spoken commands using an optimized on-device deep neural network. The app continuously senses audio from the smartphone microphone, processes the incoming audio stream with a compact keyword spotting model, and visualizes the prediction result in real time. The target commands will be a small fixed vocabulary such as **yes, no, up, down, left, right, stop, and go**, with two additional classes: **unknown** and **silence**. This design makes the task realistic for mobile deployment while still covering the core requirements of real-time sensing, DNN-based processing, model optimization, and result visualization.

The intended usage scenario is a lightweight voice-control interface for situations where touch interaction is inconvenient. For example, a user may want to control a simple mobile interface while cooking, exercising, wearing gloves, or keeping the phone at a short distance. Instead of using general-purpose speech recognition, which is often heavy and requires large models or cloud services, the app focuses on a limited set of commands that can be recognized locally on the device. When the user says “up,” “down,” “stop,” or “go,” the app displays the recognized command immediately and shows the model confidence for each class.

For the final demonstration, the app will include a simple voice-controlled command panel. The commands **up, down, left, and right** will move an on-screen indicator, while **go** and **stop** will change the system state between active and paused. The commands **yes** and **no** may be used as simple confirmation commands. This demo scenario makes the recognition result more concrete than only showing a predicted label, while still keeping the app scope small enough to complete within the project schedule.

The application will provide a live interface consisting of four main components. First, it will show a real-time waveform view of the microphone input so that users can see whether the app is actively sensing audio. Second, it will show confidence bars for all target classes, allowing users to understand not only the final prediction but also the uncertainty of the model. Third, it will display the most recently detected command with a timestamp. Fourth, it will show an inference latency badge so that the user can observe whether the optimized model is fast enough for real-time interaction. These visual components are directly aligned with the project requirement that real-time data collection and result visualization should be included in the app.

This project intentionally avoids overlap with the previous assignments. The previous assignments already covered accelerometer-based human activity recognition and camera-based object classification. In contrast, this project uses the smartphone microphone as the primary sensing modality and solves an audio keyword spotting problem. The model input, preprocessing pipeline, DNN architecture, optimization target, and user interaction scenario are therefore different from both activity recognition and camera object classification.

## 2. Related Works

Keyword spotting is a common task in mobile and embedded machine learning. Unlike full automatic speech recognition, keyword spotting focuses on detecting a small number of predefined words or short phrases. This makes it suitable for resource-constrained devices because the model does not need to understand arbitrary sentences. Many voice-trigger systems and simple voice-control interfaces use this idea: instead of transcribing all speech, they only need to detect whether a specific command was spoken.

For the dataset, this project will use the **Google Speech Commands** dataset. It contains many short one-second recordings of spoken English words and is widely used for limited-vocabulary speech recognition and keyword spotting. The dataset is appropriate for this project because it provides a standard benchmark, contains the target command words, and has enough samples to train a compact model from scratch or with a simple architecture. We will use a subset of the dataset consisting of eight command classes, plus an unknown class and a silence class. The unknown class will be built from words outside the selected command set, and the silence class will be generated from background noise or near-silent audio segments.

For model design, we will use a compact convolutional neural network inspired by **DS-CNN**, or depthwise separable convolutional neural networks. DS-CNN-style models are suitable for mobile audio inference because they reduce the number of parameters and computation compared to standard convolutional networks. The input to the model will be a one-second audio window converted into a time-frequency representation such as log-Mel spectrogram or MFCC. This representation is commonly used in speech and audio classification because it captures frequency patterns that are useful for distinguishing spoken words.

For mobile deployment, this project will use **TensorFlow Lite**. The trained model will first be exported as a float32 TFLite model, and then optimized using post-training integer quantization. Quantization is especially appropriate for this project because keyword spotting models are small and convolution-heavy, making them good candidates for reduced model size and faster inference. The final evaluation will compare the original float32 model and the optimized int8 model in terms of accuracy, model file size, and inference latency on the Android device.

## 3. Key Idea

The key idea of this project is to combine a standard public speech dataset with a custom lightweight on-device inference pipeline. Instead of downloading a pretrained model and only converting it to TFLite, we will train or fine-tune our own compact keyword spotting model using Speech Commands. This gives the project more independence and makes the model optimization experiment more meaningful.

The pipeline consists of five stages. First, one-second audio clips are loaded from the Speech Commands dataset and mapped to ten classes: eight command words, unknown, and silence. Second, the raw waveform is converted to log-Mel or MFCC features using fixed preprocessing parameters such as sample rate, window size, hop length, and number of Mel bins or cepstral coefficients. Third, a compact DS-CNN baseline model is trained on these features. Fourth, the trained model is converted to TFLite and optimized using int8 quantization. Fifth, the optimized model is integrated into an Android app that performs real-time microphone sensing and visualizes the prediction results.

The project will focus on making the comparison between the original and optimized models clear. The baseline float32 model will be used as the reference. The int8 quantized model will then be evaluated using the same test split, the same preprocessing pipeline, and the same Android inference code path as much as possible. The main expected result is that the optimized model will significantly reduce model size and inference latency while causing only marginal accuracy degradation.

The target quantitative goals are as follows. The float32 baseline should reach at least **90% test accuracy** on the selected Speech Commands split, and the int8 model should keep the accuracy drop within **3 percentage points** if possible. The optimized model should reduce the TFLite file size by at least **70%** compared with the float32 TFLite model. On the Android device, the target inference latency is **under 100 ms per inference**, with the app updating predictions every **250–500 ms**. These numbers will be treated as target goals rather than strict guarantees, but they provide clear evaluation criteria for the final presentation.

A secondary idea is to reduce the mismatch between the public dataset and the actual mobile environment. Speech Commands contains recordings from many speakers, but the app will be tested on team members’ voices and on a real smartphone microphone. Therefore, as an optional extension, we will collect approximately 100–200 short recordings from team members for the selected commands. These recordings may be used as an additional validation set or for a small accent-adaptation fine-tuning experiment. This extension is not required for the MVP, but it would strengthen the project by showing a complete path from public dataset training to real-user calibration.

## 4. Implementation Plan

we will use vscode + WSL2 + python venv.

### 4.1 Data Preparation

We will use the Google Speech Commands dataset as the main training and evaluation dataset. The selected command classes will be eight words such as **yes, no, up, down, left, right, stop, and go**. The unknown class will be constructed from other words in the dataset, and the silence class will be constructed using background noise samples and low-energy audio segments. All audio will be resampled or loaded at a fixed sample rate, preferably 16 kHz, and clipped or padded to one second.

#### downloading datasets
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("neehakurelli/google-speech-commands")

print("Path to dataset files:", path)
```

To improve robustness, we will apply simple data augmentation during training. The planned augmentations include random time shift, background noise mixing, and random volume scaling. These augmentations are useful because real mobile input may contain background noise, different speaking volumes, and slight timing differences. We will keep the augmentation pipeline simple so that the model remains easy to train and debug.

### 4.2 Feature Extraction

The model input will be either log-Mel spectrogram or MFCC features computed from a one-second waveform. At training time, feature extraction will be implemented in Python. For the Android app, we will implement the same feature extraction parameters in Kotlin or use a compatible TensorFlow Lite audio preprocessing utility if available. To prevent a mismatch between training-time and app-time features, we will fix all preprocessing parameters early, including sample rate, frame length, frame stride, number of Mel bins or MFCC coefficients, and normalization method.

A major implementation risk is that Python feature extraction and Android feature extraction may produce slightly different values. To mitigate this, we will create several test audio clips and compare the intermediate feature outputs between the Python pipeline and the Android pipeline. If exact matching becomes too time-consuming, we will simplify the preprocessing pipeline and use a representation that is easier to reproduce consistently.

### 4.3 Model Training

The baseline model will be a small DS-CNN-style network. It will use depthwise separable convolution blocks to reduce computation and parameter count. The model will take the time-frequency feature matrix as input and output probabilities over ten classes. We will train the model using cross-entropy loss and evaluate it with accuracy and confusion matrix on the held-out test set.

The initial goal is not to achieve state-of-the-art keyword spotting performance. Instead, the goal is to obtain a stable and reasonably accurate compact model that can be deployed on a smartphone and optimized without significant accuracy loss. The project will follow an MVP-first strategy: a simple small 2D CNN will be kept as a guaranteed deployable baseline, while the DS-CNN-style model will be used as the main compact model if training and conversion remain stable. This makes the project schedule safer because the app can still be completed even if the DS-CNN implementation requires more tuning than expected.

The fallback small CNN will use two or three convolutional layers, batch normalization, pooling, and a final dense classification layer. It will still satisfy the project requirements because it is a DNN model trained on the target dataset, converted to TFLite, optimized with int8 quantization, and deployed for real-time on-device inference. Therefore, the priority order is: first, complete a working real-time app with a quantized small CNN; second, replace or compare it with the DS-CNN model if time allows.

### 4.4 Model Optimization

The main optimization method will be post-training int8 quantization. After training the float32 model, we will convert it to a float32 TFLite model and then to an int8 quantized TFLite model using a representative dataset. The representative dataset will consist of a small subset of training audio features so that the quantizer can estimate the activation ranges.

The optimized model will be compared with the original model using the following metrics:

* Test accuracy on the selected Speech Commands test set
* Accuracy drop after quantization
* TFLite model file size
* Average inference latency on the Android device
* Target achievement summary: at least 70% size reduction, accuracy drop within 3 percentage points if possible, and under 100 ms inference latency
* Optional memory usage or initialization time, if measurement is feasible

If time allows, we may also test pruning-aware fine-tuning. However, pruning will be treated as an optional extension rather than the core deliverable. The required optimization comparison will focus on float32 versus int8 quantized TFLite models, because this path is more reliable within the one-month project schedule.

### 4.5 Android App Implementation

The Android app will be implemented using Kotlin in Android Studio. The app will use the device microphone through Android’s audio recording APIs to collect real-time PCM audio. The app will maintain a rolling one-second audio buffer. At a fixed interval, for example every 250–500 ms, the app will extract features from the current buffer and run TFLite inference.

The app UI will include a live waveform display, confidence bars for each class, the most recently recognized command, an inference latency badge, and a small command panel used for the demo. In the command panel, recognized directional commands will move an indicator, and start/stop commands will visibly change the app state. To reduce unstable predictions, we will apply simple temporal smoothing, such as averaging predictions across recent windows or requiring the same command to appear above a confidence threshold before updating the detected command. This will make the app demonstration more stable and realistic.

The final repository will include the model training code, model conversion and quantization code, Android Studio project, TFLite model files, and documentation for reproducing the experiment. The final presentation will show the app workflow, the model architecture, optimization results, and a demo video or captured screenshots.

## 5. Project Timeline

### Week 1: Proposal, Dataset Setup, and Baseline Pipeline

In the first week, we will finalize the project scope, select the exact command classes, and set up the Speech Commands data loader. We will implement the preprocessing pipeline for one-second waveforms and log-Mel or MFCC feature extraction. By the end of the week, we aim to train a very simple baseline model to verify that the dataset split, labels, and evaluation code are working correctly.

Main deliverables:

* Finalized project proposal
* Dataset loading and class mapping code
* Feature extraction pipeline
* First baseline training run

### Week 2: DS-CNN Training and Evaluation

In the second week, we will train a deployable small CNN baseline first and then implement the DS-CNN-style compact model as the main model. We will tune basic hyperparameters such as learning rate, batch size, number of convolution blocks, and data augmentation strength. We will evaluate the model using test accuracy and confusion matrix. If DS-CNN training or conversion is unstable, we will keep the small CNN as the final deployable model to protect the project schedule.

Main deliverables:

* Trained float32 baseline model
* Accuracy and confusion matrix on the selected test set
* Saved model ready for TFLite conversion

### Week 3: TFLite Conversion and Quantization

In the third week, we will convert the trained model to TFLite and apply int8 post-training quantization. We will compare the float32 TFLite model and the int8 TFLite model using accuracy, model size, and inference time in a local benchmark. We will also start integrating the model into the Android app and verify that sample inputs produce reasonable predictions.

Main deliverables:

* Float32 TFLite model
* Int8 quantized TFLite model
* Quantization comparison table
* Initial Android inference test

### Week 4: Android Real-Time App Integration

In the fourth week, we will complete the Android app. This includes microphone sensing, rolling buffer management, feature extraction, TFLite inference, and real-time visualization. We will implement the waveform view, confidence bars, recent command display, and latency badge. We will also measure on-device latency using the actual app.

Main deliverables:

* Working Android app with real-time microphone input
* Real-time prediction visualization
* On-device latency measurements
* Demo screenshots or video

### Final Week: Evaluation, Optional Adaptation, and Presentation

In the final stage, we will refine the app, collect final evaluation results, and prepare the presentation. If time allows, we will collect 100–200 short command recordings from team members and use them as a small real-user validation set or for optional fine-tuning. The final presentation will emphasize the complete pipeline: public dataset training, compact DNN design, TFLite deployment, quantization, and real-time Android demonstration.

Main deliverables:

* Final APK
* Final GitHub repository with training, optimization, and Android code
* Final comparison of float32 versus int8 model
* Demo video or captured images
* Final presentation slides

## Expected Outcome

By the end of the project, we expect to deliver a real-time Android keyword spotting app that recognizes a small set of spoken commands using an optimized on-device DNN model. The expected result is that int8 quantization will reduce model size by around 70% or more and improve or maintain inference latency, while keeping the accuracy drop within approximately 3 percentage points if possible. The final demo will show both the raw recognition outputs and a simple voice-controlled command panel, making the system behavior easy to understand. The project is designed to be realistic within a one-month schedule, technically distinct from the previous assignments, and well aligned with the course requirements for mobile sensing, optimized DNN processing, visualization, and final app deployment.
