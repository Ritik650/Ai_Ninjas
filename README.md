---
library_name: transformers
license: apache-2.0
base_model: openai/whisper-large-v3
tags:
- generated_from_trainer
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: speech-emotion-recognition-with-openai-whisper-large-v3
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->


# 🎧 **Speech Emotion Recognition with Whisper**
This project leverages the **Whisper** model to recognize emotions in speech. The goal is to classify audio recordings into different emotional categories, such as **Happy**, **Sad**, **Surprised**, and etc.


## 🗂 **Dataset**
The dataset used for training and evaluation is sourced from multiple datasets, including:
- [RAVDESS](https://zenodo.org/records/1188976#.XsAXemgzaUk)
- [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee/data)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)

The dataset contains recordings labeled with various emotions. Below is the distribution of the emotions in the dataset:
| **Emotion** | **Count** |
|-------------|-----------|
| sad         | 752       |
| happy       | 752       |
| angry       | 752       |
| neutral     | 716       |
| disgust     | 652       |
| fearful     | 652       |
| surprised   | 652       |
| calm        | 192       |

This distribution reflects the balance of emotions in the dataset, with some emotions having more samples than others. Excluded the "calm" emotion during training due to its underrepresentation.


## 🎤 **Preprocessing**
- **Audio Loading**: Using **Librosa** to load the audio files and convert them to numpy arrays.
- **Feature Extraction**: The audio data is processed using the **Whisper Feature Extractor**, which standardizes and normalizes the audio features for input to the model.


## 🔧 **Model**
The model used is the **Whisper Large V3** model, fine-tuned for **audio classification** tasks:
- **Model**: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) 
- **Output**: Emotion labels (`Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'`)
  
I map the emotion labels to numeric IDs and use them for model training and evaluation.


## ⚙️ **Training**
The model is trained with the following parameters:
- **Learning Rate**: `5e-05`  
- **Train Batch Size**: `2`
- **Eval Batch Size**: `2`
- **Random Seed**: `42`  
- **Gradient Accumulation Steps**: `5`  
- **Total Train Batch Size**: `10` (effective batch size after gradient accumulation)
- **Optimizer**: **Adam** with parameters: `betas=(0.9, 0.999)` and `epsilon=1e-08`
- **Learning Rate Scheduler**: `linear`
- **Warmup Ratio for LR Scheduler**: `0.1`
- **Number of Epochs**: `25`
- **Mixed Precision Training**: Native AMP (Automatic Mixed Precision)
  
These parameters ensure efficient model training and stability, especially when dealing with large datasets and deep models like **Whisper**.
The training utilizes **Wandb** for experiment tracking and monitoring.


## 📊 **Metrics**
The following evaluation metrics were obtained after training the model:
- **Loss**: `0.5008`
- **Accuracy**: `0.9199`
- **Precision**: `0.9230`
- **Recall**: `0.9199`
- **F1 Score**: `0.9198`
  
These metrics demonstrate the model's performance on the speech emotion recognition task. The high values for accuracy, precision, recall, and F1 score indicate that the model is effectively identifying emotional states from speech data.


## 🧪 **Results**
After training, the model is evaluated on the test dataset, and the results are monitored using **Wandb** in this [Link](https://wandb.ai/firdhoworking-sepuluh-nopember-institute-of-technology/speech-emotion-recognition-with-whisper?nw=nwuserfirdhoworking).
| Training Loss | Epoch   | Step | Validation Loss | Accuracy | Precision | Recall | F1     |
|:-------------:|:-------:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| 0.4948        | 0.9995  | 394  | 0.4911          | 0.8286   | 0.8449    | 0.8286 | 0.8302 |
| 0.6271        | 1.9990  | 788  | 0.5307          | 0.8225   | 0.8559    | 0.8225 | 0.8277 |
| 0.2364        | 2.9985  | 1182 | 0.5076          | 0.8692   | 0.8727    | 0.8692 | 0.8684 |
| 0.0156        | 3.9980  | 1576 | 0.5669          | 0.8732   | 0.8868    | 0.8732 | 0.8745 |
| 0.2305        | 5.0     | 1971 | 0.4578          | 0.9108   | 0.9142    | 0.9108 | 0.9114 |
| 0.0112        | 5.9995  | 2365 | 0.4701          | 0.9108   | 0.9159    | 0.9108 | 0.9114 |
| 0.0013        | 6.9990  | 2759 | 0.5232          | 0.9138   | 0.9204    | 0.9138 | 0.9137 |
| 0.1894        | 7.9985  | 3153 | 0.5008          | 0.9199   | 0.9230    | 0.9199 | 0.9198 |
| 0.0877        | 8.9980  | 3547 | 0.5517          | 0.9138   | 0.9152    | 0.9138 | 0.9138 |
| 0.1471        | 10.0    | 3942 | 0.5856          | 0.8895   | 0.9002    | 0.8895 | 0.8915 |
| 0.0026        | 10.9995 | 4336 | 0.8334          | 0.8773   | 0.8949    | 0.8773 | 0.8770 |


## 🚀 **How to Use**
```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label
```
```python
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs
```
```python
def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    
    return predicted_label
```
```python

predicted_emotion = predict_emotion(audio_path, model, feature_extractor, id2label)
print(f"Predicted Emotion: {predicted_emotion}")
```

## 🎯 Framework versions
- Transformers 4.44.2
- Pytorch 2.4.1+cu121
- Datasets 3.0.0
- Tokenizers 0.19.1
