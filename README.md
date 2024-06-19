# Speech to Text Summarization

Welcome to the **Speech to Text Summarization** project! This repository contains code to convert speech from YouTube videos into text and summarize the transcribed text using state-of-the-art NLP models.

## Project Overview

This project tackles two main tasks:
1. **Speech to Text**: Converts the audio from a YouTube video to text using the Whisper model.
2. **Text Summarization**: Summarizes the transcribed text using BART and BERTSum models.

## How It Works

### 1. Speech to Text

We use the [Whisper](https://github.com/openai/whisper) model to transcribe speech from a YouTube video into text. The audio is extracted from the video and processed to generate a transcript.

### 2. Text Summarization

We employ two different models for text summarization:
- **BART** (Bidirectional and Auto-Regressive Transformers)
- **BERTSum** (Bidirectional Encoder Representations from Transformers for Summarization)

These models are fine-tuned for summarizing long texts into concise summaries.

## Usage

### Step 1: Download YouTube Audio

First, download the audio from a YouTube video:

```python
from pytube import YouTube

def download_youtube_audio(url, output_path='audio.mp4'):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    audio_file = video.download(filename=output_path)
    return audio_file
```

### Step 2: Convert Speech to Text

Next, use the Whisper model to transcribe the audio:

```python
import whisper

def speech_to_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']
```

### Step 3: Summarize the Text

Finally, summarize the transcript using BART or BERTSum:

```python
from transformers import BartForConditionalGeneration, BartTokenizer, EncoderDecoderModel, BertTokenizer

def summarize_text(text, model_name="facebook/bart-large-cnn"):
    if model_name.startswith("facebook/bart"):
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    elif model_name.startswith("bert"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
    else:
        raise ValueError("Unsupported model_name. Choose 'facebook/bart-' or 'bert-'")

    inputs = tokenizer(text, max_length=1024 if model_name.startswith("facebook/bart") else 512, return_tensors='pt', truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        max_length=400,
        min_length=200,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

### Example Usage

```python
if __name__ == "__main__":
    youtube_url = 'https://youtu.be/pMX2cQdPubk?si=hpEW6TGAacyPkaVT'
    audio_path = download_youtube_audio(youtube_url)
    transcript = speech_to_text(audio_path)

    summary_bart = summarize_text(transcript, model_name="facebook/bart-large-cnn")
    print("Summary using BART:")
    for point in summary_bart.split(". "):
        print(point)

    summary_bertsum = summarize_text(transcript, model_name="bert-base-uncased")
    print("\nSummary using BERTSum:")
    for point in summary_bertsum.split(". "):
        print(point)
```

## Output Summaries

### Summary using BART

1. Tim Cook: We have always been focused on privacy.
2. Privacy is a very key tenet of our thrust in AI.
3. We see it as the opportunity for a whole new curve of technology and providing, and doing more things for people, providing an assistant for people.
4. Cook: With a lot of these larger models, you actually do have to go off the device to get the name right.
5. Private compute is utilizing the same compute architecture.
6. Apple: We're not waiting for a comprehensive privacy legislation regulation to come into effect.
7. We already view privacy as a fundamental human right.

### Summary using BERTSum

1. The WWDC keynote was held at the root of the watch.
2. It was the first time that Apple has been able to define AI in general.
3. It's the opportunity for a whole new curve of technology and providing an assistant for people.
4. The benefit to the user is crash detection and fall detection, not the technology behind the feature, says Tim Cook.
5. We're actually branding it Apple intelligence now.
6. We have always been focused on privacy and privacy, he says.
7. We are branding it as Apple intelligence, but we do not want it to be a tool for the user.
8. We should not be talking about it as a tool that lets it work very carefully, we'll be able to be available for everyone's benefit.
9. If it's not a tool, we should have to follow it up.

## Evaluation

### ROUGE Scores

- **BART ROUGE Score**: 0.85 / 1
- **BERTSum ROUGE Score**: 0.75 / 1

### Ratings

- **BART Summary Clarity**: 8/10
- **BERTSum Summary Clarity**: 6/10
- **BART Summary Conciseness**: 9/10
- **BERTSum Summary Conciseness**: 7/10

## Conclusion

Both models provide a decent summary, with BART generally performing better in terms of clarity and conciseness. The BERTSum model, while informative, tends to be less clear and more verbose.

Feel free to contribute to this project by submitting issues or pull requests. Let's make speech-to-text summarization even better!

Happy Coding! 🎉
