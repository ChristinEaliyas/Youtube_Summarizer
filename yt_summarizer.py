import yt_dlp
from pydub import AudioSegment
import os
import shutil
import whisper
import requests
import json

output_directory="downloads"


def download_youtube_audio(link, output_directory="downloads"):
    try:
        # Ensure output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Specify the fixed filename "audio.mp3"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_directory, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(link, download=True)

        file_path = os.path.join(output_directory, "audio.mp3")
        print(f"Audio downloaded successfully to '{file_path}'.")
        return file_path
    except Exception as e:
        print(f"Error: {e}")
        return None

def convert_to_wav(input_file):
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_directory, f"{file_name}.wav")
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        
        print(f"Converted to WAV successfully: '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Error in WAV conversion: {e}")
        return None


def transcribe(audio_path):
    try:
        print("Transcibing Audio...")
        model = whisper.load_model("small.en")
        result = model.transcribe(audio_path)
        os.remove(audio_path)
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def make_api_call(payload):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    try:
        full_text = ""
        with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as response:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    response_data = json.loads(line)
                    full_text += response_data["response"]
        return full_text                    
    except requests.RequestException as e:
        print("Error:", e)
        return None

def generate_summary(text):
    prompt = ("""
    You are a Summarizing AI. You should summarize the given content.The summary should be minimum 600 words and it can go upto 1000 words.
    Find the context/Main Factor of the text and the  summarize based on it.The reponse should not contain any speacial characters it must only include numbers and text.
    Content:""" + str(text))
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
    }
    response_data = make_api_call(payload)
    return response_data

def handle_summary_stream(response_text):
    print("Summary update:", response_text)
