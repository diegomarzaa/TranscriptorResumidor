
# "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"
WHISPER_VERSION="victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
MAX_SIZE_CHUNK = 5  *1024*1024

import replicate
import json
from pydub import AudioSegment
import os
import math
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_api_token():
    token = os.getenv('REPLICATE_API_TOKEN')
    if not token:
        raise ValueError("REPLICATE_API_TOKEN not found. Please set it in your .env file or as an environment variable.")
    return token

def get_audio_duration(filename):
    audio = AudioSegment.from_file(filename)
    return len(audio)

def split_audio(filename, max_size_bytes=MAX_SIZE_CHUNK, output_format="mp3"):
    audio = AudioSegment.from_file(filename)
    duration_ms = len(audio)
    
    # Calculate the approximate duration for a MAX_SIZE_CHUNK MB chunk
    bytes_per_ms = os.path.getsize(filename) / duration_ms
    chunk_duration_ms = int(max_size_bytes / bytes_per_ms)
    
    chunks = []
    for start in range(0, duration_ms, chunk_duration_ms):
        end = min(start + chunk_duration_ms, duration_ms)
        chunk = audio[start:end]
        chunk_name = f"chunk_{start//1000}_{end//1000}.{output_format}"
        chunk.export(chunk_name, format=output_format)
        chunks.append(chunk_name)
    
    return chunks


def transcribe_audio(filename, language="None", prompt=""):
    # language: catalan, english, spanish
    
    if WHISPER_VERSION == "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb":
        WHISPER_INPUT = {
            "debug": False,
            # "language": language, # Automatic
            "vad_onset": 0.5,
            "audio_file": open(filename, "rb"),
            "batch_size": 64,
            "vad_offset": 0.363,
            "diarization": False,
            "temperature": 0,
            "align_output": False,
            "language_detection_min_prob": 0,
            "language_detection_max_tries": 5,
            "initial_prompt": prompt  # Add the prompt here
        }
    elif WHISPER_VERSION == "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c":
        WHISPER_INPUT = {
            "debug": False,
            "language": language,
            "vad_onset": 0.5,
            "audio": open(filename, "rb"),
            "batch_size": 64,
            "vad_offset": 0.363,
            "diarization": False,
            "temperature": 0,
            "align_output": False,
            "language_detection_min_prob": 0,
            "language_detection_max_tries": 5,
        }
    
    try:
        api_token = get_api_token()
        client = replicate.Client(api_token=api_token)
        output = client.run(
            WHISPER_VERSION,
            input=WHISPER_INPUT
        )
        # Ensure output has a full_text field
        if isinstance(output, dict) and "segments" in output:
            output["full_text"] = " ".join(seg["text"] for seg in output["segments"])
        return output
    except Exception as e:
        logger.error(f"Error in transcription of {filename}: {e}")
        return None

def save_transcription(transcription, output_filename):
    try:
        # Create exports subfolder if it doesn't exist
        transcripciones_carpeta = os.path.join(os.path.dirname(output_filename), "transcripciones")
        os.makedirs(transcripciones_carpeta, exist_ok=True)

        output_path = os.path.join(transcripciones_carpeta, output_filename)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(transcription, file, ensure_ascii=False, indent=4)
        logger.info(f"Transcription saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")


def process_chunk(chunk, language, prompt, start_time):
    logger.info(f"Transcribing {chunk}...")
    transcription = transcribe_audio(chunk, language, prompt)
    os.remove(chunk)  # Remove temporary chunk file
    if transcription and "full_text" not in transcription:
        transcription["full_text"] = " ".join(seg["text"] for seg in transcription["segments"])
    return transcription, start_time


def combine_transcriptions(transcriptions):
    # Sort transcriptions based on start time
    sorted_transcriptions = sorted(transcriptions, key=lambda x: x[1])

    combined = {
        "detected_language": sorted_transcriptions[0][0]["detected_language"],
        "full_text": "",
        "segments": []
    }
    
    for trans, start_time in sorted_transcriptions:
        if "full_text" in trans:
            combined["full_text"] += trans["full_text"] + " "
        for segment in trans["segments"]:
            segment["start"] += start_time
            segment["end"] += start_time
            combined["segments"].append(segment)
    
    combined["segments"].sort(key=lambda x: x["start"])
    if not combined["full_text"]:
        combined["full_text"] = " ".join(seg["text"] for seg in combined["segments"])
    combined["full_text"] = combined["full_text"].strip()
    return combined



def main(filename, output_filename, language="None", custom_prompt=""):
    file_size_bytes = os.path.getsize(filename)
    max_chunk_size_bytes = MAX_SIZE_CHUNK

    if file_size_bytes > max_chunk_size_bytes:
        logger.info(f"File size ({file_size_bytes/1024/1024:.2f} MB) exceeds {max_chunk_size_bytes/1024/1024:.2f} MB limit for Whisper API. Splitting into chunks...")
        chunks = split_audio(filename, max_chunk_size_bytes)
        chunk_transcriptions = []
        
        with ThreadPoolExecutor() as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk, language, custom_prompt, int(chunk.split('_')[1])): chunk for chunk in chunks}
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Processing chunks"):
                chunk = future_to_chunk[future]
                try:
                    transcription, start_time = future.result()
                    if transcription:
                        chunk_transcriptions.append((transcription, start_time))
                        # Update prompt with the last part of the transcription for context
                        custom_prompt = " ".join([seg["text"] for seg in transcription["segments"][-5:]])
                except Exception as e:
                    logger.error(f"Error processing {chunk}: {e}")
        
        if chunk_transcriptions:
            logger.info("Combining transcriptions from all chunks...")
            full_transcription = combine_transcriptions(chunk_transcriptions)
            save_transcription(full_transcription, output_filename)
        else:
            logger.error("Transcription failed.")

    else:
        logger.info(f"Transcribing {filename}...")
        transcription = transcribe_audio(filename, language, custom_prompt)
        if transcription:
            logger.info("Transcription completed.")
            if "full_text" not in transcription:
                transcription["full_text"] = " ".join(seg["text"] for seg in transcription["segments"])
            save_transcription(transcription, output_filename)
        else:
            logger.error("Transcription failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper API")
    parser.add_argument("filename", help="Path to the audio file")
    parser.add_argument("-o", "--output", default="transcription.json", help="Output filename")
    parser.add_argument("-l", "--language", default="None", help="Language of the audio (default: auto)")
    parser.add_argument("-p", "--prompt", default="", help="Custom prompt to guide transcription")
    args = parser.parse_args()

    main(args.filename, args.output, args.language, args.prompt)