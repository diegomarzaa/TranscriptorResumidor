import os
import json
import math
import argparse
import logging
from tqdm import tqdm
from pydub import AudioSegment
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import replicate
from openai import OpenAI
import shutil

###########
INPUT_FOLDER = "audiosPendientes"
TRANSCRIPTIONS_FOLDER = "transcripciones"
SUMMARIES_FOLDER = "resumenes"
PROCESSED_FOLDER = "audiosTranscritosResumidos"
###########

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_SIZE_CHUNK = 5 * 1024 * 1024    # TODO: Asi lo hace poquito a poquito y no consume tanta memoria

# La otra versión posible es "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"
WHISPER_VERSION = "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
openaiClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_api_token(api_name):
    token = os.getenv(api_name)
    if not token:
        raise ValueError(f"{api_name} not found. Please set it in your .env file or as an environment variable.")
    return token

def get_audio_duration(filename):
    audio = AudioSegment.from_file(filename)
    return len(audio)

def split_audio(filename, max_size_bytes=MAX_SIZE_CHUNK, output_format="mp3"):
    audio = AudioSegment.from_file(filename)
    duration_ms = len(audio)
    
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
        api_token = get_api_token("REPLICATE_API_TOKEN")
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


def save_transcription(transcription, output_path):         #TODO: Just send the output_name, not the whole path
    try:
        # Ensure the transcriptions folder exists
        os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

        # Update the output path to use the transcriptions folder
        output_path = os.path.join(TRANSCRIPTIONS_FOLDER, os.path.basename(output_path))

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


###################### Summarization ######################

def load_transcription(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def chunk_text(text, max_tokens=500):     # TODO: Ajustar esto para que esté más o menos 1 a 1 con el número de tokens de output posible.
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        word_token_count = len(word) // 4 + 1  # Rough estimate
        if current_token_count + word_token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_token_count = 0
        current_chunk.append(word)
        current_token_count += word_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_chunk(chunk, chunk_index, total_chunks):
    system_message = (
        "You are a highly detailed summarizer. Your task is to provide a comprehensive "
        "summary of the given text, including key points, important details, and any "
        "additional context that might be valuable. If there are any unclear or ambiguous "
        "parts, make educated guesses to fill in the gaps, but indicate when you're doing so."
        "Answer in the language of the provided texts, probably Spanish."
        "Keep in mind that the text is part of a larger transcription, so don't provide introductory or concluding remarks."
    )
    user_message = (
        f"Provide a detailed summary and explanation of the following text. Include all important information and any additional context you can infer. Your summary should be comprehensive and thorough.\n"
        f"#Text to summarize\n{chunk}\n"
        f"#Instructions\n"
        f"1. Use the language of the provided text. Keep your language simple and clear, but try to maintain the complexity of the original text where necessary.\n"
        f"2. Avoid using phrases like 'the text says' or 'the speaker mentions'. Instead, explain the content as if you were teaching it to someone else.\n"
        f"3. If you're making assumptions or inferences to fill in gaps or provide context, clearly indicate when you're doing so. You can use phrases like 'It can be inferred that...' or 'This suggests that...' to mark these instances.\n"
        f"4. Remember that this is part of a larger transcription. Don't provide introductory or concluding remarks that might suggest this is a complete text.\n"
        f"5. Present your summary and explanation in a clear, organized manner. You may use paragraphs or bullet points as appropriate to structure your response.\n"
    )

    response = openaiClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=16383,
        response_format={
          "type": "text"
        }
    )

    return response.choices[0].message.content

def summarize_transcription(transcription_file, output_file):       #TODO: Just send the output_name, not the whole path
    transcription = load_transcription(transcription_file)
    full_text = transcription['full_text']
    
    chunks = chunk_text(full_text)
    summaries = []
    
    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        summary = summarize_chunk(chunk, i, len(chunks))
        summaries.append(summary)
        
    # Ensure the summaries directory exists
    os.makedirs(SUMMARIES_FOLDER, exist_ok=True)

    # Update the output file path
    output_file = os.path.join(SUMMARIES_FOLDER, os.path.basename(output_file))
    
    # Save the summaries
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, summary in enumerate(summaries):
            f.write(f"Summary {i+1}:\n{summary}\n\n")
            
    print(f"Detailed summaries saved to {output_file}")


def process_file(filePath, language="None", custom_prompt=""):
    base_filename = os.path.basename(filePath)
    transcription_output_path = os.path.join(TRANSCRIPTIONS_FOLDER, base_filename.replace(".mp3", "_transcript.json"))
    summary_output_path = os.path.join(SUMMARIES_FOLDER, base_filename.replace(".mp3", "_summary.md"))
    
    file_size_bytes = os.path.getsize(filePath)
    max_chunk_size_bytes = MAX_SIZE_CHUNK

    if file_size_bytes > max_chunk_size_bytes:
        logger.info(f"File size ({file_size_bytes/1024/1024:.2f} MB) exceeds {max_chunk_size_bytes/1024/1024:.2f} MB limit. Splitting into chunks...")
        chunks = split_audio(filePath, max_chunk_size_bytes)
        chunk_transcriptions = []
        
        with ThreadPoolExecutor() as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk, language, custom_prompt, int(chunk.split('_')[1])): chunk for chunk in chunks}
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Processing chunks"):
                chunk = future_to_chunk[future]
                try:
                    transcription, start_time = future.result()
                    if transcription:
                        chunk_transcriptions.append((transcription, start_time))
                        custom_prompt = " ".join([seg["text"] for seg in transcription["segments"][-5:]])
                except Exception as e:
                    logger.error(f"Error processing {chunk}: {e}")
        
        if chunk_transcriptions:
            logger.info("Combining transcriptions from all chunks...")
            full_transcription = combine_transcriptions(chunk_transcriptions)
            save_transcription(full_transcription, transcription_output_path)
        else:
            logger.error("Transcription failed.")

    else:
        logger.info(f"Transcribing {filePath}...")
        transcription = transcribe_audio(filePath, language, custom_prompt)
        if transcription:
            logger.info("Transcription completed.")
            save_transcription(transcription, transcription_output_path)
        else:
            logger.error("Transcription failed.")
    
    # Summarize the transcription
    summarize_transcription(transcription_output_path, summary_output_path)
    
    # Move the processed audio file
    processed_audio_path = os.path.join(PROCESSED_FOLDER, base_filename)
    shutil.move(filePath, processed_audio_path)
    logger.info(f"Moved processed audio file to {processed_audio_path}")

    

def main(folder_path=INPUT_FOLDER, language="None", custom_prompt=""):
    if not os.path.isdir(folder_path):
        logger.error(f"{folder_path} is not a valid directory.")
        return
    
    # Ensure all necessary folders exist
    os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)
    os.makedirs(SUMMARIES_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    audio_files = [f for f in os.listdir(folder_path) if f.endswith((".mp3", ".wav"))]      #TODO: Otros formatos de audio??

    for audio_file in audio_files:
        file_path = os.path.join(folder_path, audio_file)
        logger.info(f"Processing file: {file_path}")
        process_file(file_path, language, custom_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and summarize audio files using Whisper and OpenAI")
    parser.add_argument("-f", "--folder_path", default="audiosPendientes", help="Path to the folder containing audio files")
    parser.add_argument("-l", "--language", default="None", help="Language of the audio (default: auto)")
    parser.add_argument("-p", "--prompt", default="", help="Custom prompt to guide transcription")

    args = parser.parse_args()

    main(args.folder_path, args.language, args.prompt)
