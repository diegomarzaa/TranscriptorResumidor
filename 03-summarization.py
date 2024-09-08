import json
import os
import math
from openai import OpenAI
from tqdm import tqdm

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    response = client.chat.completions.create(
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


def main(transcription_file, output_file):
    # Load transcription
    transcription = load_transcription(transcription_file)
    full_text = transcription['full_text']

    # Chunk the text
    chunks = chunk_text(full_text)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        summary = summarize_chunk(chunk, i, len(chunks))
        summaries.append(summary)
        
    # Save the summaries
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, summary in enumerate(summaries):
            f.write(f"Summary {i+1}:\n{summary}\n\n")
            
    print(f"Detailed summaries saved to {output_file}")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize transcription using GPT-4o-mini")
    parser.add_argument("transcription_file", help="Path to the transcription JSON file")
    parser.add_argument("-o", "--output", default="detailed_summary.md", help="Output file for the summary")
    args = parser.parse_args()

    main(args.transcription_file, args.output)










""" 

# Providing context from the previous and next chunks can help the model create more coherent summaries with less repetition. Let's modify the `summarize_chunk` function to include this context. Here's an updated version of the function:


def summarize_chunk(chunk, chunk_index, total_chunks, prev_chunk=None, next_chunk=None):
    system_message = (
        "You are a highly detailed summarizer. Your task is to provide a comprehensive "
        "summary of the given text, including key points and important details. Focus on "
        "the main chunk, but use the context from previous and next chunks to avoid "
        "repetition and ensure continuity."
    )
    
    context = ""
    if prev_chunk:
        context += f"Previous context: {prev_chunk[-200:]}\n\n"
    if next_chunk:
        context += f"Next context: {next_chunk[:200]}\n\n"
    
    user_message = (
        f"This is chunk {chunk_index + 1} of {total_chunks} from a longer transcription. "
        f"Please provide a detailed summary of the following text, considering the provided context:\n\n"
        f"{context}"
        f"Main chunk to summarize:\n{chunk}\n\n"
        "Include all important information from the main chunk. Use the context to avoid "
        "repeating information that's better covered in other chunks. If you're making "
        "assumptions or inferences, please indicate this clearly."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1500
    )

    return response.choices[0].message.content

def main(transcription_file, output_file):
    # Load transcription
    transcription = load_transcription(transcription_file)
    full_text = transcription['full_text']

    # Chunk the text
    chunks = chunk_text(full_text)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        prev_chunk = chunks[i-1] if i > 0 else None
        next_chunk = chunks[i+1] if i < len(chunks) - 1 else None
        summary = summarize_chunk(chunk, i, len(chunks), prev_chunk, next_chunk)
        summaries.append(summary)

    # Combine summaries
    print("Combining summaries...")
    final_summary = combine_summaries(summaries)

    # Save the final summary
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_summary)

    print(f"Detailed summary saved to {output_file}")


"""