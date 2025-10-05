import gradio as gr
import onnx_asr
import torch
from pydub import AudioSegment
from pydub.effects import normalize
import numpy as np
import csv
import pprint
import os
import pandas as pd  # Added this import
from datetime import datetime  # Added this import

# Function to convert timestamps into sentence timestamps
def convert_to_sentence_timestamps(timestamps, tokens):
    sentence_timestamps = []
    start_time = None
    end_time = None
    current_tokens = []

    for i, token in enumerate(tokens):
        if token in {'.', '!', '?'}:
            if start_time is not None:
                end_time = timestamps[i]
                current_tokens.append(token)
                segment = ''.join(current_tokens).strip()
                sentence_timestamps.append({
                    'start': f"{start_time:.2f}",
                    'end': f"{end_time:.2f}",
                    'segment': segment
                })
                start_time = None
                end_time = None
                current_tokens = []
        else:
            if start_time is None:
                start_time = timestamps[i]
            current_tokens.append(token)

    return sentence_timestamps

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#providers = ['CPUExecutionProvider']

def process_audio(audio_file, chunk_duration):
    # Load model here (only when needed)
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=providers).with_timestamps()

    try:
        # Load audio file
        sound = AudioSegment.from_file(audio_file, channels=1)

        # Process audio
        ch = 1
        sw = 2
        fr = 16000
        sound = normalize(sound)
        sound = sound.set_channels(ch)
        sound = sound.set_sample_width(sw)  # PCM_16 format
        sound = sound.set_frame_rate(fr)

        # Process audio in X second chunks
        chunk_duration = chunk_duration * 1000  # X seconds in milliseconds
        total_duration = len(sound)

        start_time = 0
        end_time = 0
        final_chunk = 0
        item = 0
        sentence_timestamps = []


        while start_time < total_duration:
            # Calculate end time for this chunk
            print(f"Start time:{start_time/1000:.2f}s")
            end_time = min(start_time + chunk_duration, total_duration)
            print(f"chunk: {start_time/1000:.2f}s - {end_time/1000:.2f}s")
            # Extract audio chunk
            chunk = sound[start_time:end_time]
            chunk_len = len(chunk)
            if len(chunk) < chunk_duration:
                print("Final chunk start")
                final_chunk = 1

            print(f"Current chunk length: {(chunk_len/1000):.2f}s")

            # Convert chunk to numpy array
            chunk_array = np.array(chunk.get_array_of_samples())

            # Process chunk
            output = model.recognize(chunk_array)

            chunk_timestamps = convert_to_sentence_timestamps(output.timestamps, output.tokens)
            end_index = len(chunk_timestamps) - 2 if not final_chunk else len(chunk_timestamps)
            last_timestamp = start_time
            current_timestamps = []
            for i in range(end_index):
                item += 1
                timestamps = chunk_timestamps[i]
                timestamps['start'] = f"{(float(timestamps['start']) + start_time / 1000):.2f}"
                timestamps['end'] = f"{(float(timestamps['end']) + start_time / 1000):.2f}"
                last_timestamp = float(timestamps['end'])

                current_timestamps.append(timestamps)

            start_time = last_timestamp * 1000

            # Add timestamps with global offset
            sentence_timestamps.extend(current_timestamps)
            item += 1
            if final_chunk == 1:
                break

        # Convert to table format
        table_data = []
        for i, timestamp in enumerate(sentence_timestamps):
            table_data.append([
                i + 1,
                timestamp['start'],
                timestamp['end'],
                timestamp['segment']
            ])

        return table_data, sentence_timestamps
    finally:
        # Clean up model after processing
        del model
        # Optional: Force garbage collection
        import gc
        gc.collect()

def save_csv(timestamps, filename):
    """Save timestamps to CSV file"""
    # Convert timestamps to proper format if needed
    if isinstance(timestamps, pd.DataFrame):
        # If it's already a DataFrame, use it directly
        df = timestamps
    else:
        # If it's a list or other format, convert it
        df = pd.DataFrame(timestamps)

    # Ensure we have the right column names
    if len(df.columns) >= 4:
        df.columns = ['Index', 'Start (s)', 'End (s)', 'Segment']
    else:
        # Handle case where we get a list of dicts or similar
        df = pd.DataFrame(timestamps)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{filename}_{timestamp_str}.csv"
    csv_path = os.path.join("output", csv_filename)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Save the dataframe
    df.to_csv(csv_path, index=False)
    return csv_path

def save_srt(timestamps, filename):
    """Save timestamps to SRT file"""
    # Convert to proper format if needed
    if isinstance(timestamps, pd.DataFrame):
        df = timestamps
    else:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(timestamps)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    srt_filename = f"{filename}_{timestamp_str}.srt"
    srt_path = os.path.join("output", srt_filename)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Generate SRT content
    srt_content = []
    for i, row in df.iterrows():
        # Handle both DataFrame rows and list/dict formats
        if isinstance(row, pd.Series):
            # For DataFrame case, extract values by column name
            index = i + 1
            #pprint.pprint(row)
            start_time = float(row['start']) if 'start' in row else float(row.iloc[0])
            end_time = float(row['end']) if 'end' in row else float(row.iloc[1])
            segment = str(row['segment']) if 'segment' in row else str(row.iloc[2])
        else:
            # Handle list/dict format - properly extract data
            try:
                index = i + 1
                start_time = float(row[0])  # start time (index 1)
                end_time = float(row[1])    # end time (index 2)
                segment = str(row[2])     # segment text (index 3)
            except (ValueError, IndexError):
                # If conversion fails or index is out of bounds, skip this row
                continue

        # Convert seconds to SRT time format
        def seconds_to_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millisecs = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

        srt_content.append(str(index))
        srt_content.append(f"{seconds_to_srt_time(start_time)} --> {seconds_to_srt_time(end_time)}")
        srt_content.append(segment)
        srt_content.append("")  # Empty line between subtitles

    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))

    return srt_path

def download_csv(timestamps):
    """Download timestamps as CSV"""
    try:
        csv_path = save_csv(timestamps, "timestamps")
        return csv_path
    except Exception as e:
        print(f"Error in download_csv: {e}")
        return None

def download_srt(timestamps):
    """Download timestamps as SRT"""
    try:
        srt_path = save_srt(timestamps, "timestamps")
        return srt_path
    except Exception as e:
        print(f"Error in download_srt: {e}")
        return None

def generate_files(timestamps):
    csv_path = download_csv(timestamps)
    srt_path = download_srt(timestamps)
    new_csv_btn = gr.DownloadButton(label="Download CSV", value=csv_path, visible=True)
    new_srt_btn = gr.DownloadButton(label="Download SRT", value=srt_path, visible=True)
    return new_csv_btn, new_srt_btn 
# Add CSS to hide sort buttons
custom_css = """
.cell-menu-button{
    display: none !important;
}
"""

def parakeet_ui():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# Nvidia Parakeet v3 Timestamp Processor")
        gr.Markdown("Upload an audio file, then click Transcribe to process timestamps with parakeet-tdt-0.6b-v3-onnx.")

        timestamps_state = gr.State()

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            chunk_duration_slider = gr.Slider(
                minimum=10,
                maximum=400,
                value=150,
                step=1,
                label="Chunk Duration (seconds)"
            )

        transcribe_btn = gr.Button("Transcribe")

        with gr.Row():
            csv_btn = gr.DownloadButton(label="Download CSV", visible=False)
            srt_btn = gr.DownloadButton(label="Download SRT", visible=False)

        with gr.Row():
            table_output = gr.Dataframe(
                headers=["Index", "Start (s)", "End (s)", "Segment"],
                datatype=["number", "number", "number", "str"],
                label="Timestamps",
                interactive=False
            )



        # Process audio when button is clicked
        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, chunk_duration_slider],
            outputs=[table_output, timestamps_state]
        )

        timestamps_state.change(
            fn=generate_files,
            inputs=[timestamps_state],
            outputs=[csv_btn, srt_btn]
        )

    return demo

def extension__tts_generation_webui():
    parakeet_ui()
    return {
        "package_name": "tts_webui_extension.parakeet",
        "name": "Parakeet",
        "requirements": "git+https://github.com/mefich/tts_webui_extension.parakeet@main",
        "description": "Speech transcription via Nvidia Parakeet model",
        "extension_type": "interface",
        "extension_class": "tools",
        "author": "mefi",
        "extension_author": "mefi",
        "license": "BSD-3",
        "website": "https://github.com/mefich/tts_webui_extension.parakeet",
        "extension_website": "https://github.com/mefich/tts_webui_extension.parakeet",
        "extension_platform_version": "0.0.1",
    }

# For direct execution - this will be ignored when used as an extension
if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()

    with gr.Blocks() as demo:
        with gr.Tab("Parakeet"):
            parakeet_ui()

    demo.queue().launch()
