import os
import csv
from mutagen.mp3 import MP3
import numpy as np
import torch
import time
import pandas as pd
import random
from tqdm import tqdm
import logging

from dia.model import Dia

#--------------------------------------------------------------------------------------------
# In order to avoid hugging face download/visit limit when running model
# - Run download_dia.py - download the model to local dir ./models
#--------------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("dia_generation.log"), logging.StreamHandler()]
)

 # Server small Test Run
"""
BASE_AUDIO_PATH = '/home/of/test_audio/audio_samples' # path contains all original data
INPUT_TSV_PATH = '/home/of/test_audio/test_tsv_input/testrun.tsv' # path to original csv metadata
WAV_OUTPUT_DIR = '/home/of/test_audio/output/wav_converted_output' # not needed unless wav conversion
GENERATED_OUTPUT_PATH = '/home/of/test_audio/dia_output/generated_mp3' # path to save generated output
OUTPUT_CSV_PATH = '/home/of/test_audio/dia_output/generation_log.csv' # records the result of generation
"""
"""
BASE_AUDIO_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\audio_samples' # path contains all original data
INPUT_TSV_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\test_tsv_input\testrun.tsv' # path to subset csv metadata.
GENERATED_OUTPUT_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\output_dia' # path to generated output
OUTPUT_CSV_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\output_dia\generation_log.csv'
"""


BASE_AUDIO_PATH = '/datasets/cv-corpus/en/clips' # path contains all original data
INPUT_TSV_PATH = '/datasets/cv-corpus/en/other.tsv' # path to original csv metadata
GENERATED_OUTPUT_PATH = '/workspace/generated_data/dia-tts_results/output_wav' # path to save generated output
OUTPUT_CSV_PATH = '/workspace/generated_data/dia-tts_results/generation_log.csv' # records the result of generation'
WAV_OUTPUT_DIR = '' # folder to store WAV files (if mp3-wav conversion needed, optional)



BATCH_SIZE = 3 # set based on GPU memory.

EMOTIONS = [
    "(laughs)", "(clears throat)", "(sighs)", "(gasps)", "(coughs)", "(singing)", "(sings)",
    "(mumbles)", "(beep)", "(groans)", "(sniffs)", "(claps)", "(screams)", "(inhales)",
    "(exhales)", "(applause)", "(burps)", "(humming)", "(sneezes)", "(chuckle)", "(whistles)"
]

SPEAKERS = ["[S1]", "[S2]"]



class DiaGenreator:
    def __init__(self, dia_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dia_model = dia_model
        logging.info(f"Loading Dia model on device {self.device}")

    def generate_batch(self, text_list, reference_audio_list, sentence_id_list, output_dir, csv_path, batch_size=32, compile=False):
        os.makedirs(output_dir, exist_ok=True)
        generated_sample_count = 0
        prev_duration = get_previous_duration(csv_path)
        total_audio_duration = prev_duration

        with open(csv_path, 'a', newline = '', encoding = 'utf-8') as csvfile:
            writer = csv.writer(csvfile)

            with tqdm(total=len(text_list), desc="Generating batch") as pbar:
                for i in range(0, len(text_list), batch_size):
                    batch_texts = text_list[i:i + batch_size]
                    batch_audios = reference_audio_list[i:i + batch_size]
                    batch_sentence_ids = sentence_id_list[i:i + batch_size]

                    try:
                        logging.info(f"Generating batch {i} to {i + len(batch_texts) - 1}")
                        logging.info(f"Batch audio contains {batch_audios}")
                        output_audios = self.dia_model.generate(
                            batch_texts,
                            audio_prompt=batch_audios,
                            use_torch_compile=compile,
                            verbose=True
                        )
                        # Ensure it's always a list
                        if isinstance(output_audios, np.ndarray):
                            output_audios = [output_audios]

                        for j, audio in enumerate(output_audios):
                            basename = os.path.basename(batch_audios[j])  # e.g., "common_voice_en_40865211.mp3"
                            number = basename.replace(".mp3", "").split("_")[-1]
                            filename = f"dia_generated_{number}.mp3"
                            self.dia_model.save_audio(os.path.join(output_dir, filename), audio)
                            logging.info(f"Generated samples: {filename}")
                            print(f"Generated samples: {filename}")
                            generated_sample_count += 1
                            generated_mp3_path = os.path.join(output_dir, filename)
                            # check generated mp3 length, add it to total duration
                            if os.path.isfile(generated_mp3_path):
                                audio = MP3(generated_mp3_path)
                                duration = audio.info.length
                                total_audio_duration += duration

                            # write results to csv
                            sentence_id = batch_sentence_ids[j]
                            writer.writerow([sentence_id, filename, batch_texts[j]])

                        logging.info(f"Saved batch {i} to {i + len(batch_texts) - 1}")
                        print(f"Saved batch {i} to {i + len(batch_texts) - 1}")
                        pbar.update(len(batch_texts))
                        torch.cuda.empty_cache()

                    except Exception as e:
                        logging.error(f"Error in batch {i}-{i + batch_size - 1}: {e}")
                        continue  # Skip this batch and move on

            writer.writerow(['Total Audio Length(s)', '', total_audio_duration])

        print(f"Finished generating {generated_sample_count} samples.")
        print(f"Audio generated in this session: {(total_audio_duration - prev_duration):.2f} seconds")
        print(f"Total generated audio: {total_audio_duration:.2f} seconds")

        logging.info(f"Finished generating {generated_sample_count} samples. Total audio duration: {total_audio_duration:.2f} seconds")
        logging.info(f"Audio generated in this session: {(total_audio_duration - prev_duration):.2f} seconds")
        logging.info(f"Total generated audio: {total_audio_duration:.2f} seconds")

        return generated_sample_count

def get_duration(audio_path):
    mp3_path = os.path.join(BASE_AUDIO_PATH, audio_path)
    audio = MP3(mp3_path)
    duration = audio.info.length
    return duration

def generate_coversation(dataset, base_audio_dir, target_min=10, target_max=15, emotion_prob=0.3):
    text_list = []
    reference_audio_list = []
    client_id_list = []

    current_conv = []
    current_sentences = []
    current_paths = []
    current_duration = 0.0
    speaker_idx = 0

    for i, row in enumerate(dataset):
        audio_path = os.path.join(base_audio_dir, row['path'])
        duration = get_duration(audio_path)

        if duration > 20:
            continue

        if current_duration + duration > target_max:
            if current_duration >= target_min:
                # Build full conversation
                full_convo = " ".join(current_conv)
                # For each original sentence, create a separate input
                for sent, path in zip(current_sentences, current_paths):
                    combined = f"{sent} {full_convo}"
                    text_list.append(combined)
                    reference_audio_list.append(path)
            # reset for next group
            current_conv = []
            current_sentences = []
            current_paths = []
            current_duration = 0.0
            speaker_idx = 0

        # Speaker and sentence with emotion
        speaker = SPEAKERS[speaker_idx % len(SPEAKERS)]
        sentence = row['sentence']
        if random.random() < emotion_prob:
            emotion = random.choice(EMOTIONS)
            sentence = f"{emotion} {sentence}" if random.random() < 0.5 else f"{sentence} {emotion}"

        spoken_line = f"{speaker} {sentence}"
        current_conv.append(spoken_line)
        client_id_list.append(row['sentence_id'])
        current_sentences.append(row['sentence'])
        current_paths.append(row['path'])
        current_duration += duration
        speaker_idx += 1

    # Final group (if leftover)
    if current_conv and (current_duration >= target_min or len(current_conv) >= 1):
        full_convo = " ".join(current_conv)
        for sent, path in zip(current_sentences, current_paths):
            combined = f"{sent} {full_convo}"
            text_list.append(combined)
            reference_audio_list.append(path)

    return text_list, reference_audio_list, client_id_list

# gets the previous duration recorded in the csv file
def get_previous_duration(csv_path):
    """
    Reads the last line of the CSV file and returns the previous total audio duration
    if it ends with a 'Total Audio Length(s)' row. Returns 0.0 if not found or invalid.
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
            if lines:
                last_row = next(csv.reader([lines[-1]]))
                if last_row and last_row[0].strip() == 'Total Audio Length(s)':
                    try:
                        return float(last_row[2])
                    except ValueError:
                        return 0.0
    except Exception as e:
        logging.warning(f"Could not read previous total duration from last line: {e}")
    return 0.0

def check_and_initialize_csv(csv_path):
    if not os.path.isfile(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sentence_id', 'filename', 'sentence'])
            logging.info(f"Initialized CSV file: {csv_path}")

def load_dataset(tsv_path, log_csv_path):
    os.makedirs(GENERATED_OUTPUT_PATH, exist_ok=True) # create the output_chatter folder
    check_and_initialize_csv(OUTPUT_CSV_PATH) # check if there is a output csv already exist

    try:
        dataset = pd.read_csv(tsv_path, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8', engine='python')  # read input tsv
    except Exception as e:
        logging.error(f"Failed to read TSV file {tsv_path}: {e}")
        return []

    if os.path.exists(log_csv_path):
        try:
            # directly load tsv into filtered_data using the table cols, only using sentence_id, and sentence
            completed_df = pd.read_csv(log_csv_path)
            completed_set = set(completed_df['sentence_id'])
            filtered_data = [
                row for _, row in dataset.iterrows()
                if (row['sentence_id']) not in completed_set
            ]
            logging.info(f"Loaded {len(dataset)} completed samples from log CSV")
            logging.info(f"Excluded {len(dataset) - len(filtered_data)} completed records.")
        except Exception as e:
            logging.warning(f"Failed to read {log_csv_path}: {e}")
            filtered_data = dataset.to_dict('records')
    else:
        filtered_data = dataset.to_dict('records')

    logging.info(f"Total tasks: {len(filtered_data)}")

    valid_data = []   # store the valid sample with length < 20s (filter again from filtered_data)
    for row in filtered_data:
        try:
            mp3_path = os.path.join(BASE_AUDIO_PATH, row['path'])
            audio = MP3(mp3_path)
            duration = audio.info.length
            if duration <= 20:
                row['path'] = mp3_path
                valid_data.append(row)
            else:
                logging.info(f"Skipped {row['path']} due to duration > 20s ({duration:.2f}s)")
        except Exception as e:
            logging.warning(f"Could not process {row['path']}: {e}")

    logging.info(f"Total valid tasks after duration filtering: {len(valid_data)}")
    return valid_data

# display the generated conversation for debugging, called by main
def display_input(text_list, reference_audio_list, client_id_list):
    for t, a in zip(text_list, reference_audio_list):
        print(f"INPUT TEXT:\n{t}\nREFERENCE AUDIO:\n{a}\n{'-' * 50}")
    print(f"Length of generated conversation: {len(text_list)}")
    print(f"Length of reference audio list: {len(reference_audio_list)}")
    print(f"Length of client id list: {len(client_id_list)}")
    print("-" * 50)

if __name__ == "__main__":
    # loading dataset
    start = time.time()
    dataset = load_dataset(INPUT_TSV_PATH, OUTPUT_CSV_PATH)
    dataload_time = time.time()- start
    logging.info(f"Dataset loaded in {dataload_time:.2f} seconds")
    start = time.time()

    if len(dataset) == 0:
        logging.info(f"No tasks left in the dataset, no new samples will be created.")
        print("No tasks left in the dataset, no new samples will be created.")
        logging.info(f"Program finished.")
    else:
        text_list, reference_audio_list, client_id_list = generate_coversation(dataset, BASE_AUDIO_PATH)
        #display_input(text_list, reference_audio_list, client_id_list)

        # Create Dia model and generate in mini-batch
        dia_model = Dia.from_local("models/dia/config.json", checkpoint_path="models/dia/dia-v0_1.pth")
        generator = DiaGenreator(dia_model)
        generated_sample_count = generator.generate_batch(text_list, reference_audio_list, client_id_list,
                                 GENERATED_OUTPUT_PATH,
                                 OUTPUT_CSV_PATH,
                                 batch_size=BATCH_SIZE,
                                 compile=False)
        print(f"Generating completed in {(time.time() - start):.2f} seconds")

