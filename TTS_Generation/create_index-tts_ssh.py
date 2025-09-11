# ==============================================================================
# File: create_index-tts_ssh.py
# Author: Yuncong Yu
# Created: 06/20/2025
#
# Description:
#   Script for batch TTS (Text-to-Speech) generation using the IndexTTS model.
#   Handles dataset ingestion from TSV, multiprocessing, logging, and output
#   audio file management.
#
# Usage:
#   - Configure path variables (BASE_AUDIO_PATH, INPUT_TSV_PATH, etc.)
#   - Run script to generate synthetic audio clips from text prompts.
#   - First time run will download model weigths from Hugging Face, might take time.
#
# ==============================================================================
# Change Log:
# Date        Author            Description
# ----------  ----------------  -----------------------------------------------
# 2025-06-20  Yuncong Yu       Initial version created.
# 2025-09-11  Yuncong Yu       Changes made to santize comments 
# [YYYY-MM-DD] [Contributor]    [Describe change made...]
# ==============================================================================

import csv
import os
import torch
import time
import pandas as pd
from multiprocessing import Process, Queue
from queue import Empty
from tqdm import tqdm
import logging
import soundfile as sf

from indextts.infer import IndexTTS


# ===================== Logging Config =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("index_tts.log"), logging.StreamHandler()]
)

#-----------------------------------Path Config---------------------------------

BASE_AUDIO_PATH = ''        # path contains all original data
INPUT_TSV_PATH = ''         # path to original csv metadata
WAV_OUTPUT_DIR = ''         # folder to store WAV files (if mp3 conversion needed, optional)
GENERATED_OUTPUT_PATH = ''  # path to save generated output
OUTPUT_CSV_PATH = ''        # records the result of generation

NUM_WORKERS = 5             # change number of workers based on GPU memory
TARGET_TEXT = ''            # if set, use this text for all samples; if empty, use text from tsv


class IndexTTSGenerator:
    def __init__(self, tts_model, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tts = tts_model
        print(f"Loading Index-TTS model on device: {self.device}")


    def generate(self, reference_voice_path, target_text, output_path):
        """
        Generate speech waveform from input text.
        Args:
            reference_voice_path
            target_text: text to be converted into wav
            output_path (str): Path to save generated wav.
        Returns:
            output_path (str): Path to generated wav file.
        """
        try:
            self.tts.infer(reference_voice_path, target_text, output_path)
            print(f"Generated audio saved: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error: {e}")
            return None

def worker(task_queue, result_queue, output_dir):
    # initialize tts model for each worker
    tts_model = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
    generator = IndexTTSGenerator(tts_model, device="cuda")

    import re

    while True:
        try:
            sample = task_queue.get(timeout=3)
        except Empty:
            continue

        if sample is None:
            break

        print(f"Worker processing sample: {sample['path']}") # path field contains the filename

        input_audio_path = os.path.join(BASE_AUDIO_PATH, sample['path'])
        target_text = sample['sentence'] # setting target text

        input_filename = os.path.basename(input_audio_path)   # common_voice_en_40865225.mp3
        match = re.search(r'(\d+)', input_filename)
        if match:
            idx_str = match.group(1)  # extracted number
        else:
            idx_str = str(int(time.time()*1000))

        output_filename = f"index-tts_generated_{idx_str}.wav"
        output_path = os.path.join(output_dir, output_filename)
        print(f"Generating audio for sample {sample['path']}: {output_filename}")

        try:
            generated_path = generator.generate(input_audio_path, target_text, output_path)
            result_queue.put([sample['client_id'], os.path.basename(generated_path), target_text])

        except Exception as e:
            print(f"Error generating audio for sample {input_filename}: {e}")



def synthesize_with_multiprocessing(dataset, output_dir, csv_path, num_workers=1):
    os.makedirs(output_dir, exist_ok=True)

    task_queue = Queue()
    result_queue = Queue()
    total_audio_duration = 0.0  # accumulator for total duration (seconds)

    for i, sample in enumerate(dataset):
        sample['index'] = i # create index for each sample for easier naming
        task_queue.put(sample)
    print(f"Adding {len(dataset)} samples to task queue")

    # Add sentinel None for each worker to signal no more tasks
    for _ in range(num_workers):
        task_queue.put(None)

    processes = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(task_queue, result_queue, output_dir))
        p.start()
        processes.append(p)

    total_task = len(dataset)
    generated_files = []
    try:
        with open(csv_path, 'a', newline = '', encoding = 'utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

            with tqdm(total=total_task, desc="Synthesizing Samples") as pbar:
                completed = 0
                while completed < total_task:
                    try:
                        result = result_queue.get(timeout=10)
                        if result:
                            writer.writerow(result)
                            csvfile.flush()

                            # result[1] is filename of generated wav
                            generated_wav_path = os.path.join(output_dir, result[1])
                            if os.path.isfile(generated_wav_path):
                                # Read duration using soundfile
                                with sf.SoundFile(generated_wav_path) as f:
                                    duration = f.frames / f.samplerate
                                total_audio_duration += duration

                        pbar.update(1)
                        completed += 1
                    except Empty:
                        if not any(p.is_alive() for p in processes):
                            logging.warning("All worker processes terminated unexpectedly.")
                            break
                    except Exception as e:
                        logging.error(f"Error Writing to CSV: {e}")
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt detected, terminating workers.")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

    # collect all generated files
    while not result_queue.empty():
        generated_files.append(result_queue.get())

    # Write total duration summary line at the end of CSV
    if len(dataset) > 0:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Total Audio Length(s)', '', total_audio_duration])

        print(f"Finished generating {len(dataset)} samples with {num_workers} workers.")
        print(f"Total generated audio duration: {total_audio_duration:.2f} seconds")
        logging.info(f"Finished generating {len(dataset)} samples. Total audio duration: {total_audio_duration:.2f} seconds")

    else:
        print("No new audio files generated.")


def check_and_initialize_csv(csv_path):
    if not os.path.isfile(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['client_id', 'filename', 'sentence'])
            logging.info(f"Initialized CSV file: {csv_path}")

def load_dataset_with_logging(tsv_path, base_audio_dir, wav_output_dir, log_csv_path):
    """
    Load dataset from TSV, filter out precessed samples logged in CSV
    and prepare dataset for TTS generation

    dataset item:
    {'filename': full_audio_path, 'text': sentence, 'client_id': client_id}

        Args:
      tsv_path (str): path to input TSV with columns including client_id, path, sentence
      base_audio_dir (str): root folder to prepend to audio relative paths in TSV
      wav_output_dir (str): folder to store WAV files (if mp3 conversion needed, optional)
      log_csv_path (str): path to CSV that logs completed samples with columns client_id, filename, sentence

    Returns:
      List[dict]: filtered samples to process
    """
    check_and_initialize_csv(OUTPUT_CSV_PATH)

    try:
        dataset = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        logging.error(f"Failed to read TSV file {tsv_path}: {e}")
        return []

    if os.path.exists(log_csv_path):
        try:
            completed_df = pd.read_csv(log_csv_path)
            completed_set = set(zip(completed_df['client_id'], completed_df['sentence']))
            filtered_data = [
                row for _, row in dataset.iterrows()
                if (row['client_id'], row['sentence']) not in completed_set
            ]
            logging.info(f"Loaded {len(completed_set)} completed samples from log CSV")
            logging.info(f"Excluded {len(dataset) - len(filtered_data)} completed records.")
        except Exception as e:
            logging.warning(f"Failed to read {log_csv_path}: {e}")
            filtered_data = dataset.to_dict('records')
    else:
        filtered_data = dataset.to_dict('records')

    logging.info(f"Total tasks: {len(filtered_data)}")

    return filtered_data


if __name__ == "__main__":
    tts_model_path = "/home/of/index-tts/checkpoints"
    tts_cfg_path = "/home/of/index-tts/checkpoints/config.yaml"

    # loading dataset from csv file
    start_time = time.time()
    dataset = load_dataset_with_logging(INPUT_TSV_PATH, BASE_AUDIO_PATH, WAV_OUTPUT_DIR, OUTPUT_CSV_PATH)
    dataload_time = time.time() - start_time
    print(f"Time to load dataset: {dataload_time}s")
    start_time = time.time()

    # generate new wav samples
    #synthesize_with_multiprocessing(dataset, GENERATED_OUTPUT_PATH, OUTPUT_CSV_PATH, num_workers=NUM_WORKERS)
    if len(dataset) > 0:
        print(f"Time to synthesize {len(dataset)} samples with {NUM_WORKERS} workers: {time.time() - start_time}s")

