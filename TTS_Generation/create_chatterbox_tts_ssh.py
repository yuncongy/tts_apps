import os
import csv
#from pydub import AudioSegment
from mutagen.mp3 import MP3
import torch, torchaudio as ta
import time
import pandas as pd
import numpy as np
import random
from multiprocessing import Process, Queue
from queue import Empty
from tqdm import tqdm
import logging
import soundfile as sf

from src.chatterbox.tts import ChatterboxTTS

#--------------------------------------------------------------------------------------------
# In order to avoid hugging face download/visit limit when running model
# - Run download_model.py - download the model to local dir ./models
# - change model path in tts.py
#--------------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("chatterbox_tts_generation.log"), logging.StreamHandler()]
)

 # Server small Test Run
# BASE_AUDIO_PATH = '/home/of/test_audio/audio_samples' # path contains all original data
# INPUT_TSV_PATH = '/home/of/test_audio/test_tsv_input/testrun.tsv' # path to original csv metadata
# WAV_OUTPUT_DIR = '/home/of/test_audio/output/wav_converted_output'
# GENERATED_OUTPUT_PATH = '/home/of/test_audio/chatterbox_output/generated_wav' # path to save generated output
# OUTPUT_CSV_PATH = '/home/of/test_audio/chatterbox_output/generation_log.csv' # records the result of generation



BASE_AUDIO_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\audio_samples' # path contains all original data
INPUT_TSV_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\test_tsv_input\testrun.tsv' # path to subset csv metadata.
GENERATED_OUTPUT_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\output_chatterbox' # path to generated output
OUTPUT_CSV_PATH = r'E:\ML\Datasets\cv-corpus\test_audio\output_chatterbox\generation_log.csv'


"""
BASE_AUDIO_PATH = '/datasets/cv-corpus/en/clips' # path contains all original data
INPUT_TSV_PATH = '/datasets/cv-corpus/en/other.tsv' # path to original csv metadata
GENERATED_OUTPUT_PATH = '/workspace/generated_data/chatterbox-tts_results/output_wav' # path to save generated output
OUTPUT_CSV_PATH = '/workspace/generated_data/chatterbox-tts_results/generation_log.csv' # records the result of generation'
WAV_OUTPUT_DIR = '' # folder to store WAV files (if mp3-wav conversion needed, optional)
"""


NUM_WORKERS = 2 # set based on GPU memory


class ChatterBoxTTSGenerator:
    def __init__(self, chatterbox_model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chatterbox_model = chatterbox_model
        logging.info(f"Loading ChatterBox TTS model on device: {self.device}")

    def generate(self, model, target_text, input_audio_path, output_path):
        """
        Args:
            model
            target_text: text to be converted into wav
            input_audio_path: reference audio file path
            output_path: Path to generated wav files

        Returns:
            wav_files: generated wav file torch tensor
        """
        # setting randomized parameters
        use_default_audio = False
        if random.random() < 0.1: # 10% chance to use default parameters
            use_default_audio = True # default reference audio
            exaggeration = 0.5
            cfg_weight = 0.5
            temperature = 0.8
        else:
            # Use normal distribution
            # exaggeration = sample_parameter(mean=1.0, std=0.3, min_val=0.25, max_val=1.7)
            # cfg_weight = sample_parameter(mean=0.6, std=0.2, min_val=0.2, max_val=1.0)
            temperature = sample_parameter(mean=0.8, std=0.2, min_val=0.05, max_val=1.5)  # value > 1.3 might cause gibberish audio

            # use beta distribution (large values will be unstable)
            exaggeration = sample_beta_scaled(alpha=1.8, beta=5, min_val=0.25, max_val=2.0)
            cfg_weight = sample_beta_scaled(alpha=3.5, beta=3.5, min_val=0.2, max_val=0.9)
            #temperature = sample_beta_scaled(alpha=1.5, beta=7.0, min_val=0.05, max_val=1.5) # large temperature will mess up audio
        logging.info(
            f"Running model for audio sample {os.path.basename(input_audio_path)} with params: "
            f"Default:{use_default_audio}, exag:{exaggeration}, cfg_weight:{cfg_weight}, temperature:{temperature}")

        # if input_audio_path.endswith('212.mp3') or input_audio_path.endswith('213.mp3'):
        #     return None

        if use_default_audio:
            try:
                wav_file = self.chatterbox_model.generate(target_text)
                if wav_file is None:
                    logging.warning(f"Model returned None for {input_audio_path}, skipping.")
                    return None
                ta.save(output_path, wav_file, model.sr)
                return wav_file
                logging.info(f"Successfully generated audio for {os.path.basename(input_audio_path)}")
            except Exception as e:
                logging.error(f"Exception while generating {input_audio_path}: {e}")
                return None
        else:
            try:
                wav_file = self.chatterbox_model.generate(target_text,
                                                          audio_prompt_path=input_audio_path,
                                                          exaggeration=exaggeration,
                                                          cfg_weight=cfg_weight,
                                                          temperature=temperature
                                                          )
                if wav_file is None:
                    logging.warning(f"Model returned None for {input_audio_path}, skipping.")
                    return None
                ta.save(output_path, wav_file, model.sr)
                logging.info(f"Successfully generated audio for {os.path.basename(input_audio_path)}")
                return wav_file
            except Exception as e:
                logging.error(f"Error generating audio {os.path.basename(input_audio_path)}: {e}")
                return None


# model sample parameter scaled using normal distribution
def sample_parameter(mean, std, min_val, max_val):
    sampled = np.random.normal(loc=mean, scale=std)
    clipped = np.clip(sampled, min_val, max_val)
    return round(float(clipped), 2)

# model sample parameters scaled using beta distribution
def sample_beta_scaled(alpha, beta, min_val, max_val):
    sample = np.random.beta(alpha, beta)  # in [0, 1]
    scaled = min_val + sample * (max_val - min_val)
    return round(float(scaled), 2)

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


def worker(task_queue, result_queue, output_dir):

    # each worker gets a unique seed based on time and its process ID.
    random.seed(time.time() + os.getpid())
    np.random.seed(int(time.time() + os.getpid()) % 2 ** 32)

    # initialize chatterbox model for each worker
    chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
    chatterbox_generator = ChatterBoxTTSGenerator(chatterbox_model)

    # getting output filename path and format.
    import re

    while True:
        try:
            sample = task_queue.get(timeout=3)
        except Empty:
            continue

        if sample is None:
            break

        print(f"Worker processing sample: {sample['path']}")

        input_audio_path = os.path.join(BASE_AUDIO_PATH, sample['path'])
        target_text = sample['sentence']

        input_filename = os.path.basename(input_audio_path)   # common_voice_en_40865225.mp3
        # getting correct file name number for output, 000000.wav
        match = re.search(r'(\d+)', input_filename)
        if match:
            idx_str = match.group(1)  # extracted number
        else:
            idx_str = str(int(time.time() * 1000))

        output_filename = f"chatterbox_generated_{idx_str}.wav"
        output_path = os.path.join(output_dir, output_filename)
        print(f"Generating audio for sample {sample['path']}: {output_filename}")


        # Generate using chatterbox-tts using set parameters
        try:
            wav = chatterbox_generator.generate(chatterbox_model, target_text, input_audio_path, output_path) # output of the model. wav tensor
            if wav is not None:
                result_queue.put([sample['client_id'], output_filename, target_text]) # add generated wav filename to the result queue
        except Exception as e:
            print(f"[Error] when generating audio for sample {sample['path']}: {e}")

def synthesize_with_multiprocessing(dataset, output_dir, csv_path, num_workers=1):
    os.makedirs(output_dir, exist_ok=True)

    task_queue = Queue()
    result_queue = Queue()
    generated_sample_count = 0
    previs_duration = get_previous_duration(csv_path)
    total_audio_duration = previs_duration

    for i, sample in enumerate(dataset):
        sample['index'] = i  # create index for each sample for easier naming
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
                                generated_sample_count += 1

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

        print(f"Finished generating {generated_sample_count} samples with {num_workers} workers.")
        print(f"Audio generated in this session: {(total_audio_duration - previs_duration):.2f} seconds")
        print(f"Total generated audio: {total_audio_duration:.2f} seconds")

        logging.info(f"Finished generating {generated_sample_count} samples. Total audio duration: {total_audio_duration:.2f} seconds")
        logging.info(f"Audio generated in this session: {(total_audio_duration - previs_duration):.2f} seconds")
        logging.info(f"Total generated audio: {total_audio_duration:.2f} seconds")

    return generated_sample_count


def check_and_initialize_csv(csv_path):
    if not os.path.isfile(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['client_id', 'filename', 'sentence'])
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
            # directly load tsv into filtered_data using the table cols, only using client_id, and sentence
            completed_df = pd.read_csv(log_csv_path)
            completed_set = set(zip(completed_df['client_id'], completed_df['sentence']))
            filtered_data = [
                row for _, row in dataset.iterrows()
                if (row['client_id'], row['sentence']) not in completed_set
            ]
            logging.info(f"Loaded {len(dataset)} completed samples from log CSV")
            logging.info(f"Excluded {len(dataset) - len(filtered_data)} completed records.")
        except Exception as e:
            logging.warning(f"Failed to read {log_csv_path}: {e}")
            filtered_data = dataset.to_dict('records')
    else:
        filtered_data = dataset.to_dict('records')

    logging.info(f"Total tasks for audio generation: {len(filtered_data)}")

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


if __name__ == "__main__":
    # loading dataset
    start = time.time()
    dataset = load_dataset(INPUT_TSV_PATH, OUTPUT_CSV_PATH)
    dataload_time = time.time()- start
    logging.info(f"Dataset loaded in {dataload_time:.2f} seconds")
    start = time.time()

    # initializing chatterbox model
    result = synthesize_with_multiprocessing(dataset, GENERATED_OUTPUT_PATH, OUTPUT_CSV_PATH, NUM_WORKERS) # return 0 or 1, 0 for success run, 1 for contains problem

    logging.info(f"Time to synthesize {result} samples with {NUM_WORKERS} workers: {(time.time() - start):.2f}s")
    print(f"Time to synthesize {result} samples with {NUM_WORKERS} workers: {(time.time() - start):.2f}s")


# if LLVM ERROR appear, set these variables. AMD chip causing intel MKL error.
"""
set MKL_THREADING_LAYER=GNU
set NUMBA_DISABLE_INTEL_SVML=1
python create_chatterbox_tts_ssh.py
"""
