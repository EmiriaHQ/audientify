import soundfile as sf
import librosa
from pathlib import Path 
from audio_segment import VADJob
import os
import pandas as pd
import numpy as np
import subprocess
import csv
import argparse
from youtube import YouTubeDownloader
import yt_dlp
import pandas as pd
from scipy.spatial.distance import cosine
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import numpy as np
import contextlib
import wave
from pyannote.audio import Audio
from pyannote.core import Segment
from typing import List, Tuple
from sklearn.cluster import DBSCAN
import os
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi
from pydub.playback import play

def play_audio_segment(audio_file_path: str, start_time: float, end_time: float):
    """Play a segment of audio between start_time and end_time (in seconds)"""
    audio = AudioSegment.from_wav(audio_file_path)
    segment = audio[start_time*1000:end_time*1000]  # Convert to milliseconds
    play(segment)

def generate_speaker_embeddings(
    meeting_file_path: str, transcript
) -> np.ndarray:
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", device="cpu"
    )

    embeddings = np.zeros(shape=(len(transcript), 192))

    with contextlib.closing(wave.open(str(meeting_file_path), "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    for i, segment in transcript.iterrows():
        embeddings[i] = segment_embedding(
            meeting_file_path, duration, segment, embedding_model
        )

    embeddings = np.nan_to_num(embeddings)

    return embeddings

def segment_embedding(
    file_name: str,
    duration: float,
    segment,
    embedding_model: PretrainedSpeakerEmbedding,
) -> np.ndarray:
    audio = Audio()
    segment["start"] = segment["start"] / 1000
    segment["end"] = segment["end"] / 1000
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(file_name, clip)
    
    return embedding_model(waveform[None])

def reference_audio_embedding(file_name: str) -> np.ndarray:
    audio = Audio()
    waveform, sample_rate = audio(file_name)
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", device="cpu"
    )
    return embedding_model(waveform[None])[0]

def closest_reference_speaker(
    embedding: np.ndarray,
    references: List[Tuple[str, np.ndarray]],
    threshold: float = 0.5,
) -> str:
    min_distance = float("inf")
    closest_speaker = None

    for name, reference_embedding in references:
        distance = cosine(embedding, reference_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_speaker = name
    if min_distance > threshold:
        return "不明な話者"
    else:
        return closest_speaker


def format_speaker_output_by_segment2(
    audio_file_path: str,
    embeddings: np.ndarray,
    transcript: pd.DataFrame,
    reference_embeddings: List[Tuple[str, np.ndarray]],
    prompt_reference,
) -> pd.DataFrame:
    labeled_segments = []
    unknown_speaker_embeddings = []  # Store embeddings of unknown speakers
    unknown_speaker_indices = []  # Store indices of unknown speakers

    for i, (embedding, segment) in enumerate(
        zip(embeddings, transcript.itertuples())
    ):
        speaker_name = closest_reference_speaker(
            embedding, reference_embeddings
        )
        if speaker_name == "不明な話者":
            if prompt_reference:
                play_audio_segment(audio_file_path, segment.start/1000, segment.end/1000)
                
                # Display existing reference audios for selection
                if reference_embeddings:
                    print("Existing reference audios:")
                    for idx, (name, _) in enumerate(reference_embeddings):
                        print(f"{idx}. {name}")
                    choice = input("Enter the number of the speaker or a new speaker name: ")
                    
                    # If the user's choice is a number and valid, use the existing reference
                    if choice.isdigit() and int(choice) in range(len(reference_embeddings)):
                        speaker_name = reference_embeddings[int(choice)][0]
                    else:
                        # Otherwise, treat the choice as a new speaker name and create a new reference audio
                        new_name = choice
                        new_reference_path = f"reference_audio/{new_name}.wav"
                        AudioSplitter(audio_file_path, segment.start/1000, segment.end/1000, new_reference_path).split()
                        new_embedding = reference_audio_embedding(new_reference_path)
                        reference_embeddings.append((new_name, new_embedding))
                        speaker_name = new_name
                else:
                    new_name = input("不明な話者が見つかりました。 スピーカー名を入力: ")
                    new_reference_path = f"reference_audio/{new_name}.wav"
                    AudioSplitter(audio_file_path, segment.start/1000, segment.end/1000, new_reference_path).split()
                    new_embedding = reference_audio_embedding(new_reference_path)
                    reference_embeddings.append((new_name, new_embedding))
                    speaker_name = new_name
            else:
                unknown_speaker_embeddings.append(embedding)
                unknown_speaker_indices.append(i)
        else:
            labeled_segments.append(
                (segment.start, segment.end, speaker_name, segment.text, segment.break_until_next)
            )
    if unknown_speaker_embeddings:
 
        dbscan = DBSCAN(metric="cosine", eps=0.5, min_samples=1)
        labels = dbscan.fit_predict(unknown_speaker_embeddings)

        for i, label in zip(unknown_speaker_indices, labels):
            segment = transcript.iloc[i]
            speaker_name = f"不明な話者{label + 1}"
            labeled_segments.insert(
                i, (segment.start, segment.end, speaker_name, segment.text, segment.break_until_next)
            )

    output_df = pd.DataFrame(
        labeled_segments, columns=["start", "end", "speaker", "text", "break_until_next"]
    )

    return output_df, reference_embeddings

def ensure_16bit_mono_wav(file_path: Path) -> None:
    """Ensure that the given wav file is 16bit and mono. Convert if not."""
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

    # If the file is not 16bit or not mono, convert it
    if n_channels != 1 or sampwidth != 2:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        sf.write(file_path, y, sr, subtype='PCM_16')

class AudioSplitter:
    def __init__(self, audio_fp: Path, start_sec: int, end_sec: int, out_fp: Path):
        self.audio_fp = audio_fp
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.out_fp = out_fp

    def split(self):
        duration = self.end_sec - self.start_sec
        command = [
            "ffmpeg", "-y",
            "-ss", str(self.start_sec),
            "-i", str(self.audio_fp),
            "-t", str(duration),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(self.out_fp)
        ]
        subprocess.run(command)

def main():

    parser = argparse.ArgumentParser(description='Process audio and transcript.')
    parser.add_argument('audio', type=str, help='Path to the audio file or YouTube URL')
    parser.add_argument('transcript', type=str, help='Path to the transcript file')
    parser.add_argument('--use-yt-transcript', action='store_true', help='Use youtube_transcript_api to fetch transcript')
    parser.add_argument('--lang', default='ja', type=str, help='Language for YouTube transcript. Default is "ja" for Japanese.')
    parser.add_argument('--prompt-reference', action='store_true', 
                    help='Prompt to save unknown speaker segments as reference audio.')

    
    args = parser.parse_args()

    if not os.path.exists("content"):
        os.makedirs("content")

    if not os.path.exists("reference_audio"):
        os.makedirs("reference_audio")

    # input_argがYouTubeのリンクだったら動画を16bitのwavとしてダウンロードし、そうでなければ、単にパスを指定する

    if YouTubeDownloader.is_youtube_url(args.audio):
        print(f"Download YouTube {yt_dlp.extractor.youtube.YoutubeIE.extract_id(args.audio)} Video...")
        downloader = YouTubeDownloader("content")
        downloader.download_audio(args.audio)
        yt_id = yt_dlp.extractor.youtube.YoutubeIE.extract_id(args.audio)
        wav_data_fp = Path(os.path.join("content", yt_id, "audio.wav"))
    else:
        wav_data_fp = Path(args.audio)
        ensure_16bit_mono_wav(wav_data_fp)

    if args.use_yt_transcript:

        transcript = YouTubeTranscriptApi.get_transcript(yt_dlp.extractor.youtube.YoutubeIE.extract_id(args.audio),languages=[args.lang])

        save_dir = os.path.join("content", yt_dlp.extractor.youtube.YoutubeIE.extract_id(args.audio), 'transcript.csv')

        with open(save_dir, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end', 'text', "duration",'break_until_next'])
            prev_end = None
            for entry in transcript:
                start = int(entry['start'] * 1000)  # ミリ秒に変換
                end = int((entry['start'] + entry['duration']) * 1000)  # ミリ秒に変換
                duration = entry['duration'] * 1000
                text = entry['text']
                if prev_end is not None:
                    break_until_next = start - prev_end
                else:
                    break_until_next = 0
                writer.writerow([start, end, text,duration,break_until_next])
                prev_end = end


    else:

        # TODO
    
        vad_fp = Path("./content/vad.csv")
        
        # 無音空間単位を記録する

        VADJob(audio_fp=wav_data_fp, out_fp=vad_fp)

        content = Path("./content")

        vad_df = pd.read_csv(vad_fp)

        for _, row in vad_df.iterrows():

            wav_fp = content / "{}.wav".format(row["id"])
            
            splitter = AudioSplitter(wav_data_fp,row["start_sec"],row["end_sec"],wav_fp)
            splitter.split()

    wav_data = os.path.join("content", yt_id, 'audio.wav')

    transcript = pd.read_csv(os.path.join("content", yt_id, 'transcript.csv'))
    embeddings = generate_speaker_embeddings((wav_data), transcript)

    reference_embeddings = []

    output_by_segment2, reference_embeddings = format_speaker_output_by_segment2(
        wav_data, embeddings, transcript, reference_embeddings, args.prompt_reference
    )


    transcript_fp = os.path.join("content", yt_id, 'transcript.csv')

    output_by_segment2.to_csv(transcript_fp, index=False)

if __name__ == "__main__":
    main()