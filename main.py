import soundfile as sf
import librosa
from pathlib import Path 
from audio_segment import VADJob
import os
import pandas as pd
import numpy as np
from notion import get_or_create_today_page_id,patch_transcript
import subprocess
import sys
from youtube import YouTubeDownloader
import yt_dlp

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python main.py <audio_file_path or youtube url> <transcript_file_path>")
        sys.exit(1)

    input_arg = sys.argv[1]
    transcript_file_path_arg = sys.argv[2]

    if not os.path.exists("content"):
        os.makedirs("content")

    # input_argがYouTubeのリンクだったら動画を16bitのwavとしてダウンロードし、そうでなければ、単にパスを指定する

    if YouTubeDownloader.is_youtube_url(input_arg):
        downloader = YouTubeDownloader("content")
        downloader.download_audio(input_arg)
        yt_id = yt_dlp.extractor.youtube.YoutubeIE.extract_id(input_arg)
        meeting_file_path = Path(os.path.join("content", yt_id, "audio.wav"))
    else:
        meeting_file_path = Path(input_arg)
    
    output_meeting_file_path = Path("./content/vad.csv")
    
    transcript_fp = Path(transcript_file_path_arg)

    # 無音空間単位で音声ファイルを分割する

    VADJob(audio_fp=meeting_file_path, out_fp=output_meeting_file_path)

    vad_df = pd.read_csv(output_meeting_file_path)

    content = Path("./content")

    # whisper.cppのためモノラル化・16bit変換

    y, sr = librosa.load(meeting_file_path, sr=None, mono=False)
    
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    for _, row in vad_df.iterrows():

        wav_fp = content / "{}.wav".format(row["id"])
        
        start_sample = int(row["start_sec"] * sr)
        end_sample = int(row["end_sec"] * sr)
        y_trimmed = y[start_sample:end_sample]

        sf.write(wav_fp, y_trimmed, sr)

        if not transcript_fp.exists():
            whisper_dir = Path("./whisper.cpp")
            bin_fp = whisper_dir / "main"
            model_fp = whisper_dir / f"models/ggml-large.bin"

            subprocess.run([str(bin_fp), "--model", str(model_fp), "--language", "ja", "-f", str(wav_fp), "--output-csv"])

            fp = Path(output_meeting_file_path).parent / "{}.wav.csv".format(row["id"])
            dfs = []
            df = pd.read_csv(fp)
            df["start"] += row["start_sec"] * 1000
            df["end"] += row["start_sec"] * 1000

            dfs.append(df)
            out_df = pd.concat(dfs, axis=0).reset_index(drop=True)
            out_df.to_csv(transcript_fp, index=False)

        page_id = get_or_create_today_page_id()
        response = patch_transcript(page_id, "./content/transcript.csv")