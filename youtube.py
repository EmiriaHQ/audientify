import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Tuple

class YouTubeDownloader:

    def __init__(self, save_dir_root):
        self.SAVE_DIR_ROOT = save_dir_root
        self.ydl_opts = {
                    'format': 'bestaudio/best',
                    'extractaudio': True,  # 音声のみをダウンロード
                    'audioformat': 'wav',  # wav形式で保存
                    'audioquality': '16',  # 16bit音質
                    'noplaylist': False,   # プレイリストをダウンロード
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '16',
                    }],
                    'postprocessor_args': [
                        '-ar', '16000',
                        '-ac', '1'
                    ],
            'outtmpl': os.path.join(self.SAVE_DIR_ROOT, '%(id)s', 'audio.%(ext)s'),
        }
    
    def download_audio(self, video_url):
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([video_url])

    @staticmethod
    def is_youtube_url(url):
        """指定されたURLがYouTubeのURLかどうかを判断する"""
        if "youtube.com" in url or "youtu.be" in url:
            return True
        return False