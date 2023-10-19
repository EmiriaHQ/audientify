import os
import yt_dlp

SAVE_DIR_ROOT = 'content'

class YouTubeDownloader:

    def __init__(self, save_dir_root):
        self.SAVE_DIR_ROOT = save_dir_root
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '16',
            'noplaylist': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '16',
            }],
            'outtmpl': os.path.join(self.SAVE_DIR_ROOT, '%(id)s', 'audio.%(ext)s'),
        }

    def download_audio(self, video_url):
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([video_url])