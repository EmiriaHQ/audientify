import wave

def is_mono(wav_file_path):
    with wave.open(wav_file_path, 'rb') as wf:
        channels = wf.getnchannels()
        return channels == 1

file_path = './audio.wav'
if is_mono(file_path):
    print(f"{file_path} is mono.")
else:
    print(f"{file_path} is not")
