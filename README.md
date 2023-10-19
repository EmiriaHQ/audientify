# audientify

AI transcribes meetings, performs speaker separation, and automatically creates summaries.

- whisper / whisper.cpp / support
- mp3、wav、mp4,m3ub、youtube video support

## Setup

```bash
# clone on github
git clone https://github.com/emiria-ai/audientify
cd audientify

# clone whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
bash ./models/download-ggml-model.sh large
```

## How to use

Register a 10-second reference audio in the `reference_audio` folder, and a speaker will be automatically assigned. If no file is placed, the speaker is further clustered as an unknown speaker.

If an unknown speaker is found, there is an option to assign a name in the process.
