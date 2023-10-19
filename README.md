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
