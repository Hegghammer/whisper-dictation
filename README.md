# whisper-dictation

Simple script to use Whisper for dictation. Inspired by [wkey](https://github.com/vlad-ds/whisper-keyboard), but with support for offline models and punctuation commands. It works with any backend that supports the OpenAI SDK, such as [Speaches](https://github.com/speaches-ai/speaches/).

## Setup

1. Set up the backend (beyond the scope of this README). 

2. Store the backend's URL and API key as the environment variables `DICTATION_BASE_URL` and `DICTATION_API_KEY`

3. Install required python packages

```bash
pip install sounddevice numpy pynput scipy openai
```

4. Download the script

```bash
git clone https://github.com/Hegghammer/whisper-dictation.git
```

## Usage

1. Run the script

```bash
python dict.py $WHISPER_MODEL
```

If you are using Speaches, $WHISPER_MODEL can be one of the aliases you set in `model_aliases.json`.

2. Move the cursor to the app where you want the output.

3. Hold down right `CTRL`, speak, and release.  

## Punctuation commands

The script implements some basic punctuation commands via regex. For example, you can say "new line" and the model will insert a linebreak instead of typing "new line".

- "new line" (or "newline") -> "\n" 
- "actual new line"         -> "new line"
- "inverted comma"          -> '"'
- "actual inverted comma"   -> "inverted comma"
- "full stop"               -> "."
- "actual full stop"        -> "full stop" 
- "comma"                   -> ","
- "actual comma"            -> "comma"

This currently works only for English, but you can modify the script for similar functionality in other languages.

## Custom spelling

Whisper models take a prompt that can guide spelling. The script implements this through the `CUSTOM SPELLING` variable at the top. It currently contains a few example words and some sentences to inspire UK spelling. Tweak it to your needs.

Note: The "distil" series of Whisper models does not seem to heed custom spelling instructions. 
