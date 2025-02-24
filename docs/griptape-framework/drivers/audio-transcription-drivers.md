---
search:
  boost: 2
---

## Overview

[Audio Transcription Drivers](../../reference/griptape/drivers/audio_transcription/index.md) extract text from spoken audio.

## Audio Transcription Drivers

### OpenAI

The [OpenAI Audio Transcription Driver](../../reference/griptape/drivers/audio_transcription/openai_audio_transcription_driver.md) utilizes OpenAI's sophisticated `whisper` model to accurately transcribe spoken audio into text. This model supports multiple languages, ensuring precise transcription across a wide range of dialects.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/audio_transcription_drivers_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/audio_transcription_drivers_1.txt"
    ```

