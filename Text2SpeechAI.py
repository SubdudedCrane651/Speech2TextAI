import sys
import requests
import json
import wave
import time
import re
import subprocess
import os

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
    QComboBox,
    QLabel
)

from PyQt6.QtCore import QThread, pyqtSignal,QTimer


# ============================================================
# LOAD API KEY
# ============================================================

with open("config.json", "r") as f:
    CONFIG = json.load(f)

OPENROUTER_API_KEY = CONFIG.get("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY was not found in config.json"
    )


# ============================================================
# SETTINGS
# ============================================================

API_URL = "https://openrouter.ai/api/v1/audio/speech"

MODEL = "deepgram/flux-tts:free"

# Start conservatively.
# If 413 occurs, the program automatically reduces it.
START_CHUNK_SIZE = 1500

# Never allow automatic retry chunks to become smaller than this.
MIN_CHUNK_SIZE = 300

# Maximum number of retries for one chunk.
MAX_RETRIES = 5

# Seconds between successful requests.
REQUEST_DELAY = 0.5

# Flux TTS PCM settings.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2


# ============================================================
# TEXT SPLITTER
# ============================================================

def split_text(text, max_chars):
    """
    Split text into reasonably natural chunks.

    Priority:
        1. Paragraphs
        2. Sentences
        3. Words

    This prevents words from being cut in half.
    """

    text = text.strip()

    if not text:
        return []

    # Normalize excessive whitespace while preserving paragraphs.
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph
        ).strip()

        if not paragraph:
            continue

        # ----------------------------------------------------
        # Paragraph fits
        # ----------------------------------------------------

        if len(paragraph) <= max_chars:

            if not current:

                current = paragraph

            elif len(current) + 1 + len(paragraph) <= max_chars:

                current += " " + paragraph

            else:

                chunks.append(current)
                current = paragraph

            continue

        # ----------------------------------------------------
        # Paragraph too large.
        # Split into sentences.
        # ----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # ------------------------------------------------
            # Sentence fits
            # ------------------------------------------------

            if len(sentence) <= max_chars:

                if not current:

                    current = sentence

                elif len(current) + 1 + len(sentence) <= max_chars:

                    current += " " + sentence

                else:

                    chunks.append(current)
                    current = sentence

            # ------------------------------------------------
            # Sentence is too large.
            # Split by words.
            # ------------------------------------------------

            else:

                words = sentence.split()

                for word in words:

                    if not current:

                        current = word

                    elif len(current) + 1 + len(word) <= max_chars:

                        current += " " + word

                    else:

                        chunks.append(current)
                        current = word

    # --------------------------------------------------------
    # Last chunk
    # --------------------------------------------------------

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# TTS THREAD
# ============================================================

class TTSThread(QThread):

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, text, voice):

        super().__init__()

        self.text = text
        self.voice = voice


    # ========================================================
    # SEND ONE REQUEST
    # ========================================================

    def request_audio(self, text, chunk_number):

        chunk_size = START_CHUNK_SIZE

        retry = 0

        while retry < MAX_RETRIES:

            self.progress.emit(
                f"Generating part {chunk_number}..."
            )

            response = None

            try:

                response = requests.post(

                    url=API_URL,

                    headers={
                        "Authorization":
                            f"Bearer {OPENROUTER_API_KEY}",

                        "Content-Type":
                            "application/json",

                        "Accept":
                            "audio/pcm"
                    },

                    json={
                        "model": MODEL,
                        "input": text,
                        "voice": self.voice
                    },

                    timeout=180
                )

            except requests.exceptions.Timeout:

                retry += 1

                self.progress.emit(
                    f"Timeout on part {chunk_number}. "
                    f"Retry {retry}/{MAX_RETRIES}..."
                )

                time.sleep(2)

                continue

            except requests.exceptions.RequestException as e:

                retry += 1

                if retry >= MAX_RETRIES:

                    raise RuntimeError(
                        f"Network error on part "
                        f"{chunk_number}:\n{e}"
                    )

                self.progress.emit(
                    f"Network error on part "
                    f"{chunk_number}. "
                    f"Retry {retry}/{MAX_RETRIES}..."
                )

                time.sleep(2)

                continue

            # =================================================
            # 413 PAYLOAD TOO LARGE
            # =================================================

            if response.status_code == 413:

                old_size = chunk_size

                chunk_size = max(
                    MIN_CHUNK_SIZE,
                    chunk_size // 2
                )

                if chunk_size == old_size:

                    raise RuntimeError(
                        f"Part {chunk_number} is still "
                        f"too large even at "
                        f"{chunk_size} characters.\n\n"
                        f"Provider response:\n"
                        f"{response.text}"
                    )

                self.progress.emit(
                    f"413 received on part "
                    f"{chunk_number}.\n"
                    f"Reducing chunk size from "
                    f"{old_size} to "
                    f"{chunk_size} characters..."
                )

                # ------------------------------------------------
                # Re-split this text into smaller pieces.
                # Return a special result so run() can handle it.
                # ------------------------------------------------

                smaller_chunks = split_text(
                    text,
                    chunk_size
                )

                return {
                    "split": smaller_chunks
                }

            # =================================================
            # RATE LIMIT
            # =================================================

            if response.status_code == 429:

                retry += 1

                wait_time = min(
                    10,
                    2 ** retry
                )

                self.progress.emit(
                    f"Rate limited on part "
                    f"{chunk_number}. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            # =================================================
            # TEMPORARY SERVER ERRORS
            # =================================================

            if response.status_code in (
                500,
                502,
                503,
                504
            ):

                retry += 1

                if retry >= MAX_RETRIES:

                    raise RuntimeError(
                        f"Server error on part "
                        f"{chunk_number}.\n\n"
                        f"HTTP {response.status_code}\n\n"
                        f"{response.text}"
                    )

                self.progress.emit(
                    f"Server error "
                    f"{response.status_code} on part "
                    f"{chunk_number}. "
                    f"Retrying..."
                )

                time.sleep(2)

                continue

            # =================================================
            # OTHER ERRORS
            # =================================================

            if response.status_code != 200:

                raise RuntimeError(
                    f"Part {chunk_number} failed.\n\n"
                    f"HTTP {response.status_code}\n\n"
                    f"{response.text}"
                )

            # =================================================
            # SUCCESS
            # =================================================

            content_type = response.headers.get(
                "Content-Type",
                ""
            ).lower()

            # -------------------------------------------------
            # Make sure we received audio.
            # -------------------------------------------------

            if not content_type.startswith("audio/"):

                # Sometimes providers may not return the
                # expected content type. Check for JSON anyway.
                try:

                    error_data = response.json()

                    raise RuntimeError(
                        "The server returned JSON instead "
                        "of audio:\n\n"
                        + json.dumps(
                            error_data,
                            indent=2
                        )
                    )

                except ValueError:

                    raise RuntimeError(
                        "The server returned an unexpected "
                        f"content type:\n{content_type}\n\n"
                        f"Response:\n{response.text[:1000]}"
                    )

            # -------------------------------------------------
            # Make sure audio isn't empty.
            # -------------------------------------------------

            if not response.content:

                raise RuntimeError(
                    f"Part {chunk_number} returned "
                    f"empty audio."
                )

            return {
                "audio": response.content
            }

        raise RuntimeError(
            f"Part {chunk_number} failed after "
            f"{MAX_RETRIES} attempts."
        )


    # ========================================================
    # THREAD RUN
    # ========================================================

    def run(self):

        try:

            # ------------------------------------------------
            # Initial split
            # ------------------------------------------------

            chunks = split_text(
                self.text,
                START_CHUNK_SIZE
            )

            if not chunks:

                raise RuntimeError(
                    "No text was found."
                )

            self.progress.emit(
                f"Text split into "
                f"{len(chunks)} parts."
            )

            # ------------------------------------------------
            # Audio storage
            # ------------------------------------------------

            all_pcm_data = bytearray()

            # ------------------------------------------------
            # Process chunks
            # ------------------------------------------------

            index = 0

            while index < len(chunks):
                
                chunk = chunks[index]

                self.progress.emit(
                    f"Processing part "
                    f"{index + 1} of "
                    f"{len(chunks)} "
                    f"({len(chunk)} characters)..."
                )

                result = self.request_audio(
                    chunk,
                    index + 1
                )

                # ------------------------------------------------
                # 413 caused this chunk to be split smaller.
                # ------------------------------------------------

                if "split" in result:

                    smaller_chunks = result["split"]

                    self.progress.emit(
                        f"Part {index + 1} was too large. "
                        f"Splitting into "
                        f"{len(smaller_chunks)} smaller parts..."
                    )

                    # Replace the large chunk with smaller chunks.
                    chunks[index:index + 1] = smaller_chunks

                    # Retry same index.
                    continue

                # ------------------------------------------------
                # Successful audio
                # ------------------------------------------------

                all_pcm_data.extend(
                    result["audio"]
                )

                index += 1

                # ------------------------------------------------
                # Small delay between requests.
                # ------------------------------------------------

                if index < len(chunks):

                    time.sleep(
                        REQUEST_DELAY
                    )

            # ------------------------------------------------
            # Make sure we actually received audio.
            # ------------------------------------------------

            if not all_pcm_data:

                raise RuntimeError(
                    "No audio was generated."
                )

            # ------------------------------------------------
            # Save final WAV
            # ------------------------------------------------

            output_wav = "tts_output.wav"

            self.progress.emit(
                "All parts generated. Combining audio..."
            )

            with wave.open(output_wav, "wb") as wav_file:

                wav_file.setnchannels(NUM_CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(all_pcm_data)


            # ------------------------------------------------
            # Convert WAV to MP3
            # ------------------------------------------------

            output_mp3 = "tts_output.mp3"

            self.progress.emit(
                "WAV created. Converting to MP3..."
            )

            try:

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        output_wav,
                        "-codec:a",
                        "libmp3lame",
                        "-b:a",
                        "128k",
                        output_mp3
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            except FileNotFoundError:

                raise RuntimeError(
                    "FFmpeg was not found.\n\n"
                    "Please install FFmpeg and make sure "
                    "it is available in your Windows PATH."
                )

            except subprocess.CalledProcessError as e:

                raise RuntimeError(
                    "FFmpeg failed to convert the WAV to MP3.\n\n"
                    + e.stderr
                )


            # ------------------------------------------------
            # Finished
            # ------------------------------------------------

            self.finished.emit(output_mp3)

        except Exception as e:

            self.error.emit(
                str(e)
            )


# ============================================================
# GUI
# ============================================================

class TextToSpeechGUI(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Text → Speech (Flux-TTS Free)"
        )

        self.setMinimumWidth(700)

        layout = QVBoxLayout()

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        self.info = QLabel(
            "Enter text and choose a voice."
        )

        self.info.setWordWrap(True)

        layout.addWidget(
            self.info
        )

        # ----------------------------------------------------
        # Text box
        # ----------------------------------------------------

        self.textbox = QTextEdit()

        self.textbox.setPlaceholderText(
            "Type English text here..."
        )

        layout.addWidget(
            self.textbox
        )

        # ----------------------------------------------------
        # Voices
        # ----------------------------------------------------

        voices = [

            "flux-alexis-en",
            "flux-bree-en",
            "flux-brittany-en",
            "flux-brooke-en",
            "flux-bruce-en",
            "flux-cliff-en",
            "flux-cole-en",
            "flux-colin-en",
            "flux-conor-en",
            "flux-donovan-en",
            "flux-drew-en",
            "flux-elise-en",
            "flux-gemma-en",
            "flux-haley-en",
            "flux-hannah-en",
            "flux-heather-en",
            "flux-jack-en",
            "flux-kai-en",
            "flux-kelsey-en",
            "flux-kit-en",
            "flux-maeve-en",
            "flux-marcelo-en",
            "flux-marcus-en",
            "flux-meena-en",
            "flux-meghan-en",
            "flux-miles-en",
            "flux-naveen-en",
            "flux-paige-en",
            "flux-priya-en",
            "flux-rufus-en",
            "flux-sean-en",
            "flux-sharon-en",
            "flux-sienna-en",
            "flux-tanner-en",
            "flux-wade-en",
            "flux-wes-en"
        ]

        self.voice_selector = QComboBox()

        self.voice_selector.addItems(
            voices
        )

        layout.addWidget(
            self.voice_selector
        )
  
        self.timer_label = QLabel("Elapsed: 00:00")
        layout.addWidget(self.timer_label)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.elapsed_seconds = 0

        # ----------------------------------------------------
        # Generate button
        # ----------------------------------------------------

        self.btn = QPushButton(
            "Generate Speech"
        )

        self.btn.clicked.connect(
            self.generate_speech
        )

        layout.addWidget(
            self.btn
        )

        self.setLayout(
            layout
        )

        self.thread = None

    # ========================================================
    # START GENERATION
    # ========================================================
    
    def update_timer(self):
                self.elapsed_seconds += 1
                minutes = self.elapsed_seconds // 60
                seconds = self.elapsed_seconds % 60
                self.timer_label.setText(f"Elapsed: {minutes:02d}:{seconds:02d}")
        

    def generate_speech(self):

        text = self.textbox.toPlainText().strip()

        if not text:

            self.info.setText(
                "Please enter text first."
            )

            return

        voice = (
            self.voice_selector.currentText()
        )

        self.info.setText(
            "Starting speech generation..."
        )
     
        self.btn.setEnabled(
            False
        )

        self.thread = TTSThread(
            text,
            voice
        )

        self.thread.finished.connect(
            self.show_result
        )

        self.thread.error.connect(
            self.show_error
        )

        self.thread.progress.connect(
            self.show_progress
        )
        
        self.elapsed_seconds = 0
        self.timer_label.setText("Elapsed: 00:00")
        self.timer.start(1000)

        self.thread.start()


    # ========================================================
    # PROGRESS
    # ========================================================

    def show_progress(self, message):

        self.info.setText(
            message
        )
        
        


    # ========================================================
    # SUCCESS
    # ========================================================

    def show_result(self, path):

        self.info.setText(
            f"Speech generated successfully: {path}"
        )

        self.textbox.append(
            f"\n\nSaved as: {path}"
        )

        self.btn.setEnabled(
            True
        )
        
        self.timer.stop()

    # ========================================================
    # ERROR
    # ========================================================

    def show_error(self, err):

        self.info.setText(
            "Error occurred."
        )

        self.textbox.append(
            "\n\nERROR:\n" + err
        )

        self.btn.setEnabled(
            True
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = TextToSpeechGUI()

    window.show()

    sys.exit(
        app.exec()
    )
