import pyaudio
import numpy as np

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Initialize PyAudio here to handle exceptions during initialization.
try:
    p = pyaudio.PyAudio()
except Exception as e:
    print(f"An error occurred while initializing PyAudio: {e}")
    # Optionally, re-raise the exception if you want to halt the program
    # raise e

def record_audio():
    stream = None
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )
        print("* recording")

        frames = []
        for _ in range(0, int(RATE / FRAMES_PER_BUFFER)):
            data = stream.read(FRAMES_PER_BUFFER)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        print("* done recording")

    except Exception as e:
        print(f"An error occurred during recording: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()

    return np.concatenate(frames)

def terminate():
    try:
        p.terminate()
    except Exception as e:
        print(f"An error occurred while terminating PyAudio: {e}")

# Test the function
if __name__ == "__main__":
    audio_data = record_audio()
    print(audio_data)
    terminate()
