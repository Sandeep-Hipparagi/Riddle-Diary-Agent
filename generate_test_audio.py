import wave
import math
import struct

def create_synthetic_wav(path="test_audio.wav", duration_sec=1.5, sample_rate=16000):
    """
    Creates a clean 16kHz mono 16-bit PCM WAV file with a gentle sine tone
    mimicking audio recording silence/tone for testing pipeline nodes.
    """
    num_samples = int(duration_sec * sample_rate)
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            # 440 Hz gentle tone
            value = int(1000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            wav_file.writeframes(struct.pack('<h', value))
    print(f"Created synthetic test WAV: {path}")

if __name__ == "__main__":
    create_synthetic_wav()
