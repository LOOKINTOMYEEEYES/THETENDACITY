import os
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from scipy.fft import fft, fftfreq

# === STEP 1: Extract pitch using pydub and scipy ===
def extract_pitch(audio_file):
    print("Extracting pitch...")
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file)
    
    # Convert audio to mono and get raw samples
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Sampling rate of the audio
    samplerate = audio.frame_rate
    
    # Define chunk size for FFT
    chunk_size = 2048
    times = []
    pitches = []
    
    # Iterate over the samples in chunks
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start+chunk_size]
        if len(chunk) < chunk_size:
            break
        
        # Perform Fast Fourier Transform (FFT) to get frequency spectrum
        spectrum = np.abs(fft(chunk))
        freqs = fftfreq(len(chunk), 1/samplerate)
        
        # Find the peak frequency (most dominant frequency)
        peak_freq = abs(freqs[np.argmax(spectrum[:len(spectrum)//2])])  # Only consider positive frequencies
        pitches.append(peak_freq)
        times.append(start / samplerate)
    
    return times, pitches, samplerate

# === STEP 2: Convert pitch to MIDI (for later use with SAM) ===
def pitch_to_midi(pitch_track):
    midi_track = []
    for freq in pitch_track:
        if freq > 0:
            midi = int(round(69 + 12 * np.log2(freq / 440.0)))  # Convert frequency to MIDI note
        else:
            midi = 0  # Silence
        midi_track.append(midi)
    return midi_track

# === MAIN FUNCTION ===
def main():
    audio_file = 'your_audio_vocals.wav'  # Replace with your audio path (vocal-only track)
    
    # Step 1: Extract pitch from vocals
    times, pitch_track, sr = extract_pitch(audio_file)

    # Step 2: Convert pitch to MIDI
    midi_track = pitch_to_midi(pitch_track)

    # Output pitch data
    print("\nTime (s)\tMIDI Note")
    for t, midi in zip(times, midi_track):
        if midi > 0:
            print(f"{t:.2f}s\t{midi}")

if __name__ == "__main__":
    main()

