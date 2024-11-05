from pedalboard.io import AudioFile
from pedalboard import Reverb

chunk_size = 500_000
samplerate = 44100
input_file_name = "./data/katstrings.wav"
with AudioFile(input_file_name).resampled_to(samplerate) as f:
    with AudioFile("./data/processed_file.wav", "w", f.samplerate, f.num_channels) as o:
        while f.tell() < f.frames:
            audio = f.read(chunk_size)
            reverb = Reverb(room_size=0.9)
            effect = reverb(audio, f.samplerate)
            o.write(effect)
            #plt.figure(figsize=(14, 5))
            #librosa.display.waveshow(effect, sr=samplerate)