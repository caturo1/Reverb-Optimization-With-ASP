from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile

def reverb_application(input: str, output: str, parameters: dict) -> AudioFile:
    board = Pedalboard([Reverb(
                    room_size=parameters.get("size", 0.5),
                    damping=parameters.get("damping", 0.5),
                    wet_level=parameters.get("wet", 0.5),
                    dry_level=1 - parameters.get("wet", 0.5),
                    width=parameters.get("spread", 0.5)
                )])
    
    with AudioFile(input) as f:
        with AudioFile(output, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                audio = f.read(f.samplerate)
                effected = board(audio, f.samplerate, reset=False)
                o.write(effected)