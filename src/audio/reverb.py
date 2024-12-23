from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile

def reverb_application(input: str, output: str, parameters: dict) -> None:
    """
    Application of reverb

    Parameters:
    -----------
        input: Path to application input audio
        output: Path to destination
        parameter: dict with parameter setting

    Returns:
    --------
        output: output file path
    """

    board = Pedalboard([Reverb(
                    room_size=parameters.get("selected_size", 0.5),
                    damping=parameters.get("selected_damp", 0.5),
                    wet_level=parameters.get("selected_wet", 0.5),
                    dry_level=1 - parameters.get("selected_wet", 0.5),
                    width=parameters.get("selected_spread", 0.5)
                )])
    
    with AudioFile(input) as f:
        with AudioFile(output, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                audio = f.read(f.samplerate)
                effected = board(audio, f.samplerate, reset=False)
                o.write(effected)