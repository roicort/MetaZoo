import pretty_midi
from metazoo.bio.evolutionary import GeneticAlgorithm

def melody_to_midi(melody, filename="melody.mid", velocity=100, instrument_name="Acoustic Grand Piano"):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
    start = 0
    for note_number, duration in melody:
        if note_number and note_number > 0:  # Solo agrega nota si es mayor a 0
            note = pretty_midi.Note(velocity=velocity, pitch=int(note_number), start=start, end=start+duration)
            instrument.notes.append(note)
        # Avanza el tiempo aunque sea silencio
        start += duration
    pm.instruments.append(instrument)
    pm.write(filename)
    return filename

class MelodyGA(GeneticAlgorithm):
    def __init__(self, *args, **kwargs):
        kwargs['minimize'] = False
        super().__init__(*args, **kwargs)