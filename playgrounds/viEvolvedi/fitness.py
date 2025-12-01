import numpy as np
import plotly.graph_objects as go

class ScoreFitness:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = [
                1.0,  # Diversity
                1.0,  # Rhythmic Variation
                1.0,  # Pitch Range
                1.0,  # Temporal Density
                1.0,  # Contour Variation
                1.0,  # Intervallic Variety
                1.0,  # Syncopation Index
                1.0,  # Tonal Stability
                0.1,  # Motivic Repetition
                1.0,  # Phrase Structure
                0.1,  # Tension Release
                1.0,  # Articulation Variation
                0.1,  # Dynamic Range
                1.0,  # Silent Beats
                0.1,  # Harmonic Relation
                1.0,  # Excessive Repetition
                1.0   # Melodic Resolution
            ]
        else:
            self.weights = weights

    def eval(self, melody):
        scores = [
            self.diversity(melody),
            self.rhythmic_variation(melody),
            self.pitch_range(melody),
            self.temporal_density(melody),
            self.contour_variation(melody),
            self.intervallic_variety(melody),
            self.syncopation_index(melody),
            self.tonal_stability(melody),
            self.motivic_repetition(melody),
            self.phrase_structure(melody),
            self.tension_release(melody),
            self.articulation_variation(melody),
            self.dynamic_range(melody),
            self.silent_beats(melody),
            self.harmonic_relation(melody),
            self.excessive_repetition(melody),
            self.melodic_resolution(melody)
        ]
        return scores * np.array(self.weights)

    @staticmethod
    def diversity(melody):
        """
        DIVERSIDAD
        Descripción: Regresa el número de pares únicos (nota, duración) en la melodía normalizada.
        Normaliza dividiendo por la longitud total de la melodía. 
        Rango: [0, 1].
        """
        if len(melody) == 0:
            return 0
        return len(set(tuple(pair) for pair in melody)) / len(melody)

    @staticmethod
    def rhythmic_variation(melody):
        """
        VARIACIÓN RÍTMICA
        Descripción: Regresa la variedad rítmica como la proporción de duraciones únicas sobre el total de duraciones.
        Normaliza dividiendo por la longitud total de la melodía. 
        Rango: [0, 1].
        """
        durations = [float(duration) for note, duration in melody]
        return len(set(durations)) / len(durations) if durations else 0

    @staticmethod
    def pitch_range(melody):
        """
        RANGO DE PITCH
        Descripción: Calcula el rango de pitch (diferencia entre la nota más alta y la más baja).
        Ignora silencios (notas = 0). Regresa 0 si no hay notas.
        Normaliza dividiendo por 127 (rango MIDI completo). 
        Rango: [0, 1].
        """
        pitches = [float(note) for note, duration in melody if note > 0]
        return (max(pitches) - min(pitches)) / 127 if pitches else 0

    @staticmethod
    def temporal_density(melody):
        """
        DENSIDAD TEMPORAL
        Descripción: Calcula la densidad temporal, que es el número de notas por unidad de tiempo.
        Rango: [0, ∞).
        """
        total_duration = sum(float(duration) for note, duration in melody) # Duración total
        return len(melody) / total_duration if total_duration > 0 else 0 # Notas por unidad de tiempo
    
    @staticmethod
    def contour_variation(melody):
        """
        VARIACIÓN DE CONTORNO
        Descripción: Calcula la variación del contorno melódico como el número de cambios de dirección (ascendente, descendente, igual).
        Regresa el número de cambios de dirección únicos.
        Normaliza dividiendo por 2 (máximo número de cambios únicos). 
        Rango: [0, 1].
        """
        contour = []
        for i in range(1, len(melody)):
            if melody[i][0] > melody[i-1][0]:
                contour.append(1)  # Ascendente
            elif melody[i][0] < melody[i-1][0]:
                contour.append(-1)  # Descendente
            else:
                contour.append(0)  # Igual
        return len(set(contour)) / 2 if contour else 0
    
    @staticmethod
    def intervallic_variety(melody):
        """
        VARIEDAD INTERVALICA
        Descripción: Calcula la variedad intervalica como el número de intervalos únicos entre notas consecutivas.
        Ignora silencios (notas = 0).
        Regresa el número de intervalos únicos.
        Normaliza dividiendo por 12 (número de semitonos en una octava).
        Rango: [0, 1].
        """
        intervals = set()
        for i in range(1, len(melody)):
            if melody[i][0] > 0 and melody[i-1][0] > 0:
                interval = abs(melody[i][0] - melody[i-1][0])
                intervals.add(interval)
        return len(intervals) / 12  # Normaliza por una octava
    
    @staticmethod
    def syncopation_index(melody):
        """
        ÍNDICE DE SINCOPACIÓN
        Descripción: Calcula un índice simple de sincopación contando notas en tiempos débiles (duraciones menores a 1.0).
        Normaliza dividiendo por la longitud total de la melodía. 
        Rango: [0, 1].
        """
        syncopation = 0
        for note, duration in melody:
            if note > 0 and duration < 1.0:  # Nota en tiempo débil
                syncopation += 1
        return syncopation / len(melody) if len(melody) > 0 else 0

    @staticmethod
    def tonal_stability(melody, key_center=60):
        """
        ESTABILIDAD TONAL
        Descripción: Calcula la estabilidad tonal contando notas que pertenecen a la tonalidad mayor (tónica, tercera mayor, quinta).
        Normaliza dividiendo por la longitud total de la melodía. 
        Rango: [0, 1].
        """
        stability = 0
        for note, duration in melody:
            if note > 0:
                interval = abs(note - key_center) % 12
                if interval in [0, 4, 7]:  # Notas estables en la tonalidad mayor
                    stability += 1
        return stability / len(melody) if len(melody) > 0 else 0
    
    @staticmethod
    def motivic_repetition(melody, motif_length=4):
        """
        REPETICIÓN MOTÍVICA
        Descripción: Cuenta cuántas veces se repiten motivos de longitud motif_length en la melodía.
        Penaliza si no hay repeticiones.
        """
        motifs = {}
        for i in range(len(melody) - motif_length + 1):
            motif = tuple(map(tuple, melody[i:i+motif_length]))
            if motif in motifs:
                motifs[motif] += 1
            else:
                motifs[motif] = 1
        repetitions = sum(count - 1 for count in motifs.values() if count > 1)
        return repetitions if repetitions > 0 else -100
    
    @staticmethod
    def phrase_structure(melody, phrase_length=16):
        """
        ESTRUCTURA DE FRASE
        Descripción: Cuenta el número de frases únicas en la melodía.
        Una frase es una secuencia de phrase_length notas.
        phrase_length: cuántas notas por frase
        Regresa el número de frases únicas.
        """
        phrases = [tuple(map(tuple, melody[i:i+phrase_length])) for i in range(0, len(melody), phrase_length)]
        unique_phrases = set(phrases)
        return len(unique_phrases)
    
    @staticmethod
    def tension_release(melody):
        """
        TENSIÓN Y RESOLUCIÓN
        Descripción: Calcula la tensión basada en ascensos y descensos en la melodía.
        Ascensos generan tensión, descensos liberan tensión.
        Regresa el valor absoluto de la tensión acumulada.
        Rango: [0, ∞).
        """
        tension = 0
        for i in range(1, len(melody)):
            if melody[i][0] > melody[i-1][0]:
                tension += 1  # Ascenso genera tensión
            elif melody[i][0] < melody[i-1][0]:
                tension -= 1  # Descenso libera tensión
        return abs(tension)
    
    @staticmethod
    def articulation_variation(melody):
        """
        VARIACIÓN DE ARTICULACIÓN
        Descripción: Cuenta el número de tipos de articulación presentes en la melodía (staccato, legato).
        Asume que duraciones <= 0.5 son staccato y > 0.5 son legato.
        Regresa el número de tipos de articulación.
        Rango: [0, 1].
        """
        articulations = set()
        for note, duration in melody:
            if note > 0:
                if duration <= 0.5:
                    articulations.add('staccato')
                else:
                    articulations.add('legato')
        return len(articulations) / 2  # Normaliza por 2 tipos posibles
    
    @staticmethod
    def dynamic_range(melody, base_velocity=100):
        """
        RANGO DINÁMICO
        Descripción: Calcula el rango dinámico basado en la variación de velocidades asignadas a las notas.
        Asume una velocidad base para todas las notas y calcula la variación.
        Regresa la diferencia entre la velocidad máxima y mínima.
        Rango: [0, ∞).
        """
        velocities = []
        for note, duration in melody:
            if note > 0:
                velocities.append(base_velocity)
            else:
                velocities.append(0)
        return max(velocities) - min(velocities)

    @staticmethod
    def silent_beats(melody):
        """
        SILENCIOS
        Descripción: Cuenta el número de silencios (notas con valor 0) en la melodía.
        Regresa el número de silencios.
        Rango: (-∞, 0].
        """
        silent_count = sum(1 for note, duration in melody if note == 0)

        return -silent_count

    @staticmethod
    def harmonic_relation(melody, chords=[(60, 64, 67), (67, 71, 74), (69, 72, 76), (65, 69, 72)], phrase_length=4):
        """
        RELACIÓN ARMÓNICA
        Descripción: Premia notas que coinciden con las notas de los acordes dados.
        chords: lista de acordes como tuplas de notas MIDI.
        phrase_length: número de notas por frase para asignar acordes.
        Regresa el número de notas que coinciden con las notas de los acordes.
        Rango: [0, ∞).
        """
        score = 0
        for i, (note, duration) in enumerate(melody):
            if note == 0:
                continue
            chord = chords[(i // phrase_length) % len(chords)]
            if note % 12 in [n % 12 for n in chord]:
                score += 1
        return score

    @staticmethod
    def excessive_repetition(melody, max_repeats=3):
        """
        REPETICIÓN EXCESIVA
        Descripción: Penaliza si una nota se repite más de max_repeats veces consecutivamente.
        Regresa una penalización negativa proporcional al exceso de repeticiones.
        Rango: (-∞, 0].
        """
        penalty = 0
        count = 1
        last_note = None
        for note, duration in melody:
            if note == last_note and note != 0:
                count += 1
                if count > max_repeats:
                    penalty += 1
            else:
                count = 1
            last_note = note
        return -penalty

    @staticmethod
    def melodic_resolution(melody, tonic=60):
        """
        RESOLUCIÓN MELÓDICA
        Descripción: Premia si la última nota no silenciosa es la tónica.
        Regresa 1 si la última nota no silenciosa es la tónica, de lo contrario 0.
        Rango: [0, 1].
        """
        for note, duration in reversed(melody):
            if note > 0:
                return 1 if note % 12 == tonic % 12 else 0
        return 0

    def fitness_function(self, melody):
        """
        Función de fitness total que suma todas las sub-métricas ponderadas.
        """
        return self.eval(melody).sum()

    def plot(self, melody):
        scores = self.eval(melody)
        aspects = [
            ("Diversity", scores[0]),
            ("Rhythmic Variation", scores[1]),
            ("Pitch Range", scores[2]),
            ("Temporal Density", scores[3]),
            ("Contour Variation", scores[4]),
            ("Intervallic Variety", scores[5]),
            ("Syncopation Index", scores[6]),
            ("Tonal Stability", scores[7]),
            ("Motivic Repetition", scores[8]),
            ("Phrase Structure", scores[9]),
            ("Tension Release", scores[10]),
            ("Articulation Variation", scores[11]),
            ("Dynamic Range", scores[12]),
            ("Silent Beats", scores[13]),
            ("Harmonic Relation", scores[14]),
            ("Excessive Repetition", scores[15]),
            ("Melodic Resolution", scores[16])
        ]
        labels = [a[0] for a in aspects]
        values = [a[1] for a in aspects]
        labels += [labels[0]]
        values += [values[0]]
        fig = go.Figure(
            data=[go.Scatterpolar(r=values, theta=labels, fill='toself', name='Aspectos')]
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False,
            title="Perfil de Fitness de la Melodía"
        )
        return fig