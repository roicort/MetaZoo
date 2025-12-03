import os
import sys
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
from datetime import datetime

# MIDI y MusicXML
from music21 import converter, environment
from midi2audio import FluidSynth

# MetaZoo

from metazoo.bio.evolutionary.operators import selection
from metazoo.bio.utils import Population

# Utils
from viEvolvedi.melody_encoding import MelodyEncoding
from viEvolvedi.utils import melody_to_midi
from viEvolvedi.fitness import ScoreFitness
from viEvolvedi.operators import MelodyOperators
from viEvolvedi.utils import MelodyGA


# Configuración de MusicXML
us = environment.UserSettings()

if sys.platform.startswith("darwin"):
    us['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
    us['musescoreDirectPNGPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
    SAVE_PATH = os.path.join(os.getcwd(), "generated")
if sys.platform.startswith("linux"):
    us['musicxmlPath'] = '/usr/bin/musescore'
    us['musescoreDirectPNGPath'] = '/usr/bin/musescore'
    SAVE_PATH = os.path.join('/tmp', 'generated')

# Save Path
os.makedirs(SAVE_PATH, exist_ok=True)


st.title("ViEVOLVEdi")
st.subheader("Generador Evolutivo de Melodías")

note_range = st.slider("Rango de notas (MIDI)", 21, 108, (60, 72))
col1, col2 = st.columns([1, 1])
with col1:
    length = st.number_input("Longitud de la melodía", min_value=8, max_value=256, value=32)
    population_size = st.number_input("Tamaño de la población", min_value=10, max_value=500, value=100)
    generations = st.number_input("Generaciones", min_value=10, max_value=1000, value=100)
with col2:
    crossover_rate = st.slider("Tasa de cruce", 0.0, 1.0, 0.7, step=0.1)
    mutation_rate = st.slider("Tasa de mutación", 0.0, 1.0, 0.1, step=0.1)
    elitism = st.slider("Elitismo", 0.0, 1.0, 0.1, step=0.1)

default_weights = [
    ("Diversity", 1.0),
    ("Rhythmic Variation", 1.0),
    ("Pitch Range", 1.0),
    ("Temporal Density", 1.0),
    ("Contour Variation", 1.0),
    ("Intervallic Variety", 1.0),
    ("Syncopation Index", 1.0),
    ("Tonal Stability", 1.0),
    ("Motivic Repetition", 0.1),
    ("Phrase Structure", 5.0),
    ("Tension Release", 0.1),
    ("Articulation Variation", 1.0),
    ("Dynamic Range", 0.1),
    ("Silent Beats", 1.0),
    ("Harmonic Relation", 0.1),
    ("Excessive Repetition", 1.0),
    ("Melodic Resolution", 1.0)
]

df_defaults = pd.DataFrame(default_weights, columns=["Metric", "Weight"])

with st.expander("Configuración de pesos de fitness", expanded=False):
    # editable table
    try:
        weights_df = st.data_editor(df_defaults, num_rows="fixed", use_container_width=True, key="weights_table")
    except Exception:
        # fallback if old streamlit version
        weights_df = st.experimental_data_editor(df_defaults, num_rows="fixed", use_container_width=True, key="weights_table")
    if st.button("Resetear valores por defecto"):
        # Reset the table by writing the defaults back via session state key
        st.session_state["weights_table"] = df_defaults.copy()
        weights_df = df_defaults.copy()

# Validate and extract weights in order
weights = []
try:
    weights = [float(w) for w in list(weights_df["Weight"])[:len(df_defaults)]]
except Exception:
    st.error("Los pesos deben ser valores numéricos. Corrige la tabla antes de generar la melodía.")

if len(weights) != len(df_defaults):
    st.error(f"Se esperan {len(df_defaults)} pesos. Tabla tiene {len(weights)}.")


if st.button("Generar melodía"):
    encoding = MelodyEncoding(note_range=note_range, length=length)
    population = Population(population_size=population_size, encoding=encoding)
    # Validate extracted weights again before building fitness function
    if len(weights) != len(df_defaults) or any(map(lambda x: not isinstance(x, (int, float)), weights)):
        st.error("Los pesos no son válidos. Asegúrate de que sean numéricos y que la tabla tenga la cantidad correcta de filas.")
        st.stop()

    fitness = ScoreFitness(
        weights=weights
    )

    ga = MelodyGA(
        fitness_function=fitness.fitness_function,
        selection_function=selection.tournament,
        crossover_function=MelodyOperators.crossover,
        mutation_function=MelodyOperators.mutation,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        encoder=encoding,
        elitism=elitism
    )

    def streamlit_runner(self, generations: int, history: bool = False):
        progress_text = "Evolucionando melodía..."
        my_bar = st.progress(0, text=progress_text)

        pop_history, best_history = [], []
        for step in range(generations):
            self.evolve()
            pop_history.append(self.population.individuals.copy())
            best_history.append(self.best_individual)
            pct = int((step + 1) / generations * 100)
            my_bar.progress(pct, text=progress_text)

        my_bar.empty()
        if history:
            pop_history = [
                [self.population.encoding.decode(ind) for ind in gen]
                for gen in pop_history
            ]
            return best_history, pop_history
        
    best_history, pop_history = streamlit_runner(ga, generations=generations, history=True)

    with st.spinner("Post-processing..."):

        melody = ga.best_individual
        date = datetime.now().strftime("%d-%m-%y, %H.%M")
        midi_path = os.path.join(SAVE_PATH, f"melody_{date}.mid")
        midi_file = melody_to_midi(melody, filename=midi_path)

        # Convertir MIDI a WAV
        wav_path = midi_path.replace('.mid', '.wav')
        FluidSynth().midi_to_audio(midi_path, wav_path)

        score = converter.parse(midi_file)
        png_path = os.path.join(SAVE_PATH, f"melody_{date}-1.png")
        score.write('musicxml.png', fp=png_path)
        new_path = png_path.replace('.png', '-1.png')

        if st.context.theme.type == "dark":
            # Snippet para invertir colores de la partitura 
            # Solo de los canales RGB, manteniendo la transparencia
            img = Image.open(new_path)
            img_rgba = img.convert("RGBA")
            r, g, b, a = img_rgba.split()
            rgb = Image.merge("RGB", (r, g, b))
            inverted_rgb = ImageOps.invert(rgb)
            inv_r, inv_g, inv_b = inverted_rgb.split()
            inverted_image = Image.merge("RGBA", (inv_r, inv_g, inv_b, a))
            inverted_image.save(new_path)
        else:
            new_path = png_path

        st.image(new_path, caption="Partitura generada")

        # Mostrar reproductor de audio en Streamlit
        st.audio(wav_path, format="audio/wav")

        # Leer el MIDI como bytes para Streamlit
        with open(midi_path, "rb") as _midi_f:
            midi_bytes = _midi_f.read()

        st.download_button(
            label="Descargar MIDI",
            data=midi_bytes,
            file_name=os.path.basename(midi_path),
            mime="audio/midi"  # Tipo MIME para archivos MIDI
        )

        st.subheader("Fitness")
        fig = fitness.plot(melody)
        st.plotly_chart(fig)

        st.subheader("Convergencia")

        fig2 = ga.fitness_plot()
        st.plotly_chart(fig2)

        st.balloons()