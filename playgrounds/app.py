import os
import sys
import random
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import shutil
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

# Initialize session state before creating the editor widget.
# Use a different key for storage to avoid modifying the widget key after instantiation.
if "weights_table_data" not in st.session_state:
    st.session_state["weights_table_data"] = df_defaults.copy()

with st.expander("Configuración de pesos de fitness", expanded=False):
    # editable table
    try:
        weights_df = st.data_editor(
            st.session_state["weights_table_data"],
            num_rows="fixed",
            use_container_width=True,
            key="weights_table",
        )
    except Exception:
        # fallback if old streamlit version
        weights_df = st.experimental_data_editor(
            st.session_state["weights_table_data"],
            num_rows="fixed",
            use_container_width=True,
            key="weights_table",
        )

    # Persist the edited table back to session state for the next run
    try:
        st.session_state["weights_table_data"] = weights_df.copy()
    except Exception:
        # If weights_df isn't defined for some reason, keep existing session state value
        pass

    if st.button("Resetear valores por defecto"):
        # Reset the stored table values and force a re-run so the widget is re-instantiated
        st.session_state["weights_table_data"] = df_defaults.copy()
        st.experimental_rerun()

# Validate and extract weights in order
weights = []
# Prefer the editor result, but fall back to the session state stored copy if the widget didn't produce a value
effective_weights_df = None
if 'weights_df' in locals() and isinstance(weights_df, pd.DataFrame):
    effective_weights_df = weights_df
else:
    effective_weights_df = st.session_state.get("weights_table_data", df_defaults)
try:
    weights = [float(w) for w in list(effective_weights_df["Weight"])[:len(df_defaults)]]
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
        date = datetime.now().strftime("%d-%m-%y")
        randomstring = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))
        midi_path = os.path.join(SAVE_PATH, f"melody_{date}_{randomstring}.mid")
        midi_file = melody_to_midi(melody, filename=midi_path)

        # Convertir MIDI a WAV
        wav_path = midi_path.replace('.mid', '.wav')
        FluidSynth().midi_to_audio(midi_path, wav_path)

        score = converter.parse(midi_file)
        base_png = os.path.join(SAVE_PATH, f"melody_{date}_{randomstring}")
        expected_png = f"{base_png}.png"

        # Write the score image to the expected PNG location
        score.write('musicxml.png', fp=expected_png)

        # music21 / Musescore can append "-1" to the filename if it creates an alternative
        # file (e.g. melody_...-1.png). Detect that and rename safely to the expected filename.
        if not os.path.exists(expected_png):
            alt_png = f"{base_png}-1.png"
            if os.path.exists(alt_png):
                # Use shutil.move (or os.replace) instead of os.move which doesn't exist
                shutil.move(alt_png, expected_png)
                new_path = expected_png
            else:
                # Last-resort scan: find a matching PNG in SAVE_PATH that starts with the base name
                matches = [p for p in os.listdir(SAVE_PATH) if p.startswith(os.path.basename(base_png)) and p.endswith('.png')]
                if matches:
                    new_path = os.path.join(SAVE_PATH, matches[0])
                else:
                    # If nothing was created, fallback to the expected filename (it might still get created in some cases)
                    new_path = expected_png
        else:
            new_path = expected_png

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
            # Already set to expected_png (or detected alternative) above
            pass

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