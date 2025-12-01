import os
import streamlit as st
from datetime import datetime

# MIDI y MusicXML
from music21 import converter, environment

# MetaZoo
from metazoo.bio.evolutionary import GeneticAlgorithm
from metazoo.bio.evolutionary.operators import selection
from metazoo.bio.utils import Population

# Utils
from viEvolvedi.melody_encoding import MelodyEncoding
from viEvolvedi.utils import melody_to_midi
from viEvolvedi.fitness import ScoreFitness
from viEvolvedi.operators import MelodyOperators

# Configuración de MusicXML
us = environment.UserSettings()
us['musicxmlPath'] = '/usr/bin/musescore'
us['musescoreDirectPNGPath'] = '/usr/bin/musescore'
SAVE_PATH = os.path.join(os.getcwd(), "generated")
os.makedirs(SAVE_PATH, exist_ok=True)

st.title("ViEVOLVEdi - Generador Evolutivo de Melodías")

note_range = st.slider("Rango de notas (MIDI)", 21, 108, (60, 72))
length = st.number_input("Longitud de la melodía", min_value=8, max_value=256, value=16)
population_size = st.number_input("Tamaño de la población", min_value=10, max_value=500, value=100)
generations = st.number_input("Generaciones", min_value=10, max_value=1000, value=100)
crossover_rate = st.slider("Tasa de cruce", 0.0, 1.0, 0.7)
mutation_rate = st.slider("Tasa de mutación", 0.0, 1.0, 0.1)
elitism = st.slider("Elitismo", 0.0, 1.0, 0.1)

if st.button("Generar melodía"):
    encoding = MelodyEncoding(note_range=note_range, length=length)
    population = Population(population_size=population_size, encoding=encoding)
    fitness = ScoreFitness()

    class MelodyGA(GeneticAlgorithm):
        def __init__(self, *args, **kwargs):
            kwargs['minimize'] = False
            super().__init__(*args, **kwargs)

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

    with st.spinner("Evolucionando melodía..."):
        ga.run(generations=int(generations))

    melody = ga.best_individual
    date = datetime.now().strftime("%d-%m-%y, %H.%M")
    midi_path = os.path.join(SAVE_PATH, f"melody_{date}.mid")
    midi_file = melody_to_midi(melody, filename=midi_path)
    st.success(f"MIDI guardado en: {midi_path}")

    score = converter.parse(midi_file)
    png_path = os.path.join(SAVE_PATH, f"melody_{date}.png")
    score.write('musicxml.png', fp=png_path)
    st.image(png_path, caption="Partitura generada")

    st.subheader("Fitness")
    fig = fitness.plot(melody)
    st.plotly_chart(fig)