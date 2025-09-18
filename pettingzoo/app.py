import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from metazoo.bio.evolutionary import GeneticAlgorithm
from metazoo.bio.evolutionary.operators import selection, mutation, crossover
from metazoo.gym.mono import Function
from metazoo.gym.combinatorial import TSP, NQueens
from metazoo.bio.evolutionary.utils import encoding

from plots import parametric_population_history, combinatorial_population_history

available_algorithms = ["Genetic Algorithm"]
colorscales = px.colors.named_colorscales()

st.set_page_config(page_title="Petting Zoo", layout="wide")
st.title("Petting Zoo")

problem_options = None
problem = None
algorithm = None
problem_type = None
encoding_type = None
precision = None
crossover_options = []
mutation_options = []

with st.sidebar:
    st.title("Petting Zoo")
    st.header("Configuration")
    with st.expander("About", expanded=False):
        st.markdown(
            """
            **Petting Zoo** is an interactive playground to experiment with optimization & evolutionary algorithms on analytic test functions.
            It is powered by the **MetaZoo** library (core evolutionary / gym components).
            Repo: [MetaZoo on GitHub](https://github.com/roicort/MetaZoo)
            Select a function, tune GA parameters, run the algorithm and visualize population dynamics and best candidate.
            """
        )
    algorithm = st.selectbox("Algorithm", available_algorithms)
    problem_type = st.selectbox(
        "Type",
        ["Parametric", "Combinatorial"],
        index=0,
        help="Parametric: continuous or discrete variables; Combinatorial: permutations or combinations.",
    )

    if problem_type == "Combinatorial":
        problem_options = st.selectbox("Problem", ["TSP", "NQueens"], index=0)
        if problem_options == "TSP":
            problem = st.selectbox("Problem", ["Berlin52"], index=0)
            minimize = True  
        if problem_options == "NQueens":
            problem = st.select_slider("Problem", options=[4, 8, 12, 16, 32, 64], value=8)
            minimize = False # We maximize the number of non-attacking pairs

        if algorithm == "Genetic Algorithm":
            encoding_type = st.selectbox("Encoding", ["permutation"], index=0)
            crossover_options = ["pmx"]
            mutation_options = ["swap"]

    if problem_type == "Parametric":
        problem = st.selectbox("Function", Function().available_functions)
        reverse = st.checkbox(
            "Reverse function?",
            value=False,
            help="Flip sign (f(x) -> -f(x)). [This affects both visualization and function evaluation.]",
        )

        minimize = st.checkbox(
            "Minimize?",
            value=True,
            help="If checked the algorithm searches minimum; otherwise it maximizes. Does not change the plotted surface.",
        )

        if algorithm == "Genetic Algorithm":
            encoding_type = st.selectbox("Encoding", ["binary", "real"], index=0)
            colorscale = st.selectbox("Color Scale", colorscales)
            crossover_options = ["onepoint"]
            if encoding_type == "binary":
                mutation_options = ["bitflip"]
                precision = st.selectbox(
                    "Precision (bits)",
                    options=[2, 4, 8, 16],
                    help="Number of bits per variable for binary encoding.",
                )
            if encoding_type == "real":
                mutation_options = ["gaussian"]

st.subheader(f"Problem: {problem} ({problem_type})")

if problem_type == "Parametric":
    func_obj = Function(problem, reverse=reverse)
    if encoding_type == "binary":
        encoder = encoding.Binary(
            bounds=func_obj.bounds, precision=precision
        )
    if encoding_type == "real":
        encoder = encoding.Real(bounds=func_obj.bounds)

if problem_type == "Combinatorial":
    if problem_options == "TSP":
        if problem == "Berlin52":
            func_obj = TSP.Berlin52()
        encoder = encoding.Permutation(permutation_size=func_obj.dimension)
    if problem_options == "NQueens":
        func_obj = NQueens(n=problem)
        encoder = encoding.Permutation(permutation_size=func_obj.n)

if problem_type == "Parametric" and len(func_obj.bounds) == 2:
    st.latex(f"{func_obj.formula()}")
    fig = func_obj.plot(func_obj.bounds, dim=2, num_points=100, population=None, mode='surface', colorscale='Viridis')
    fig.update_layout(title="Function Landscape with Best Individual", autosize=True)
    st.plotly_chart(fig, use_container_width=True)

if problem_type == "Combinatorial":
    if problem_options == "NQueens":
        st.latex(f"Objective: Minimize number of attacking pairs of queens.")
        fig = func_obj.plot(solution=None, attacks=None)
        fig.update_layout(title="N-Queens Board with Best Individual", autosize=True)
        st.plotly_chart(fig, use_container_width=True)
    if problem_options == "TSP":
        st.latex(f"Objective: Minimize total travel distance.")
        fig = func_obj.plot(solution=None, show_optimal=True)
        fig.update_layout(title="TSP Route with Best Individual", autosize=True)
        st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Parameters of {algorithm}")

if algorithm == "Genetic Algorithm":
    st.caption(
        "Configure GA parameters: precision controls binary resolution; elitism preserves top individuals."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        population_size = st.number_input(
            "Population Size", min_value=10, max_value=50000, value=200, step=10
        )
        generations = st.number_input(
            "Generations", min_value=1, max_value=20000, value=200, step=10
        )
    with col2:

        mutation_type = st.selectbox("Mutation Function", options=mutation_options)
        mutation_rate = st.slider(
            "Mutation Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )

    with col3:

        crossover_type = st.selectbox(
            "Crossover Function",
            options=crossover_options,
            index=0,
            help="Crossover method to combine parents.",
        )

        crossover_rate = st.slider(
            "Crossover Rate", min_value=0.0, max_value=1.0, value=0.8, step=0.01
        )

    with col4:

        selection_function = st.selectbox(
            "Selection Function",
            options=[
                "tournament",
                "roulette",
                "rank",
                "uniform",
            ],
            index=0,
            help="Method to select parents for reproduction.",
        )

        elitism_option = st.selectbox(
            "Elitism",
            options=["None", 0.1, 0.3, 1.0],
            index=0,
            help="Fraction of population preserved. 1.0 = full population (no evolutionary change).",
        )
        elitism_fraction = None if elitism_option == "None" else float(elitism_option)

run = st.button(f"Run {algorithm}")

if run and algorithm == "Genetic Algorithm":

    if problem_type == "Parametric":
        if encoding_type == "binary":
            encoder = encoding.Binary(
                bounds=func_obj.bounds, precision=precision
            )
        if encoding_type == "real":
            encoder = encoding.Real(bounds=func_obj.bounds)
    
    if problem_type == "Combinatorial":
        if problem_options == "TSP":
            encoder = encoding.Permutation(permutation_size=func_obj.dimension)
        if problem_options == "NQueens":
            encoder = encoding.Permutation(permutation_size=func_obj.n)

    if mutation_type == "bitflip":
        mutation_function = mutation.flip_bit
    if mutation_type == "gaussian":
        mutation_function = mutation.gaussian
    if mutation_type == "swap":
        mutation_function = mutation.swap

    if crossover_type == "onepoint":
        crossover_function = crossover.onepoint
    if crossover_type == "pmx":
        crossover_function = crossover.PMX

    if selection_function == "tournament":
        selection_function = selection.tournament
    if selection_function == "roulette":
        selection_function = selection.roulette
    if selection_function == "rank":
        selection_function = selection.rank
    if selection_function == "uniform":
        selection_function = selection.uniform

    ga = GeneticAlgorithm(
        fitness_function=func_obj,
        crossover_function=crossover_function,
        mutation_function=mutation_function,
        selection_function=selection_function,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        encoder=encoder,
        minimize=minimize,
        elitism=elitism_fraction,
    )
    st.session_state["ga"] = ga
    st.session_state["func"] = func_obj
    st.caption(
        f"GA => pop:{population_size} gens:{generations} mut:{mutation_rate} cross:{crossover_rate} enc:{encoding_type} elitism:{elitism_option} minimize:{minimize}"
    )

    st.subheader(f"Results of {algorithm}")

    if "ga" in st.session_state:
        ga = st.session_state["ga"]
        with st.spinner(f"Running GA for {generations} generations..."):
            best_history, pop_history = ga.run(
                generations=generations, history=True, verbose=False
            )
        best = ga.best_individual.reshape(1, -1)
        st.session_state["pop_history"] = pop_history
        st.session_state["best_history"] = best_history
        st.session_state["best"] = best
        st.session_state["best_fitness"] = ga.best_fitness

        st.write(f"Best Individual: {st.session_state['best']}")
        st.write(f"Best Fitness: {st.session_state['best_fitness']}")

        if problem_type == "Parametric" and len(func_obj.bounds) == 2:
            film = parametric_population_history(
                pop_history,
                func_obj,
                colorscale=colorscale,
                best=best,
            )
            st.plotly_chart(film, use_container_width=True)

        if problem_type == "Combinatorial":
            film = combinatorial_population_history(
                best_history,
                func_obj,
                best=best[0],
            )
            st.plotly_chart(film, use_container_width=True)