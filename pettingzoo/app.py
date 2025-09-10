import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from metazoo.bio.evolutionary import GeneticAlgorithm
from metazoo.bio.evolutionary.operators import selection, mutation, crossover
from metazoo.gym.mono import Function

available_algorithms = ["Genetic Algorithm"]
available_functions = Function().available_functions
colorscales = px.colors.named_colorscales()

st.set_page_config(page_title="Petting Zoo", layout="wide")
st.title("Petting Zoo")

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
    function = st.selectbox("Function", available_functions)
    colorscale = st.selectbox("Color Scale", colorscales)

    col_fun1, col_fun2 = st.columns(2)
    with col_fun1:
        reverse = st.checkbox(
            "Reverse function?",
            value=False,
            help="Flip sign (f(x) -> -f(x)). [This affects both visualization and function evaluation.]",
        )
    with col_fun2:
        minimize = st.checkbox(
            "Minimize?",
            value=True,
            help="If checked the algorithm searches minimum; otherwise it maximizes. Does not change the plotted surface.",
        )

if algorithm and function:
    st.subheader(f"Function: {function}")
    func_obj = Function(function, reverse=reverse)
    st.latex(func_obj.formula())
    plot_container = st.empty()

    # Config que define cuándo recalcular la figura base
    function_config = (function, reverse, colorscale)
    if (
        "function_fig" not in st.session_state
        or "function_config" not in st.session_state
        or st.session_state["function_config"] != function_config
    ):
        # Recalcular figura base SIN mejor individuo (no depende de params GA)
        base_fig = func_obj.plot(
            bounds=func_obj.bounds,
            dim=2,
            num_points=100,
            mode="surface",
            colorscale=colorscale,
        )
        base_fig.update_layout(title="")
        st.session_state["function_fig"] = base_fig
        st.session_state["function_config"] = function_config

    # Mostrar (si hay best y ya se corrió, la figura se actualizará tras Run)
    plot_container.plotly_chart(st.session_state["function_fig"])

    st.subheader(f"Parameters of {algorithm}")

    if algorithm == "Genetic Algorithm":
        st.caption(
            "Configure GA parameters: precision controls binary resolution; elitism preserves top individuals."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            population_size = st.number_input(
                "Population Size", min_value=10, max_value=50000, value=200, step=10
            )
            generations = st.number_input(
                "Generations", min_value=1, max_value=20000, value=200, step=10
            )
        with col2:
            mutation_rate = st.slider(
                "Mutation Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01
            )
            crossover_rate = st.slider(
                "Crossover Rate", min_value=0.0, max_value=1.0, value=0.8, step=0.01
            )
        with col3:
            encoding = st.selectbox("Encoding", ["binary", "real"], index=0)
            precision = st.number_input(
                "Precision (dec)",
                min_value=1,
                max_value=10,
                value=3,
                help="Desired decimal precision for binary encoding",
            )
        elitism_option = st.selectbox(
            "Elitism",
            options=["None", 0.1, 0.3, 1.0],
            index=0,
            help="Fraction of population preserved. 1.0 = full population (no evolutionary change).",
        )
        elitism_fraction = None if elitism_option == "None" else float(elitism_option)

        selection_function = selection.roulette
        crossover_function = crossover.onepoint
        mutation_function = (
            mutation.flip_bit if encoding == "binary" else mutation.gaussian
        )

        ga = GeneticAlgorithm(
            fitness_function=func_obj,
            crossover_function=crossover_function,
            mutation_function=mutation_function,
            selection_function=selection_function,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            encoding=encoding,
            bounds=func_obj.bounds,
            precision=precision,
            minimize=minimize,
            elitism=elitism_fraction,
        )
        st.session_state["ga"] = ga
        st.caption(
            f"GA => pop:{population_size} gens:{generations} mut:{mutation_rate} cross:{crossover_rate} enc:{encoding} prec:{precision} elitism:{elitism_option} minimize:{minimize}"
        )

    run = st.button(f"Run {algorithm}")

    if run:
        st.subheader(f"Results of {algorithm}")

        if "ga" in st.session_state:
            ga = st.session_state["ga"]
            with st.spinner(f"Running GA for {generations} generations..."):
                pop_history = ga.run(
                    generations=generations, history=True, verbose=False
                )
            best = ga.best_individual.reshape(1, -1)
            st.session_state["pop_history"] = pop_history
            st.session_state["best"] = best
            st.session_state["best_fitness"] = ga.best_fitness

            st.write(f"Best Individual: {st.session_state['best']}")
            st.write(f"Best Fitness: {st.session_state['best_fitness']}")
            # Solo ahora actualizamos la figura con el best
            updated_fig = func_obj.plot(
                bounds=func_obj.bounds,
                dim=2,
                num_points=100,
                mode="surface",
                colorscale=colorscale,
                population=st.session_state["best"],
            )
            updated_fig.update_layout(title="")
            st.session_state["function_fig"] = updated_fig
            plot_container.plotly_chart(updated_fig)

        else:
            st.warning("Please configure the algorithm before running it.")
            st.stop()

        if "pop_history" in st.session_state:
            if len(func_obj.bounds) == 2:
                pop_history = st.session_state["pop_history"]
                contour_fig = func_obj.plot(
                    bounds=func_obj.bounds,
                    dim=2,
                    num_points=100,
                    mode="contour",
                    colorscale=colorscale,
                )

                frames = []

                for gen_idx, gen in enumerate(pop_history):
                    scatter = go.Scatter(
                        x=[ind[0] for ind in gen],
                        y=[ind[1] for ind in gen],
                        mode="markers",
                        marker=dict(color="white", size=4, opacity=0.9),
                        name=f"Population Generation {gen_idx + 1}",
                    )
                    frame = go.Frame(
                        data=list(contour_fig.data) + [scatter], name=str(gen_idx + 1)
                    )
                    frames.append(frame)

                final_frame = go.Frame(
                    data=list(contour_fig.data)
                    + [
                        go.Scatter(
                            x=[ind[0] for ind in st.session_state["best"]],
                            y=[ind[1] for ind in st.session_state["best"]],
                            mode="markers",
                            marker=dict(color="blue", size=20, opacity=1),
                            name="Best Individual",
                        )
                    ],
                    name="Final",
                )
                frames.append(final_frame)

                data = list(contour_fig.data)
                data.append(
                    go.Scatter(
                        x=[ind[0] for ind in pop_history[0]],
                        y=[ind[1] for ind in pop_history[0]],
                        mode="markers",
                        marker=dict(color="red", size=8, opacity=0.7),
                        name="Population Generation 1",
                    )
                )

                layout = go.Layout(
                    xaxis=dict(
                        range=[func_obj.bounds[0][0], func_obj.bounds[0][1]], title="X"
                    ),
                    yaxis=dict(
                        range=[func_obj.bounds[1][0], func_obj.bounds[1][1]], title="Y"
                    ),
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="down",
                            showactive=False,
                            y=1,
                            x=1.1,
                            xanchor="right",
                            yanchor="top",
                            pad=dict(t=0, r=10),
                            buttons=[
                                dict(
                                    label="Play",
                                    method="animate",
                                    args=[
                                        None,
                                        {
                                            "frame": {"duration": 500, "redraw": True},
                                            "fromcurrent": True,
                                        },
                                    ],
                                ),
                                dict(
                                    label="Pause",
                                    method="animate",
                                    args=[
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                        },
                                    ],
                                ),
                            ],
                        )
                    ],
                )
                sliders = [
                    dict(
                        steps=[
                            dict(
                                method="animate",
                                args=[
                                    [str(i + 1)],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                    },
                                ],
                                label=str(i + 1),
                            )
                            for i in range(len(pop_history))
                        ],
                        active=0,
                        transition=dict(duration=300, easing="cubic-in-out"),
                        x=0.1,
                        y=0,
                        currentvalue=dict(
                            font=dict(size=16),
                            prefix="Generation: ",
                            visible=True,
                            xanchor="right",
                        ),
                        len=0.9,
                    )
                ]
                layout["sliders"] = sliders

                fig = go.Figure(data=data, layout=layout, frames=frames)
                st.plotly_chart(fig)
