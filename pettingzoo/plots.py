import plotly.graph_objs as go


def parametric_population_history(pop_history, func_obj, colorscale, best=None):
    if len(func_obj.bounds) != 2:
        raise ValueError(
            "Only 2D functions are supported for parametric_population_history."
        )

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
        frame = go.Frame(data=list(contour_fig.data) + [scatter], name=str(gen_idx + 1))
        frames.append(frame)

    if best is not None:
        final_frame = go.Frame(
            data=list(contour_fig.data)
            + [
                go.Scatter(
                    x=[ind[0] for ind in best],
                    y=[ind[1] for ind in best],
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
        xaxis=dict(range=[func_obj.bounds[0][0], func_obj.bounds[0][1]], title="X"),
        yaxis=dict(range=[func_obj.bounds[1][0], func_obj.bounds[1][1]], title="Y"),
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
    return fig


def combinatorial_population_history(pop_history, combinatorial_obj, best=None):
    base_fig = combinatorial_obj.plot(solution=None)
    data = list(base_fig.data)

    if pop_history:
        sol = pop_history[0]
        fig1 = combinatorial_obj.plot(solution=sol)
        data = list(fig1.data)

    frames = []
    for gen_idx in range(len(pop_history)):
        sol = pop_history[gen_idx]
        fig_gen = combinatorial_obj.plot(solution=sol)
        frames.append(go.Frame(data=list(fig_gen.data), name=str(gen_idx + 1)))

    if best is not None:
        fig_best = combinatorial_obj.plot(solution=best)
        frames.append(go.Frame(data=list(fig_best.data), name="Final"))
    layout = base_fig.layout
    layout["updatemenus"] = [
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
    ]
    layout["sliders"] = [
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
                font=dict(size=16), prefix="Generación: ", visible=True, xanchor="right"
            ),
            len=0.9,
        )
    ]
    fig = go.Figure(data=data, layout=layout, frames=frames)
    return fig