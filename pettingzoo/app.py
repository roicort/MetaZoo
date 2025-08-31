import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from metazoo.bio.evolutionary import GeneticAlgorithm
from metazoo.bio.evolutionary.operators import selection, mutation, crossover
from metazoo.gym.mono import Function

available_algorithms = ['Genetic Algorithm']
available_functions = Function().available_functions
colorscales = px.colors.named_colorscales()

st.title("Petting Zoo")

algorithm = st.selectbox("Algorithm Selection", available_algorithms)
function = st.selectbox("Function Selection", available_functions)
colorscale = st.selectbox("Color Selection", colorscales)

if algorithm and function:
	reverse = st.checkbox("Reverse Function?", value=True)
	func_obj = Function(function, reverse=reverse)
	fig = func_obj.plot(bounds=func_obj.bounds, dim=2, num_points=100, mode='surface', colorscale=colorscale)
	st.plotly_chart(fig)

	st.subheader(f"Parameters of {algorithm}")

	if algorithm == 'Genetic Algorithm':
		population_size = st.number_input("Population Size", min_value=10, max_value=10000, value=100)
		mutation_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.1)
		crossover_rate = st.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=0.7)
		encoding = 'binary'
		mutation_function = mutation.flip_bit if encoding == 'binary' else mutation.gaussian
		generations = st.number_input("Generations", min_value=1, max_value=10000, value=100)

		ga = GeneticAlgorithm(
			fitness_function=func_obj,
			crossover_function=crossover.onepoint,
			mutation_function=mutation_function,
			selection_function=selection.roulette,
			population_size=population_size,
			mutation_rate=mutation_rate,
			crossover_rate=crossover_rate,
			encoding=encoding,
			bounds=func_obj.bounds,
			binary_precision=3
		)

		st.session_state['ga'] = ga

	run = st.button(f"Run {algorithm}")

	if run:

		st.subheader(f"Results of {algorithm}")

		if 'ga' in st.session_state:
			
			ga = st.session_state['ga']
			pop_history = ga.run(generations=generations, history=True)
			best = ga.best_individual.reshape(1, -1)
			st.session_state['pop_history'] = pop_history
			st.session_state['best'] = best
			st.session_state['best_fitness'] = ga.best_fitness

			st.write(f"Best Individual: {st.session_state['best']}")
			st.write(f"Best Fitness: {st.session_state['best_fitness']}")
			#contour = func_obj.plot(bounds=func_obj.bounds, dim=2, num_points=100, mode='contour', population=st.session_state['best'])
			surface = func_obj.plot(bounds=func_obj.bounds, dim=2, num_points=100, mode='surface', population=st.session_state['best'], colorscale=colorscale)
			#st.plotly_chart(contour)
			st.plotly_chart(surface)

		else:
			st.warning("Please configure the algorithm before running it.")
			st.stop()
		

		if 'pop_history' in st.session_state:

			st.subheader(f"Steps of {algorithm}")

			if len(func_obj.bounds) == 2:

				pop_history = st.session_state['pop_history']
				contour_fig = func_obj.plot(bounds=func_obj.bounds, dim=2, num_points=100, mode='contour', colorscale=colorscale)

				frames = []

				for gen_idx, gen in enumerate(pop_history):
					scatter = go.Scatter(
						x=[ind[0] for ind in gen],
						y=[ind[1] for ind in gen],
						mode='markers',
						marker=dict(color='white', size=4, opacity=0.9),
						name=f'Population Epoch {gen_idx+1}'
					)
					frame = go.Frame(data=list(contour_fig.data) + [scatter], name=str(gen_idx+1))
					frames.append(frame)

				final_frame = go.Frame(data=list(contour_fig.data) + [go.Scatter(
					x=[ind[0] for ind in st.session_state['best']],
					y=[ind[1] for ind in st.session_state['best']],
					mode='markers',
					marker=dict(color='blue', size=20, opacity=1),
					name='Best Individual'
				)], name='Final')
				frames.append(final_frame)

				data = list(contour_fig.data)
				data.append(go.Scatter(
					x=[ind[0] for ind in pop_history[0]],
					y=[ind[1] for ind in pop_history[0]],
					mode='markers',
					marker=dict(color='red', size=8, opacity=0.7),
					name='Population Epoch 1'
				))

				layout = go.Layout(
					xaxis=dict(range=[func_obj.bounds[0][0], func_obj.bounds[0][1]], title='X'),
					yaxis=dict(range=[func_obj.bounds[1][0], func_obj.bounds[1][1]], title='Y'),
					updatemenus=[dict(
						type='buttons',
						direction='down',
						showactive=False,
						y=1,
						x=1.1,
						xanchor='right',
						yanchor='top',
						pad=dict(t=0, r=10),
						buttons=[
							dict(label='Play', method='animate', args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]),
							dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}])
						]
					)]
				)
				sliders = [dict(
					steps=[dict(method='animate', args=[[str(i+1)], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}], label=str(i+1)) for i in range(len(pop_history))],
					active=0,
					transition=dict(duration=300, easing='cubic-in-out'),
					x=0.1,
					y=0,
					currentvalue=dict(font=dict(size=16), prefix='Epoch: ', visible=True, xanchor='right'),
					len=0.9
				)]
				layout['sliders'] = sliders

				fig = go.Figure(data=data, layout=layout, frames=frames)
				st.plotly_chart(fig)