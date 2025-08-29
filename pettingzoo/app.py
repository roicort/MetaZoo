import streamlit as st

from metazoo.bio.evolutionary import GeneticAlgorithm
from metazoo.bio.evolutionary.operators import selection, mutation, crossover
from metazoo.gym.mono import Function

available_algorithms = ['Genetic Algorithm']
available_functions = Function().available_functions

st.title("Petting Zoo")

algorithm = st.selectbox("Selecciona un algoritmo", available_algorithms)
function = st.selectbox("Selecciona una función", available_functions)

if algorithm and function:
	reverse = st.checkbox("Revertir la función (reverse)", value=True)
	func_obj = Function(function, reverse=reverse)
	fig = func_obj.plot(bounds=func_obj.bounds, dim=2, num_points=100, mode='surface')
	st.plotly_chart(fig)

	st.subheader("Parámetros del Algoritmo Genético")
	population_size = st.number_input("Tamaño de población", min_value=10, max_value=10000, value=100)
	mutation_rate = st.slider("Tasa de mutación", min_value=0.0, max_value=1.0, value=0.1)
	crossover_rate = st.slider("Tasa de cruce", min_value=0.0, max_value=1.0, value=0.7)
	generations = st.number_input("Generaciones", min_value=1, max_value=10000, value=100)

	run_ga = st.button("Ejecutar Algoritmo Genético")

	if run_ga:
		# Configuración y ejecución del algoritmo genético
		encoding = 'binary'
		mutation_function = mutation.flip_bit if encoding == 'binary' else mutation.gaussian
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
		ga.run(generations=generations)
		if encoding == 'binary':
			best = ga.decode(ga.best_individual).reshape(1, -1)
		else:
			best = ga.best_individual.reshape(1, -1)
		st.write(f"Mejor individuo: {best}")
		st.write(f"Mejor fitness: {ga.best_fitness}")

		contour = func_obj.plot(bounds=func_obj.bounds, dim=2, num_points=100, mode='contour', population=best)
		surface = func_obj.plot(bounds=func_obj.bounds, dim=2, num_points=100, mode='surface', population=best)

		st.plotly_chart(contour)
		st.plotly_chart(surface)