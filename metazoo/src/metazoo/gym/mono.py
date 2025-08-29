# Mono-objective optimization test functions

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

class Function:

    def __init__(self, name: str, reverse: bool = False):
        self.name = name
        self.reverse = reverse
        self.metadata = {
            'sphere': {
                'formula': r'f(\mathbf{x}) = \sum_{i=1}^n x_i^2',
                'bounds': [(-5.12, 5.12)]
            },
            'bukin': {
                'formula': r'f(x_1, x_2) = 100 \sqrt{|x_2 - 0.01 x_1^2|} + 0.01 |x_1 + 10|',
                'bounds': [(-15, -5), (-3, 3)]
            },
            'eggholder': {
                'formula': r'f(x_1, x_2) = -(x_2 + 47) \sin\left(\sqrt{|x_1/2 + (x_2 + 47)|}\right) - x_1 \sin\left(\sqrt{|x_1 - (x_2 + 47)|}\right)',
                'bounds': [(-512, 512), (-512, 512)]
            },
            'himmelblau': {
                'formula': r'f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2',
                'bounds': [(-5, 5), (-5, 5)]
            },
            'ackley': {
                'formula': r'f(\mathbf{x}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}\right) - \exp\left(\frac{1}{n} \sum_{i=1}^n \cos(2\pi x_i)\right) + 20 + e',
                'bounds': [(-32, 32), (-32, 32)]
            },
            'rastrigin': {
                'formula': r'f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[x_i^2 - 10 \cos(2\pi x_i)\right]',
                'bounds': [(-5.12, 5.12), (-5.12, 5.12)]
            }
        }

        self.bounds = self.metadata.get(name, {}).get('bounds', [])
        self.available_functions = list(self.metadata.keys())

    def formula(self):
        from IPython.display import display, Math
        formula = self.metadata.get(self.name, None)
        if formula:
            display(Math(formula))
        else:
            print("No hay fórmula LaTeX disponible para esta función.")

    def __call__(self, X: np.ndarray) -> float:
        value = getattr(self, self.name)(X)
        if self.reverse:
            return -value
        return value
    
    @staticmethod
    def sphere(X: np.ndarray) -> float:
        """
        Esfera
        Fórmula: :math:`f(\mathbf{x}) = \sum_{i=1}^n x_i^2`
        """
        return np.sum(X * X)
    
    @staticmethod
    def bukin(X: np.ndarray) -> float:
        """
        Bukin
        Fórmula: :math:`f(x_1, x_2) = 100 \sqrt{|x_2 - 0.01 x_1^2|} + 0.01 |x_1 + 10|`
        """
        if len(X) != 2:
            raise ValueError("Bukin function is only defined for 2D inputs.")
        x1, x2 = X
        return 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10)
    
    @staticmethod
    def eggholder(X: np.ndarray) -> float:
        """
        Eggholder
        Fórmula: :math:`f(x_1, x_2) = -(x_2 + 47) \sin\left(\sqrt{|x_1/2 + (x_2 + 47)|}\right) - x_1 \sin\left(\sqrt{|x_1 - (x_2 + 47)|}\right)`
        """
        if len(X) != 2:
            raise ValueError("Eggholder function is only defined for 2D inputs.")
        x1, x2 = X
        return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

    @staticmethod
    def himmelblau(X: np.ndarray) -> float:
        """
        Himmelblau
        Fórmula: :math:`f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
        """
        if len(X) != 2:
            raise ValueError("Himmelblau function is only defined for 2D inputs.")
        x1, x2 = X
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

    @staticmethod
    def ackley(X: np.ndarray) -> float:
        """
        Ackley
        Fórmula: :math:`f(\mathbf{x}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}\right) - \exp\left(\frac{1}{n} \sum_{i=1}^n \cos(2\pi x_i)\right) + 20 + e`
        """
        n = len(X)
        square_sum = (1 / n) * np.sum(X * X)
        trigonometric_sum = (1 / n) * np.sum(np.cos(2 * np.pi * X))
        return -20 * np.exp(-0.2 * np.sqrt(square_sum)) - np.exp(trigonometric_sum) + 20 + np.e

    @staticmethod
    def rastrigin(X: np.ndarray) -> float:
        """
        Rastrigin
        Fórmula: :math:`f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[x_i^2 - 10 \cos(2\pi x_i)\right]`
        """
        n = len(X)
        A = 10
        return A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X))

    def plot(self, bounds=(-5, 5), dim=1, num_points=100, population=None, mode='surface'):
        pio.renderers.default = 'notebook'  # Puedes cambiar a 'browser' si lo prefieres
        metadata = getattr(self, 'metadata', {}).get(self.name, None)
        title = f'{self.name} function'
        if metadata:
            title += (
                f'<br><span style="font-size:16px">$${metadata["formula"]}$$</span>'
            )
        # Usar original function
        func = getattr(self, self.name)
        if dim == 1:
            x = np.linspace(bounds[0], bounds[1], num_points)
            y = np.array([func(np.array([xi])) for xi in x])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Function'))
            if population is not None:
                pop_x = population[:, 0]
                pop_y = np.array([func(np.array([xi])) for xi in pop_x])
                fig.add_trace(go.Scatter(x=pop_x, y=pop_y, mode='markers', name='Population', marker=dict(color='red', size=8)))
            fig.update_layout(title=title, xaxis_title='x', yaxis_title='f(x)')
            fig.show()
        elif dim == 2:
            x = np.linspace(bounds[0], bounds[1], num_points)
            y = np.linspace(bounds[0], bounds[1], num_points)
            X, Y = np.meshgrid(x, y)
            Z = np.array([func(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
            Z = Z.reshape(X.shape)
            fig = go.Figure()
            if mode == 'surface':
                fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7, name='Function'))
                if population is not None:
                    pop_x = population[:, 0]
                    pop_y = population[:, 1]
                    pop_z = np.array([func(np.array([xi, yi])) for xi, yi in zip(pop_x, pop_y)])
                    fig.add_trace(go.Scatter3d(x=pop_x, y=pop_y, z=pop_z, mode='markers', name='Population', marker=dict(color='red', size=4)))
                fig.update_layout(title=title, scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'))
            elif mode == 'contour':
                fig.add_trace(go.Contour(z=Z, x=x, y=y, colorscale='Viridis', name='Function'))
                if population is not None:
                    pop_x = population[:, 0]
                    pop_y = population[:, 1]
                    fig.add_trace(go.Scatter(x=pop_x, y=pop_y, mode='markers', name='Population', marker=dict(color='red', size=8)))
                fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
            else:
                raise ValueError("El modo debe ser 'surface' o 'contour'.")
            fig.show()
        else:
            raise ValueError('Solo se soporta graficar funciones de 1 o 2 dimensiones.')