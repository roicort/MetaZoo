# Mono-objective optimization test functions

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

class Function:

    def __init__(self, name: str = 'Ackley', reverse: bool = False):
        self.name = name
        self.reverse = reverse
        self.metadata = {
            
            'Rastrigin': {
                'formula': r'f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[x_i^2 - 10 \cos(2\pi x_i)\right]',
                'bounds': [(-5.12, 5.12), (-5.12, 5.12)]
            },
            'Ackley': {
                'formula': r'f(\mathbf{x}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}\right) - \exp\left(\frac{1}{n} \sum_{i=1}^n \cos(2\pi x_i)\right) + 20 + e',
                'bounds': [(-5, 5), (-5, 5)]
            },
            'Sphere': {
                'formula': r'f(\mathbf{x}) = \sum_{i=1}^n x_i^2',
                'bounds': [(-5.12, 5.12),(-5.12, 5.12)]
            },
            'Rosenbrock': {
                'formula': r'f(x_1, x_2) = (1 - x_1)^2 + 100 (x_2 - x_1^2)^2',
                'bounds': [(-2, 2), (-1, 3)]
            },
            'Beale': {
                'formula': r'f(x_1, x_2) = (1.5 - x_1 + x_1 x_2)^2 + (2.25 - x_1 + x_1 x_2^2)^2 + (2.625 - x_1 + x_1 x_2^3)^2',
                'bounds': [(-4.5, 4.5), (-4.5, 4.5)]
            },
            'GoldsteinPrice': {
                'formula': r'f(x_1, x_2) = (1 + \frac{(x_1 + x_2 + 1)^2}{(x_1 + x_2)^2}) (19 - 14x_1 + 3x_1^2 - 14x_2 + 6x_1x_2)',
                'bounds': [(-2, 2), (-2, 2)]
            },
            'Booth': {
                'formula': r'f(x_1, x_2) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2',
                'bounds': [(-10, 10), (-10, 10)]
            },

            'Bukin': {
                'formula': r'f(x_1, x_2) = 100 \sqrt{|x_2 - 0.01 x_1^2|} + 0.01 |x_1 + 10|',
                'bounds': [(-15, -5), (-3, 3)]
            },
            'Matyas': {
                'formula': r'f(x_1, x_2) = 0.26(x_1^2 + x_2^2) - 0.48x_1x_2',
                'bounds': [(-10, 10), (-10, 10)]
            },
            'Levi_N13': {
                'formula': r'f(x_1, x_2) = \sin(3\pi x_1)^2 + (x_1 - 1)^2(1 + \sin(3\pi x_2)^2) + (x_2 - 1)^2(1 + \sin(2\pi x_2)^2)',
                'bounds': [(-10, 10), (-10, 10)]
            },
            'Griewank': {
                'formula': r'f(\mathbf{x}) = 1 + \frac{1}{4000} \sum_{i=1}^n x_i^2 - \prod_{i=1}^n \cos\left(\frac{x_i}{\sqrt{i}}\right)',
                'bounds': [(-600, 600), (-600, 600)]
            },
            'Himmelblau': {
                'formula': r'f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2',
                'bounds': [(-5, 5), (-5, 5)]
            },
            'ThreeHumpCamel': {
                'formula': r'f(x_1, x_2) = 2x_1^2 - 1.05x_1^4 + \frac{x_1^6}{6} + x_1x_2 + x_2^2',
                'bounds': [(-5, 5), (-5, 5)]
            },
            'Easom': {
                'formula': r'f(x_1, x_2) = -\cos(x_1) \cos(x_2) \exp\left(-\left(x_1 - \pi\right)^2 - \left(x_2 - \pi\right)^2\right)',
                'bounds': [(-100, 100), (-100, 100)]
            },
            'Cross_In_Tray': {
                'formula': r'f(x_1, x_2) = \sin(x_1) \sin(x_2) \left(1 + 0.001(x_1^2 + x_2^2)\right)',
                'bounds': [(-10, 10), (-10, 10)]
            },
            'EggHolder': {
                'formula': r'f(x_1, x_2) = -(x_2 + 47) \sin\left(\sqrt{|x_1/2 + (x_2 + 47)|}\right) - x_1 \sin\left(\sqrt{|x_1 - (x_2 + 47)|}\right)',
                'bounds': [(-512, 512), (-512, 512)]
            },
            'HolderTable': {
                'formula': r'f(x_1, x_2) = -\sin(x_1) \cos(x_2) \exp\left(-\left(x_1 + x_2\right)^2\right)',
                'bounds': [(-10, 10), (-10, 10)]
            },
            'McCormick': {
                'formula': r'f(x_1, x_2) = \sin(x_1 + x_2) + (x_1 - x_2)^2 - 1.5x_1 + 2.5x_2 + 1',
                'bounds': [(-1.5, 4), (-3, 4)]
            },
            'Schaffer_N2': {
                'formula': r'f(x_1, x_2) = 0.5 + \frac{\sin^2(x_1^2 - x_2^2)}{1 + 0.001(x_1^2 + x_2^2)}',
                'bounds': [(-100, 100), (-100, 100)]
            },
            'StyblinskiTang': {
                'formula': r'f(x_1, x_2) = \sum_{i=1}^n \left[x_i^3 - 5x_i\right]',
                'bounds': [(-5, 5), (-5, 5)]
            },
            'Schekel': {
                'formula': r'f(x_1, x_2) = \left(1 - \frac{x_1}{\sqrt{1 + x_2^2}}\right)^2 + \left(1 - \frac{x_2}{\sqrt{1 + x_1^2}}\right)^2',
                'bounds': [(-100, 100), (-100, 100)]
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
            print("No LaTeX formula available for this function.")

    def __call__(self, X: np.ndarray) -> float:
        value = getattr(self, self.name)(X)
        if self.reverse:
            return -value
        return value
    
    @staticmethod
    def Rastrigin(X: np.ndarray) -> float:
        """
        Rastrigin Function
        Wiki: https://en.wikipedia.org/wiki/Rastrigin_function
        """
        n = len(X)
        A = 10
        return A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X))
    
    @staticmethod
    def Ackley(X: np.ndarray) -> float:
        """
        Ackley
        Wiki: https://en.wikipedia.org/wiki/Ackley_function
        """
        n = len(X)
        square_sum = (1 / n) * np.sum(X * X)
        trigonometric_sum = (1 / n) * np.sum(np.cos(2 * np.pi * X))
        return -20 * np.exp(-0.2 * np.sqrt(square_sum)) - np.exp(trigonometric_sum) + 20 + np.e
    
    @staticmethod
    def Sphere(X: np.ndarray) -> float:
        """
        Sphere
        """
        return np.sum(X * X)
    
    @staticmethod
    def Rosenbrock(X: np.ndarray) -> float:
        """
        Rosenbrock
        Wiki: https://en.wikipedia.org/wiki/Rosenbrock_function
        """
        if len(X) != 2:
            raise ValueError("Rosenbrock function is only defined for 2D inputs.")
        x1, x2 = X
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2

    @staticmethod
    def Beale(X: np.ndarray) -> float:
        """
        Beale
        Wiki: https://en.wikipedia.org/wiki/Beale_function
        """
        if len(X) != 2:
            raise ValueError("Beale function is only defined for 2D inputs.")
        x1, x2 = X
        return (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2

    @staticmethod
    def GoldsteinPrice(X: np.ndarray) -> float:
        """
        Goldstein-Price
        """
        if len(X) != 2:
            raise ValueError("Goldstein-Price function is only defined for 2D inputs.")
        x1, x2 = X
        term1 = 1 + ((x1 + x2 + 1)**2) * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2)
        term2 = (x1 + x2)**2
        return term1 / term2
    
    @staticmethod
    def Booth(X: np.ndarray) -> float:
        """
        Booth
        """
        if len(X) != 2:
            raise ValueError("Booth function is only defined for 2D inputs.")
        x1, x2 = X
        return (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2

    @staticmethod
    def Bukin(X: np.ndarray) -> float:
        """
        Bukin
        """
        if len(X) != 2:
            raise ValueError("Bukin function is only defined for 2D inputs.")
        x1, x2 = X
        return 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10)
    
    @staticmethod
    def Matyas(X: np.ndarray) -> float:
        """
        Matyas
        """
        if len(X) != 2:
            raise ValueError("Matyas function is only defined for 2D inputs.")
        x1, x2 = X
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

    @staticmethod
    def Levi_N13(X: np.ndarray) -> float:
        """
        Levi N.13
        """
        if len(X) != 2:
            raise ValueError("Levi N.13 function is only defined for 2D inputs.")
        x1, x2 = X
        return np.sin(x1 + x2)**2 + (x1 - x2)**2
    
    @staticmethod
    def Griewank(X: np.ndarray) -> float:
        """
        Griewank
        """
        if len(X) != 2:
            raise ValueError("Griewank function is only defined for 2D inputs.")
        x1, x2 = X
        return 1 + (x1**2 + x2**2) / 4000 - np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
    
    @staticmethod
    def Himmelblau(X: np.ndarray) -> float:
        """
        Himmelblau
        """
        if len(X) != 2:
            raise ValueError("Himmelblau function is only defined for 2D inputs.")
        x1, x2 = X
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    
    @staticmethod
    def ThreeHumpCamel(X: np.ndarray) -> float:
        """
        Three Hump Camel
        """
        if len(X) != 2:
            raise ValueError("Three Hump Camel function is only defined for 2D inputs.")
        x1, x2 = X
        return 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2
    
    @staticmethod
    def Aasom(X: np.ndarray) -> float:
        """
        Easom
        """
        if len(X) != 2:
            raise ValueError("Easom function is only defined for 2D inputs.")
        x1, x2 = X
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))
    
    @staticmethod
    def Cross_In_Tray(X: np.ndarray) -> float:
        """
        Cross In Tray
        """
        if len(X) != 2:
            raise ValueError("Cross In Tray function is only defined for 2D inputs.")
        x1, x2 = X
        return -0.5 * np.sin(x1) * np.sin(x2) * np.cos(x1) * np.cos(x2) * np.exp(1 - (x1**2 + x2**2))
    
    @staticmethod
    def EggHolder(X: np.ndarray) -> float:
        """
        EggHolder
        """
        if len(X) != 2:
            raise ValueError("EggHolder function is only defined for 2D inputs.")
        x1, x2 = X
        return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    
    @staticmethod
    def HolderTable(X: np.ndarray) -> float:
        """
        Holder Table
        """
        if len(X) != 2:
            raise ValueError("Holder Table function is only defined for 2D inputs.")
        x1, x2 = X
        return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi)))
    
    @staticmethod
    def McCormick(X: np.ndarray) -> float:
        """
        McCormick
        """
        if len(X) != 2:
            raise ValueError("McCormick function is only defined for 2D inputs.")
        x1, x2 = X
        return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1
    
    @staticmethod
    def Schaffer_N2(X: np.ndarray) -> float:
        """
        Schaffer N.2
        """
        if len(X) != 2:
            raise ValueError("Schaffer N.2 function is only defined for 2D inputs.")
        x1, x2 = X
        return 0.5 + (np.sin(x1**2 - x2**2)**2 - 0.5) / (1 + 0.001 * (x1**2 + x2**2))**2
    
    @staticmethod
    def StyblinskiTang(X: np.ndarray) -> float:
        """
        Styblinski-Tang
        """
        if len(X) != 2:
            raise ValueError("Styblinski-Tang function is only defined for 2D inputs.")
        x1, x2 = X
        return 0.5 * (x1**4 - 16 * x1**2 + 5 * x1 + x2**4 - 16 * x2**2 + 5 * x2)
    
    @staticmethod
    def Shekel(X: np.ndarray) -> float:
        """
        Shekel
        """
        if len(X) != 2:
            raise ValueError("Shekel function is only defined for 2D inputs.")
        x1, x2 = X
        A = np.array([4, 4, 4, 4, 4])
        B = np.array([4, 4, 4, 4, 4])
        C = np.array([1, 2, 3, 4, 5])
        denom = (x1 - A)**2 + (x2 - B)**2 + C
        return -np.sum(1 / denom)

    def plot(self, bounds=(-5, 5), dim=1, num_points=100, population=None, mode='surface') -> go.Figure:
        pio.renderers.default = 'notebook'  # Puedes cambiar a 'browser' si lo prefieres
        metadata = getattr(self, 'metadata', {}).get(self.name, None)
        title = f'${{\\text{{{self.name.capitalize()} function: }} {metadata["formula"]}}}$' if metadata else f'{self.name.capitalize()} function'
        # Use Original Function
        func = getattr(self, self.name)
        if dim == 1:
            if isinstance(bounds[0], (tuple, list)):
                x_bounds = bounds[0]
            else:
                x_bounds = bounds
            x = np.linspace(x_bounds[0], x_bounds[1], num_points)
            y = np.array([func(np.array([xi])) for xi in x])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Function'))
            if population is not None:
                pop_x = population[:, 0]
                pop_y = np.array([func(np.array([xi])) for xi in pop_x])
                fig.add_trace(go.Scatter(x=pop_x, y=pop_y, mode='markers', name='Population', marker=dict(color='red', size=8)))
            fig.update_layout(title=title, xaxis_title='x', yaxis_title='f(x)')
        elif dim == 2:
            if isinstance(bounds[0], (tuple, list)) and isinstance(bounds[1], (tuple, list)):
                x_bounds = bounds[0]
                y_bounds = bounds[1]
            else:
                x_bounds = y_bounds = bounds
            x = np.linspace(x_bounds[0], x_bounds[1], num_points)
            y = np.linspace(y_bounds[0], y_bounds[1], num_points)
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
                raise ValueError("Mode should be 'surface' or 'contour'.")
        else:
            raise ValueError('Only 1D or 2D functions are supported for plotting.')
        
        return fig