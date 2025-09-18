# Combinatorial optimization problems

import numpy as np
import plotly.graph_objects as go


class TSP:
    """
    A class representing a TSP problem for optimization testing.
    """

    def __init__(self, coords=None, optimal_value=None, optimal_tour=None, distance_matrix=None, name="TSP"):
        self.coords = coords
        self.dimension = len(coords) if coords is not None else distance_matrix.shape[0] if distance_matrix is not None else ValueError("Either coords or distance_matrix must be provided.")
        self.optimal_value = optimal_value
        self.optimal_tour = optimal_tour if optimal_tour is not None else []
        self.__name__ = name

        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = self._compute_distance_matrix()

    def __call__(self, solution: list) -> float:
        """
        Calculate the total distance of the given solution path.
        """
        distance = 0.0
        for i in range(len(solution)):
            city1 = solution[i % len(solution)]
            city2 = solution[(i + 1) % len(solution)]
            distance += self.distance_matrix[city1, city2]
        return distance
    
    def _compute_distance_matrix(self):
        n = self.dimension
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = np.linalg.norm(np.array(self.coords[i]) - np.array(self.coords[j]))
        return matrix

    def plot(self, solution=None, show_optimal=False):
        """
        Visualize the route using plotly. If solution is None, shows only the cities.
        If show_optimal=True and there is an optimal_tour, also shows the optimal route.
        """

        if self.coords is None or len(self.coords) == 0:
            raise ValueError("No coordinates available to plot.")

        x, y = zip(*self.coords)
        fig = go.Figure()
        # Points
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=8, color="blue"),
                name="Cities",
            )
        )
        # Proposed route
        if solution is not None:
            route_x = [self.coords[i][0] for i in solution] + [
                self.coords[solution[0]][0]
            ]
            route_y = [self.coords[i][1] for i in solution] + [
                self.coords[solution[0]][1]
            ]
            fig.add_trace(
                go.Scatter(
                    x=route_x,
                    y=route_y,
                    mode="lines+markers",
                    marker=dict(size=6, color="red"),
                    line=dict(color="red"),
                    name="Route Found",
                )
            )
        # Optimal route
        if show_optimal and self.optimal_tour:
            opt = [i - 1 for i in self.optimal_tour]  # Convert to 0-based index
            opt_x = [self.coords[i][0] for i in opt] + [self.coords[opt[0]][0]]
            opt_y = [self.coords[i][1] for i in opt] + [self.coords[opt[0]][1]]
            fig.add_trace(
                go.Scatter(
                    x=opt_x,
                    y=opt_y,
                    mode="lines+markers",
                    marker=dict(size=6, color="green"),
                    line=dict(color="green", dash="dash"),
                    name="Optimal",
                )
            )
        fig.update_layout(
            title=f"TSP: {self.__name__}",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True,
        )
        return fig


class Berlin52(TSP):
    """
    Berlin52 TSP instance from TSPLIB.
    Optimal value: 7542.0
    """

    def __init__(self):
        coords = [
            (565.0, 575.0),
            (25.0, 185.0),
            (345.0, 750.0),
            (945.0, 685.0),
            (845.0, 655.0),
            (880.0, 660.0),
            (25.0, 230.0),
            (525.0, 1000.0),
            (580.0, 1175.0),
            (650.0, 1130.0),
            (1605.0, 620.0),
            (1220.0, 580.0),
            (1465.0, 200.0),
            (1530.0, 5.0),
            (845.0, 680.0),
            (725.0, 370.0),
            (145.0, 665.0),
            (415.0, 635.0),
            (510.0, 875.0),
            (560.0, 365.0),
            (300.0, 465.0),
            (520.0, 585.0),
            (480.0, 415.0),
            (835.0, 625.0),
            (975.0, 580.0),
            (1215.0, 245.0),
            (1320.0, 315.0),
            (1250.0, 400.0),
            (660.0, 180.0),
            (410.0, 250.0),
            (420.0, 555.0),
            (575.0, 665.0),
            (1150.0, 1160.0),
            (700.0, 580.0),
            (685.0, 595.0),
            (685.0, 610.0),
            (770.0, 610.0),
            (795.0, 645.0),
            (720.0, 635.0),
            (760.0, 650.0),
            (475.0, 960.0),
            (95.0, 260.0),
            (875.0, 920.0),
            (700.0, 500.0),
            (555.0, 815.0),
            (830.0, 485.0),
            (1170.0, 65.0),
            (830.0, 610.0),
            (605.0, 625.0),
            (595.0, 360.0),
            (1340.0, 725.0),
            (1740.0, 245.0),
        ]
        optimal_value = 7542.0
        optimal_tour = [
            1,
            49,
            32,
            45,
            19,
            41,
            8,
            9,
            10,
            43,
            33,
            51,
            11,
            52,
            14,
            13,
            47,
            26,
            27,
            28,
            12,
            25,
            4,
            6,
            15,
            5,
            24,
            48,
            38,
            37,
            40,
            39,
            36,
            35,
            34,
            44,
            46,
            16,
            29,
            50,
            20,
            23,
            30,
            2,
            7,
            42,
            21,
            17,
            3,
            18,
            31,
            22,
        ]
        super().__init__(coords, optimal_value, optimal_tour, name="Berlin52")


class Eil76(TSP):
    """
    Eil76 TSP instance from TSPLIB.
    Optimal value: 538.0
    """

    def __init__(self):
        coords = [
            (22, 22),
            (36, 26),
            (21, 45),
            (45, 35),
            (55, 20),
            (33, 34),
            (50, 50),
            (55, 45),
            (26, 59),
            (40, 66),
            (55, 65),
            (35, 51),
            (62, 35),
            (62, 57),
            (62, 24),
            (21, 36),
            (33, 44),
            (9, 56),
            (62, 48),
            (66, 14),
            (44, 13),
            (26, 13),
            (11, 28),
            (7, 43),
            (17, 64),
            (41, 46),
            (55, 34),
            (35, 16),
            (52, 26),
            (43, 26),
            (31, 76),
            (22, 53),
            (26, 29),
            (50, 40),
            (55, 50),
            (54, 10),
            (60, 15),
            (47, 66),
            (30, 60),
            (30, 50),
            (12, 17),
            (15, 14),
            (16, 19),
            (21, 48),
            (50, 30),
            (51, 42),
            (50, 15),
            (48, 21),
            (12, 38),
            (15, 56),
            (29, 39),
            (54, 38),
            (55, 57),
            (67, 41),
            (10, 70),
            (6, 25),
            (65, 27),
            (40, 60),
            (70, 64),
            (64, 4),
            (36, 6),
            (30, 20),
            (20, 30),
            (15, 5),
            (50, 70),
            (57, 72),
            (45, 42),
            (38, 33),
            (50, 4),
            (66, 8),
            (59, 5),
            (35, 60),
            (27, 24),
            (40, 20),
            (40, 37),
            (40, 40),
        ]
        optimal_value = 538.0
        optimal_tour = [
            1,
            33,
            63,
            16,
            3,
            44,
            32,
            9,
            39,
            72,
            58,
            10,
            31,
            55,
            25,
            50,
            18,
            24,
            49,
            23,
            56,
            41,
            43,
            42,
            64,
            22,
            61,
            21,
            47,
            36,
            69,
            71,
            60,
            70,
            20,
            37,
            5,
            15,
            57,
            13,
            54,
            19,
            14,
            59,
            66,
            65,
            38,
            11,
            53,
            7,
            35,
            8,
            46,
            34,
            52,
            27,
            45,
            29,
            48,
            30,
            4,
            75,
            76,
            67,
            26,
            12,
            40,
            17,
            51,
            6,
            68,
            2,
            74,
            28,
            62,
            73,
        ]
        super().__init__(coords, optimal_value, optimal_tour, name="Eil76")


class TSP:
    Berlin52 = Berlin52
    Eil76 = Eil76
    Custom = TSP 

    @classmethod
    def available_problems(cls):
        return [
            name
            for name in cls.__dict__
            if not name.startswith("__") and isinstance(getattr(cls, name), type)
        ]


class NQueens:
    """
    N-Queens problem for combinatorial optimization testing.
    The goal is to place N queens on an NxN chessboard such that no two queens threaten each other.
    """

    def __init__(self, n):
        self.n = n
        self.__name__ = f"{n}-Queens"

    def __call__(self, solution: list) -> int:
        """
        Calculate the fitness of the given solution.
        The fitness is defined as the number of non-attacking pairs of queens.
        """
        # We assume the algorithm is always maximizing the fitness
        # So we return the number of non-attacking pairs (the higher the better)
        # Also, if two sols have the same amount of attacks,
        # we prefer the one with fewer total queens.
        max_attacks = (self.n * (self.n - 1)) // 2  # Total pairs of queens
        actual_attacks = len(self.attacks(solution))
        fitness = max_attacks - actual_attacks
        return fitness

    def attacks(self, solution: list) -> int:
        """
        Calculate the number of pairs of queens that are attacking each other.
        The solution is represented as a list where the index represents the column and the value at that index represents the row.
        """
        attacks = []
        # All pairs of queens
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                # Check if they are in the same row or on the same diagonal
                # solution[i] == solution[j] checks for same row
                # abs(solution[i] - solution[j]) == abs(i - j) checks for same diagonal
                if solution[i] == solution[j] or abs(solution[i] - solution[j]) == abs(
                    i - j
                ):
                    # Update count of attacking pairs
                    attacks.append((i, j))
        return attacks

    def plot(self, solution=None, attacks: bool = True):
        """
        Visualiza el tablero N-Queens usando plotly. El grid está centrado en 0.5 y las coordenadas de la solución se mantienen originales.
        """
        fig = go.Figure()
        # GRID
        for i in range(self.n + 1):
            fig.add_shape(
                type="line",
                x0=i - 0.5,
                y0=-0.5,
                x1=i - 0.5,
                y1=self.n - 0.5,
                line=dict(color="LightGray", width=2),
            )
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=i - 0.5,
                x1=self.n - 0.5,
                y1=i - 0.5,
                line=dict(color="LightGray", width=2),
            )
        # Queens
        if solution is not None:
            for col, row in enumerate(solution):
                fig.add_trace(
                    go.Scatter(
                        x=[col],
                        y=[row],
                        mode="markers",
                        marker=dict(size=100/self.n + 1, color="blue", symbol="x"),
                        name="Queen",
                    )
                )
        # Attacks
        if attacks:
            for (col1, col2) in self.attacks(solution):
                row1 = solution[col1]
                row2 = solution[col2]
                fig.add_trace(
                    go.Scatter(
                        x=[col1, col2],
                        y=[row1, row2],
                        mode="lines",
                        line=dict(color="red", width=4),
                        name="Attack",
                    )
                )
        fig.update_layout(
            title=f"{self.n}-Queens",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(self.n)),
                ticktext=list(range(self.n)),
                range=[-0.5, self.n - 0.5],
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(self.n)),
                ticktext=list(range(self.n)),
                range=[-0.5, self.n - 0.5],
            ),
            xaxis_title="Column",
            yaxis_title="Row",
            showlegend=False,
            width=500,
            height=500,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig
