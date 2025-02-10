from abc import ABC, abstractmethod

class AbstractIntegrator(ABC):
    def __init__(self, solver='LSODA', time_step=1e-3):
        self.solver = solver
        self.time_step = time_step

    @abstractmethod
    def system(self, t, x, p):
        """
        Abstract method to define the system of equations.
        Must be implemented by subclasses.

        Parameters:
        - t: float, time
        - x: array-like, current state vector of the sytem
        - p: array-like, parameters of the system
        """
        pass

    @abstractmethod
    def integrate(self, t_span, t_tr):
        """
        Abstract method to integrate the system of equations.
        Must be implemented by subclasses.
        Integrates the system over the given time span.

        Parameters:
        - t_span: tuple, (start_time, end_time) for integration.
        - t_tr: transient time.

        Returns:
        - sol: object, solution object from `solve_ivp`.
        """
        pass