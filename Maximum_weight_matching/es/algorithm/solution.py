import abc


class Solution(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def size(self):
        """
        Returns:
            dim(`int`): The dimension of the parameter vector
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self):
        """
        Returns:
            parameter vector(`np.array`)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_params(self, params):
        """
        Args:
            params(`np.array`): A flat numpy array representing the solution

        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, instance):
        """
        Returns:
            fitness(`float`): The fitness or score of the current solution
        """
        raise NotImplementedError