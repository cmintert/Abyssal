from numpy.random import Generator, PCG64


class RandomGenerator:
    """
    A centralized random number generator class using NumPy's modern Generator API.
    This provides more control over random number generation and uses better algorithms
    than the legacy NumPy random functions.
    """

    _instance = None

    @classmethod
    def get_instance(cls, seed=None):
        """
        Get the singleton instance of RandomGenerator.

        Args:
            seed (int, optional): Seed for the random number generator.
                                 If None, uses the seed from config.py if available,
                                 otherwise uses a random seed.

        Returns:
            RandomGenerator: The singleton instance
        """
        if cls._instance is None:
            cls._instance = RandomGenerator(seed)
        return cls._instance

    @classmethod
    def reset_instance(cls, seed=None):
        """
        Reset the singleton instance with a new seed.

        Args:
            seed (int, optional): New seed for the random number generator.

        Returns:
            RandomGenerator: The new singleton instance
        """
        cls._instance = RandomGenerator(seed)
        return cls._instance

    def __init__(self, seed=None):
        """
        Initialize the random number generator.

        Args:
            seed (int, optional): Seed for the random number generator.
                                 If None, uses the seed from config.py if available,
                                 otherwise uses a random seed.
        """
        # Try to import config for the default seed
        try:
            import config
            default_seed = getattr(config, 'SEED', None)
        except (ImportError, AttributeError):
            default_seed = None

        # Use provided seed, or default from config, or generate a random one
        self.seed = seed if seed is not None else default_seed

        # If still None, generate a random seed
        if self.seed is None:
            # Use the current time to create a random seed
            import time
            self.seed = int(time.time() * 1000) % 2 ** 32
            print(f"RandomGenerator: Using random seed {self.seed}")

        # Create the generator
        self.rng = Generator(PCG64(self.seed))

    def rand(self, *args):
        """
        Random values in a given shape.
        Equivalent to np.random.rand

        Args:
            *args: Dimensions of the return array

        Returns:
            ndarray: Random values of shape args
        """
        return self.rng.random(size=args if args else None)

    def randint(self, low, high=None, size=None, dtype=int):
        """
        Random integers from low (inclusive) to high (exclusive).
        Equivalent to np.random.randint

        Args:
            low (int): Lowest (signed) integer to be drawn from the distribution
            high (int, optional): If provided, one above the largest (signed) integer
                                 to be drawn from the distribution
            size (int or tuple, optional): Output shape
            dtype (dtype, optional): Desired dtype of the result

        Returns:
            ndarray or int: Random integers of shape size
        """
        return self.rng.integers(low, high, size=size, dtype=dtype)

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        Draw samples from a uniform distribution.
        Equivalent to np.random.uniform

        Args:
            low (float, optional): Lower boundary of the output interval
            high (float, optional): Upper boundary of the output interval
            size (int or tuple, optional): Output shape

        Returns:
            ndarray or float: Drawn samples from the uniform distribution
        """
        return self.rng.uniform(low, high, size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        Draw random samples from a normal (Gaussian) distribution.
        Equivalent to np.random.normal

        Args:
            loc (float, optional): Mean of the distribution
            scale (float, optional): Standard deviation of the distribution
            size (int or tuple, optional): Output shape

        Returns:
            ndarray or float: Drawn samples from the normal distribution
        """
        return self.rng.normal(loc, scale, size=size)

    def choice(self, a, size=None, replace=True, p=None):
        """
        Generate a random sample from a given 1-D array.
        Equivalent to np.random.choice

        Args:
            a (1-D array-like or int): If an ndarray, a random sample is generated.
                                     If an int, interpreted as np.arange(a)
            size (int or tuple, optional): Output shape
            replace (bool, optional): Whether the sample is with or without replacement
            p (1-D array-like, optional): Probabilities associated with each entry in a

        Returns:
            ndarray or scalar: The generated random samples
        """
        return self.rng.choice(a, size=size, replace=replace, p=p)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """
        Draw samples from a log-normal distribution.
        Equivalent to np.random.lognormal

        Args:
            mean (float, optional): Mean value of the underlying normal distribution
            sigma (float, optional): Standard deviation of the underlying normal distribution
            size (int or tuple, optional): Output shape

        Returns:
            ndarray or float: Drawn samples from the log-normal distribution
        """
        return self.rng.lognormal(mean, sigma, size=size)