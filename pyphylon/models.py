'''
Handling reduced-dimension models of P
'''

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for modeling techniques in pyphylon.
    This class outlines shared behavior and interface for models like NMF, HDBSCAN,
    UMAP, and MCA.
    """
    
    def __init__(self, n_components=None, random_state=None, **kwargs):
        """
        Initialize the model with common parameters.

        Parameters:
        - n_components: int, optional (default=None)
            Number of components to use, if applicable.
        - random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.
        - kwargs: Additional keyword arguments.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.params = kwargs

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the model to data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data to fit.
        - y: array-like, shape (n_samples,) or (n_samples, n_outputs), optional
            Target values (class labels in classification, real numbers in regression).
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Transform the data using the model.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Data to transform.
        """
        pass

    @abstractmethod
    def fit_transform(self, X, y=None):
        """
        Fit the model to data and transform it in one step.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data to fit and transform.
        - y: array-like, shape (n_samples,) or (n_samples, n_outputs), optional
            Target values.
        """
        pass

    @abstractmethod
    def score(self, X, y=None):
        """
        Returns a score of the model given data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Test samples.
        - y: array-like, shape (n_samples,) or (n_samples, n_outputs), optional
            True labels for X.
        """
        pass
