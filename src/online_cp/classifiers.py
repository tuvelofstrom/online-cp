import numpy as np
import time
import warnings
from scipy.spatial.distance import pdist, cdist, squareform
from joblib import Parallel, delayed
from sklearn.neighbors import BallTree

default_epsilon = 0.1

class ConformalPredictionSet:

    def __init__(self, Gamma:np.array, epsilon):
        self.elements = Gamma
        self.epsilon = epsilon

    def __contains__(self, y):
        return y in self.elements
    
    def __len__(self):
        return self.elements.shape[0]
    
    def __repr__(self):
        return repr(self.elements)

    def __str__(self):
        return str(self.elements)
    
    def size(self):
        return self.__len__()

class ConformalClassifier:
    '''
    Parent class for classifiers
    '''

    def __init__(self, epsilon=default_epsilon):
        self.Err = 0
        # Preferred efficiency criteria (See Protocol 3.1 ALRW)
        self.OE = 0
        self.OF = 0
        self.epsilon = epsilon


    @staticmethod
    def _compute_p_value(Alpha, tau=1, score_type='nonconformity', return_string=False):
        '''
        Assumes that the (non) conformity scores are organised so that the 
        test example is the last element.
        If tau is not provided, the non-smoothed p-value is computed.
        '''
        alpha_n = Alpha[-1]
        if score_type == 'nonconformity':
            gt = np.sum(Alpha > alpha_n)
            eq = np.sum(Alpha == alpha_n)
            p = (gt + tau * eq) / Alpha.shape[0]
            string = f'({gt} + {eq}*tau)/{Alpha.shape[0]}'

        elif score_type == 'conformity':
            lt = np.sum(Alpha < alpha_n)
            eq = np.sum(Alpha == alpha_n)
            p = (lt + tau * eq) / Alpha.shape[0]
            string = f'({lt} + {eq}*tau)/{Alpha.shape[0]}'
        
        if return_string:
            return float(p), string
        else:
            return float(p)


    def _compute_Gamma(self, p_values, epsilon):
        Gamma = []
        for y in self.label_space:
            if p_values[y] > epsilon:
                Gamma.append(y)
        return ConformalPredictionSet(np.array(Gamma), epsilon)
    

    def err(self, Gamma, y):
        err = int(not(y in Gamma))
        self.Err += err
        return err
    

    def oe(self, Gamma, y):
        if y in Gamma:
            oe = len(Gamma) - 1
        else:
            oe = len(Gamma)
        self.OE += oe
        return oe
    

    def of(self, p_values, y):
        of = 0
        for label, p in p_values.items():
            if not label == y:
                of += p
        self.OF += of
        return of
    
    def learn_many(self, X, y):
        for x1, y1 in zip(X, y):
            self.learn_one(x1,y1)

    # TEST
    def process_dataset(self, X, y, epsilon=0.1, init_train=0, return_results=False):

        self.label_space = np.unique(y)

        X_train = X[:init_train]
        y_train = y[:init_train]
        X_run = X[init_train:]
        y_run = y[init_train:]

        if return_results:
            res = np.zeros(shape=(y_run.shape[0], 3))
            prediction_sets = {}

        self.learn_initial_training_set(X=X_train, y=y_train)

        time_init = time.time()
        for i, (obj, lab) in enumerate(zip(X_run, y_run)):
            
            # Make prediction
            Gamma, p_values= self.predict(obj, epsilon=epsilon, return_p_values=True) 

            # Check error
            self.err(Gamma, lab)

            # Learn the label
            self.learn_one(obj, lab)
            
            # Prefferred efficiency criteria

            # Observed excess
            self.oe(Gamma, lab)

            # Observed fuzziness
            self.of(p_values, lab)

            if return_results:
                res[i, 0] = self.OE
                res[i, 1] = self.OF
                res[i, 2] = self.Err
                prediction_sets[i] = Gamma

        time_process = time.time() - time_init

        result = {
            'Efficiency': {
                'Average error': self.Err/self.y.shape[0],
                'Average OE': self.OE/self.y.shape[0],
                'Average OF': self.OF/self.y.shape[0],
                'Time': time_process
                }
            }
        if return_results:
            result['Prediction sets'] = prediction_sets,
            result['Cummulative Err'] = res[:, 2]
            result['Cummulative OE'] = res[:, 0]
            result['Cummulative OF'] = res[:, 1]
        
        return result


class SpatialIndexManager:
    """
    Manages spatial indexing for efficient k-nearest neighbor queries.
    Uses BallTree for exact distance computations required by conformal prediction.
    """

    def __init__(self, distance_metric='euclidean'):
        self.distance_metric = distance_metric
        self.tree = None
        self.X = None
        self.y = None

    def build_index(self, X, y):
        """
        Build the spatial index from training data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        self.X = np.atleast_2d(X)
        self.y = np.asarray(y)
        self.tree = BallTree(self.X, metric=self.distance_metric)

    def add_point(self, x, y):
        """
        Add a new point to the spatial index.

        Parameters:
        x: array-like, shape (n_features,)
        y: scalar
        """
        x = np.atleast_2d(x)
        if self.X is None:
            self.X = x
            self.y = np.array([y])
        else:
            self.X = np.vstack([self.X, x])
            self.y = np.append(self.y, y)

        # Rebuild the tree with the new point
        self.tree = BallTree(self.X, metric=self.distance_metric)

    def query_k_nearest(self, x, k, exclude_self=False, self_index=None):
        """
        Query k nearest neighbors for a given point.

        Parameters:
        x: array-like, shape (n_features,)
        k: int, number of neighbors to find
        exclude_self: bool, whether to exclude the point itself if it's in the index
        self_index: int, index of the point to exclude (used when point is already in index)

        Returns:
        distances: array, shape (k,)
        indices: array, shape (k,)
        """
        x = np.atleast_2d(x)
        if self.tree is None:
            raise ValueError("Index not built. Call build_index first.")

        distances, indices = self.tree.query(x, k=k+1 if exclude_self else k)

        distances = distances[0]
        indices = indices[0]

        if exclude_self and self_index is not None:
            # Remove the self index if present
            mask = indices != self_index
            distances = distances[mask][:k]
            indices = indices[mask][:k]
        elif exclude_self:
            # If exclude_self is True but no self_index provided, assume first result is self
            distances = distances[1:k+1]
            indices = indices[1:k+1]

        return distances, indices

    def get_labels_by_indices(self, indices):
        """
        Get labels for given indices.

        Parameters:
        indices: array-like, shape (n_indices,)

        Returns:
        labels: array, shape (n_indices,)
        """
        return self.y[indices]


def validate_distance_computations(original_distances, optimized_distances, tolerance=1e-10):
    """
    Validate that optimized distance computations match original results.

    Parameters:
    original_distances: array-like, distances from original implementation
    optimized_distances: array-like, distances from optimized implementation
    tolerance: float, acceptable difference between results

    Returns:
    bool: True if distances match within tolerance
    """
    original = np.asarray(original_distances)
    optimized = np.asarray(optimized_distances)

    if original.shape != optimized.shape:
        return False

    return np.allclose(original, optimized, atol=tolerance, rtol=0)


def compute_k_nearest_distances_optimized(index_manager, x_test, y_extended, k):
    """
    Compute k nearest neighbor distances using spatial indexing (optimized version).

    Parameters:
    index_manager: SpatialIndexManager instance
    x_test: array-like, test point coordinates
    y_extended: array-like, extended labels including the test point label
    k: int, number of neighbors

    Returns:
    same_label_distances: array, mean distance to k nearest same-label neighbors
    different_label_distances: array, mean distance to k nearest different-label neighbors
    """
    n = len(y_extended)
    same_label_distances = np.full(n, np.inf)
    different_label_distances = np.full(n, np.inf)

    # Compute distances from all training points to test point
    X_train = index_manager.X
    distances_to_test, _ = index_manager.tree.query(x_test.reshape(1, -1), k=len(X_train))
    distances_to_test = distances_to_test[0]  # Distances to all training points

    # For training points (all except the last one)
    for i in range(n - 1):
        x = X_train[i]
        true_label = y_extended[i]

        # Find k+1 nearest neighbors among training points (including self)
        distances_train, indices_train = index_manager.query_k_nearest(x, k+1, exclude_self=True, self_index=i)

        # Create candidate neighbors: training neighbors + test point
        candidate_distances = np.append(distances_train, distances_to_test[i])
        candidate_labels = np.append(y_extended[indices_train], y_extended[-1])  # Test point label

        # Separate same and different label candidates
        same_mask = candidate_labels == true_label
        different_mask = candidate_labels != true_label

        if np.any(same_mask):
            same_distances = candidate_distances[same_mask]
            same_label_distances[i] = np.mean(same_distances[:k])

        if np.any(different_mask):
            different_distances = candidate_distances[different_mask]
            different_label_distances[i] = np.mean(different_distances[:k])

    # For test point (last element)
    test_label = y_extended[-1]
    distances, indices = index_manager.query_k_nearest(x_test, k)

    # Get labels of neighbors
    neighbor_labels = y_extended[indices]

    # Separate same and different label neighbors
    same_mask = neighbor_labels == test_label
    different_mask = neighbor_labels != test_label

    if np.any(same_mask):
        same_distances = distances[same_mask][:k]
        same_label_distances[-1] = np.mean(same_distances)

    if np.any(different_mask):
        different_distances = distances[different_mask][:k]
        different_label_distances[-1] = np.mean(different_distances)

    return same_label_distances, different_label_distances


class ConformalNearestNeighboursClassifier(ConformalClassifier):
    """
    Classifier using nearest neighbours as the nonconformity measure.

    >>> cp = ConformalNearestNeighboursClassifier(k=1, rnd_state=1337, epsilon=0.1)
    >>> Gamma, p_values = cp.predict(3, return_p_values=True)
    >>> Gamma # predict both labels, as this is the first
    array([-1,  1])
    >>> [p_values[i] for i in [-1, 1]]
    [0.8781019003471183, 0.8781019003471183]

    >>> cp.learn_one(np.int64(3), 1)

    >>> Gamma, p_values = cp.predict(-2, return_p_values=True)
    >>> Gamma # predict both labels, as this is the first
    array([-1,  1])
    >>> [p_values[i] for i in [-1, 1]]
    [0.18552796163759344, 0.18552796163759344]
    """
    # TODO: implement: cp.learn_several([[3,1],[4,7],[5,2]], [1, -1, 1])

    # TODO Write tests

    def __init__(self, k=1, label_space=np.array([-1, 1]), distance='euclidean', distance_func=None, verbose=0, rnd_state=None, n_jobs=None, epsilon=default_epsilon, use_optimization=False):
        super().__init__(epsilon=epsilon)
        self.label_space = label_space

        self.k = k

        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = 'custom'

        self.y = np.empty(0)
        self.X = None
        self.D = None

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.n_jobs = n_jobs

        # Optimization components
        self.use_optimization = use_optimization
        if use_optimization:
            self.spatial_index = SpatialIndexManager(distance_metric=distance)
            # Pre-sorted distance lists for optimization
            self.sorted_same_distances = []  # List of sorted distances to same-label points
            self.sorted_diff_distances = []  # List of sorted distances to different-label points
        else:
            self.spatial_index = None
            self.sorted_same_distances = None
            self.sorted_diff_distances = None
    
    def reset(self):

        self.__init__(self.k, self.label_space, self.distance, self.distance_func, self.verbose, self.rnd_gen, self.n_jobs, self.epsilon, self.use_optimization)

    def _standard_distance_func(self, X, y=None):
        '''
        By default we use scipy to compute distances
        '''
        X = np.atleast_2d(X)
        if y is None:
            dists = squareform(pdist(X, metric=self.distance))
        else:
            y = np.atleast_2d(y)
            dists = cdist(X, y, metric=self.distance)
        return dists
    

    def learn_initial_training_set(self, X, y):
        if X.shape[0] > 0:
            self.X = X
            self.y = y
            self.D = self.distance_func(X)

            # Build spatial index if optimization is enabled
            if self.use_optimization and self.spatial_index is not None:
                self.spatial_index.build_index(X, y)
                # Build sorted distance lists for optimization
                self._build_sorted_distance_lists()

    def _build_sorted_distance_lists(self):
        """Build sorted distance lists for optimization."""
        if not self.use_optimization:
            return

        n = len(self.y)
        self.sorted_same_distances = []
        self.sorted_diff_distances = []

        for i in range(n):
            same_label_mask = (self.y == self.y[i])
            diff_label_mask = (self.y != self.y[i])
            same_label_mask[i] = False  # Exclude self

            if np.any(same_label_mask):
                same_distances = np.sort(self.D[i, same_label_mask])
                self.sorted_same_distances.append(same_distances)
            else:
                self.sorted_same_distances.append(np.array([]))

            if np.any(diff_label_mask):
                diff_distances = np.sort(self.D[i, diff_label_mask])
                self.sorted_diff_distances.append(diff_distances)
            else:
                self.sorted_diff_distances.append(np.array([]))


    @staticmethod
    def update_distance_matrix(D, d):
        return np.block([[D, d], [d.T, np.array([0])]])
    

    def _find_nearest_distances(self, D, y):
        n = D.shape[0]
        
        # Initialize arrays to store the results
        same_label_distances = np.full(n, np.inf)
        different_label_distances = np.full(n, np.inf)

        for i in range(n):
            # Create a mask for the same and different labels
            same_label_mask = (y == y[i])
            different_label_mask = (y != y[i])

            # Ignore the distance to itself by setting it to np.inf
            same_label_mask[i] = False

            # Extract distances for the same label
            if np.any(same_label_mask):
                same_label_distances[i] = np.sort(D[i, same_label_mask])[:self.k].mean()
            
            # Extract distances for the different label
            if np.any(different_label_mask):
                different_label_distances[i] = np.sort(D[i, different_label_mask])[:self.k].mean()

        return same_label_distances, different_label_distances
    

    def learn_one(self, x, y, D=None):
        # Learn label y
        self.y = np.append(self.y, y)

        # Learn object
        if self.X is None:
            self.X = x.reshape(1,-1)
            self.D = self.distance_func(self.X)
        else:
            if D is None:
                d = self.distance_func(self.X, x)
                D = self.update_distance_matrix(self.D, d)
            self.D = D
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)

        # Add to spatial index if optimization is enabled
        if self.use_optimization and self.spatial_index is not None:
            self.spatial_index.add_point(x, y)
            # Update sorted distance lists
            self._update_sorted_distance_lists(x, y)

    def _update_sorted_distance_lists(self, x, y):
        """Update sorted distance lists when adding a new point."""
        if not self.use_optimization:
            return

        n = len(self.y)  # New total number of points
        new_idx = n - 1  # Index of the new point

        # Add empty lists for the new point
        self.sorted_same_distances.append(np.array([]))
        self.sorted_diff_distances.append(np.array([]))

        # Update lists for all existing points
        for i in range(n - 1):
            dist_to_new = self.D[i, new_idx]

            # Update same-label distances
            if self.y[i] == y:
                # Insert distance to new point into sorted list
                self.sorted_same_distances[i] = np.sort(np.append(self.sorted_same_distances[i], dist_to_new))
            else:
                # Update different-label distances
                self.sorted_diff_distances[i] = np.sort(np.append(self.sorted_diff_distances[i], dist_to_new))

        # Build lists for the new point
        same_label_mask = (self.y[:-1] == y)  # Exclude self
        diff_label_mask = (self.y[:-1] != y)

        if np.any(same_label_mask):
            same_distances = self.D[new_idx, :-1][same_label_mask]
            self.sorted_same_distances[new_idx] = np.sort(same_distances)

        if np.any(diff_label_mask):
            diff_distances = self.D[new_idx, :-1][diff_label_mask]
            self.sorted_diff_distances[new_idx] = np.sort(diff_distances)

    def _find_nearest_distances_optimized(self, x, y_extended):
        """
        Optimized version of _find_nearest_distances using pre-sorted lists.
        """
        n = len(y_extended)
        same_label_distances = np.full(n, np.inf)
        different_label_distances = np.full(n, np.inf)

        # Compute distances from test point (last in extended) to all training points
        test_point_idx = n - 1
        x_test = x
        distances_to_training = self.distance_func(self.X, x_test).flatten()

        for i in range(n):
            if i < n - 1:  # Training point
                true_label = y_extended[i]
                dist_to_test = distances_to_training[i]

                # Get sorted distances to same-label training points
                same_training_distances = self.sorted_same_distances[i]

                # Get sorted distances to different-label training points
                diff_training_distances = self.sorted_diff_distances[i]

                # Check if test point has same label
                test_same_label = (y_extended[test_point_idx] == true_label)

                # Combine training distances with test point distance if applicable
                if test_same_label:
                    all_same_distances = np.append(same_training_distances, dist_to_test)
                else:
                    all_same_distances = same_training_distances

                if not test_same_label:
                    all_diff_distances = np.append(diff_training_distances, dist_to_test)
                else:
                    all_diff_distances = diff_training_distances

                # Take mean of k smallest
                if len(all_same_distances) > 0:
                    same_label_distances[i] = np.mean(np.sort(all_same_distances)[:self.k])
                if len(all_diff_distances) > 0:
                    different_label_distances[i] = np.mean(np.sort(all_diff_distances)[:self.k])

            else:  # Test point
                true_label = y_extended[i]

                # For test point, only consider training points
                same_mask = (y_extended[:n-1] == true_label)
                diff_mask = (y_extended[:n-1] != true_label)

                if np.any(same_mask):
                    same_distances = distances_to_training[same_mask]
                    same_label_distances[i] = np.mean(np.sort(same_distances)[:self.k])

                if np.any(diff_mask):
                    diff_distances = distances_to_training[diff_mask]
                    different_label_distances[i] = np.mean(np.sort(diff_distances)[:self.k])

        return same_label_distances, different_label_distances


    def _predict_optimized(self, x, epsilon=None, return_p_values=False, return_update=False, verbose=0):
        """
        Optimized prediction using pre-sorted distance lists.
        """
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if self.y.shape[0] >= 1:
            tic = time.time()
            # For optimized version, we don't need to update distance matrix
            # Instead, we'll use pre-sorted distance lists
            time_update_D = time.time() - tic

            tic = time.time()
            if self.n_jobs is not None:
                def process_label(label):
                    y_extended = np.append(self.y, label)
                    same_label_distances, different_label_distances = self._find_nearest_distances_optimized(x, y_extended)

                    Alpha = same_label_distances / different_label_distances
                    if verbose > 10:
                        print(f'Nonconformity scores for hypothesis y={label}: {Alpha}')
                        _, string = self._compute_p_value(Alpha, tau, 'nonconformity', return_string=True)
                        print(f'p-value for hypothesis y={label}: {string}')

                    return label, self._compute_p_value(Alpha, tau, 'nonconformity')

                results = Parallel(n_jobs=self.n_jobs)(delayed(process_label)(label) for label in self.label_space)
                p_values = dict(results)
            else:
                for label in self.label_space:
                    y_extended = np.append(self.y, label)
                    same_label_distances, different_label_distances = self._find_nearest_distances_optimized(x, y_extended)

                    Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)

                    if verbose > 10:
                        print(f'Nonconformity scores for hypothesis y={label}: {Alpha}')
                        p_values[label], string = self._compute_p_value(Alpha, tau, 'nonconformity', return_string=True)
                        print(f'p-value for hypothesis y={label}: {string}')

                    p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
            time_compute_p_values = time.time() - tic

            tic = time.time()
            Gamma = self._compute_Gamma(p_values, epsilon)
            time_Gamma = time.time()- tic

            self.time_dict = {
                'Update distance matrix': time_update_D,
                'Compute p-values': time_compute_p_values,
                'Compute Gamma': time_Gamma
            }

        else:
            for label in self.label_space:
                Alpha = np.array([np.inf])
                if verbose > 10:
                    print(f'Nonconformity scores for hypothesis y={label}: {Alpha}')
                    p_values[label], string = self._compute_p_value(Alpha, tau, 'nonconformity', return_string=True)
                    print(f'p-value for hypothesis y={label}: {string}')
                p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
            Gamma = self._compute_Gamma(p_values, epsilon)
            self.time_dict = {}

        if return_update:
            if return_p_values:
                return Gamma, p_values, None  # No D matrix in optimized version
            else:
                return Gamma, None
        else:
            if return_p_values:
                return Gamma, p_values
            else:
                return Gamma

    def _predict_original(self, x, epsilon=None, return_p_values=False, return_update=False, verbose=0):
        """
        Original prediction implementation using full distance matrix.
        """
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if self.y.shape[0] >= 1: 
            tic = time.time()
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            time_update_D = time.time() - tic
            
            tic = time.time()
            if self.n_jobs is not None:
                def process_label(label):
                    y = np.append(self.y, label)
                    same_label_distances, different_label_distances = self._find_nearest_distances(D, y)

                    Alpha = same_label_distances / different_label_distances
                    if verbose > 10:
                        print(f'Nonconformity scores for hypothesis y={label}: {Alpha}')
                        _, string = self._compute_p_value(Alpha, tau, 'nonconformity', return_string=True)
                        print(f'p-value for hypothesis y={label}: {string}')

                    return label, self._compute_p_value(Alpha, tau, 'nonconformity')

                results = Parallel(n_jobs=self.n_jobs)(delayed(process_label)(label) for label in self.label_space)
                p_values = dict(results)
            else:
                for label in self.label_space:
                    y = np.append(self.y, label)
                    
                    same_label_distances, different_label_distances = self._find_nearest_distances(D, y)              

                    Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)

                    if verbose > 10:
                        print(f'Nonconformity scores for hypothesis y={label}: {Alpha}')
                        p_values[label], string = self._compute_p_value(Alpha, tau, 'nonconformity', return_string=True)
                        print(f'p-value for hypothesis y={label}: {string}')

                    p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
            time_compute_p_values = time.time() - tic

            tic = time.time()
            Gamma = self._compute_Gamma(p_values, epsilon)
            time_Gamma = time.time()- tic

            self.time_dict = {
                'Update distance matrix': time_update_D,
                'Compute p-values': time_compute_p_values,
                'Compute Gamma': time_Gamma
            }
            
        else:
            for label in self.label_space:
                Alpha = np.array([np.inf])
                if verbose > 10:
                    print(f'Nonconformity scores for hypothesis y={label}: {Alpha}')
                    p_values[label], string = self._compute_p_value(Alpha, tau, 'nonconformity', return_string=True)
                    print(f'p-value for hypothesis y={label}: {string}')
                p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
            Gamma = self._compute_Gamma(p_values, epsilon)
            D = None
            self.time_dict = {}

        if return_update: 
            if return_p_values:
                return Gamma, p_values, D
            else:
                return Gamma, D
        else:
            if return_p_values:
                return Gamma, p_values
            else:
                return Gamma

    def predict(self, x, epsilon=None, return_p_values=False, return_update=False, verbose=0):
        """
        Predict conformal prediction set for input x.
        
        Dispatches to optimized or original implementation based on use_optimization flag.
        """
        if self.use_optimization:
            return self._predict_optimized(x, epsilon, return_p_values, return_update, verbose)
        else:
            return self._predict_original(x, epsilon, return_p_values, return_update, verbose)


class ConformalClassifierWrapper(ConformalClassifier):
    '''
    The following scikit-learn classifiers should in priciple be compatible (they have a predict_proba method):
    AdaBoostClassifier
    BaggingClassifier
    BernoulliNB
    CalibratedClassifierCV
    CategoricalNB
    ComplementNB
    DecisionTreeClassifier
    DummyClassifier
    ExtraTreeClassifier
    ExtraTreesClassifier
    GaussianNB
    GaussianProcessClassifier
    GradientBoostingClassifier
    HistGradientBoostingClassifier
    KNeighborsClassifier
    LabelPropagation
    LabelSpreading
    LinearDiscriminantAnalysis
    LogisticRegression
    LogisticRegressionCV
    MLPClassifier
    MultinomialNB
    NearestCentroid
    QuadraticDiscriminantAnalysis
    RadiusNeighborsClassifier
    RandomForestClassifier
    '''

    def __init__(self, learner, label_space=np.array([-1, 1]), epsilon=default_epsilon, verbose=0, rnd_state=None, n_jobs=None):
        super().__init__(epsilon)

        assert hasattr(learner, 'predict_proba')

        self.learner = learner

        self.label_space = label_space

        self.y = np.empty(0)
        self.X = None

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.n_jobs = n_jobs

    def learn_one(self, x, y, D=None):
        # Learn label y
        self.y = np.append(self.y, y)
        # Learn object
        if self.X is None:
            self.X = x.reshape(1,-1)
        else:
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)

    def predict(self, x, epsilon=None, return_p_values=False, verbose=0):
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon       

        # TODO: Fix parallellisation

        # TODO: Some models have minimum requirements for the training set.
        # if self.y.shape[0] >= 1:
        try:
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            # Label loop
            for y in self.label_space:
                Y = np.append(self.y, y)
                self.learner.fit(X, Y)

                Prob = self.learner.predict_proba(X)
                if Prob.shape[1] < self.label_space.size:
                    zeros_to_add = self.label_space.size - Prob.shape[1]
                    Prob = np.hstack([Prob, np.zeros((Prob.shape[0], zeros_to_add))])

                Alpha = Prob[np.arange(len(Y)), Y.astype('int')] 
                p_values[y] = self._compute_p_value(Alpha, tau, 'conformity')
            Gamma = self._compute_Gamma(p_values, epsilon)
        # else:
        except ValueError:
            for label in self.label_space:
                Alpha = np.array([np.inf])
                p_values[label] = self._compute_p_value(Alpha, tau, 'conformity')
            Gamma = self._compute_Gamma(p_values, epsilon)
        
        if return_p_values:
            return Gamma, p_values
        else:
            return Gamma

class ConformalSupportVectorMachine(ConformalClassifier):
    # The Lagrange multipliers can be used as nonconformity measure. 
    # TODO: Figure out how to use sklearn's SVM
    # TODO: Check the caveat in ALRW
    # TODO: Implement
    def __init__(self, epsilon=default_epsilon):
        super().__init__(epsilon)
    
        
if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
