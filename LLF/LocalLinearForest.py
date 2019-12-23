from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import numpy as np


class LocalLinearForest(RandomForestRegressor):
    def __init__(self, lam=0.1, n_estimators=100, **argv):
        super().__init__(n_estimators=n_estimators, **argv)
        self.B = n_estimators # n_estimator
        self.lam = lam
        self.train_x = None
        self.train_y = None
        self.leaf_indices = None  ## Leaf indices matrix : n_samples x n_estimator
        self.n_features = None
        self.n_samples = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_trans = self.scaler.fit_transform(X)
        self.train_x = X_trans ## n_samples x n_features
        self.train_y = y ## n_samples x 1
        self.n_samples = X_trans.shape[0]
        self.n_features = X_trans.shape[1]
        RandomForestRegressor.fit(self, X_trans, y)
        self.leaf_indices = self.apply(X_trans) ## n_sample x n_estimators

    # def _set_leaves_count(self):
    #     max_indices = np.amax(self.leaf_indices)
    #     self.leaf_indices_count = np.zeros((max_indices, self.B))

    def predict_one(self, X0):
        '''
        X0 must be a array of shape 1 x n_features 
        '''
        assert X0.shape == (1, self.n_features), "The shape of X0 should be 1 x n_features"
        predict_one_leaf_indices = RandomForestRegressor.apply(self, X0) ## 1 x n_estimators
        leaf_equal_bool = np.equal(self.leaf_indices, predict_one_leaf_indices) ## n_sample x n_estimator
        leaf_count = np.sum(leaf_equal_bool, axis=0).reshape(1, -1) ## 1 x n_estimators
        alpha_weights = 1 / self.B * np.sum(leaf_equal_bool.astype(float) / leaf_count, axis=1) ## n_sample x 1
        # print(leaf_equal_bool.shape, leaf_equal_bool)
        # print(leaf_count.shape, leaf_count)
        # print(alpha_weights, np.sum(alpha_weights))
        # print(alpha_weights.shape, np.sum(alpha_weights))
        assert abs(np.sum(alpha_weights) - 1) < 0.01, "alpha weights calculation is wrong"
        
        ## A diagonal matrix
        A = np.diag(alpha_weights) ## n_sample x n_sample
        J_1d = np.ones((self.n_features + 1, 1)) 
        J_1d[0] = 0
        J = np.diag(J_1d) ## n_features + 1 x n_features + 1
        delta_m = np.ones((self.n_samples, self.n_features + 1)) ## n_sample x n_features + 1
        delta_m[:, 1:] = self.train_x - X0
        local_mu_theta = np.linalg.inv(delta_m.T @ A @ delta_m + self.lam * J ) @ delta_m.T @ A @ self.train_y
        mu = local_mu_theta[0]
        theta = local_mu_theta[1:]
        return mu, theta

    def predict(self, X):
        X_trans = self.scaler.transform(X)
        # print("made a scale transform")
        result = [] 
        for i in range(X_trans.shape[0]):
            # print(i)
            mu, theta = self.predict_one(X_trans[i, :].reshape(1, -1))
            result.append(mu)
        return np.array(result)

if __name__ == "__main__":
    f = lambda x: np.log(1 + np.exp(6 * x)) + np.random.normal(0, 2, x.shape)
    x = np.arange(-1, 1, 0.1)
    y = f(x)
    llf_1 = LocalLinearForest(lam=0.3, n_estimators=10, max_depth=3)
    llf_1.fit(x.reshape(-1, 1), y)
    print(llf_1.predict(x.reshape(-1, 1)))