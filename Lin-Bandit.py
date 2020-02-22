
import itertools
import numpy as np
import time

import os

from scipy.stats import norm
import scipy

class LinBandit:
    """Linear bandit."""

    def __init__(self, X, theta, noise="normal", sigma=0.5, seed=None):
        self.X = np.copy(X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.theta = np.copy(theta)
        self.noise = noise
        if self.noise == "normal":
            self.sigma = sigma

        self.mu = self.X.dot(self.theta)
        self.best_arm = np.argmax(self.mu)

        self.seed = seed
        self.random = np.random.RandomState(seed)
    
    def reset_random(self):
        self.random = np.random.RandomState(self.seed)
        
    def randomize(self):
        # generate random rewards
        if self.noise == "normal":
            self.rt = self.mu + self.sigma * self.random.randn(self.K)
        elif self.noise == "bernoulli":
            self.rt = (self.random.rand(self.K) < self.mu).astype(float)
        elif self.noise == "beta":
            self.rt = self.random.beta(4 * self.mu, 4 * (1 - self.mu))

    def reward(self, arm):
        # instantaneous reward of the arm
        return self.rt[arm]

    def regret(self, arm):
        # instantaneous regret of the arm
        return self.rt[self.best_arm] - self.rt[arm]

    def pregret(self, arm):
        # expected regret of the arm
        return self.mu[self.best_arm] - self.mu[arm]

    def print(self):
        if self.noise == "normal":
            return "Linear bandit: %d dimensions, %d arms" % (self.d, self.K)
        elif self.noise == "bernoulli":
            return "Bernoulli linear bandit: %d dimensions, %d arms" % (self.d, self.K)
        elif self.noise == "beta":
            return "Beta linear bandit: %d dimensions, %d arms" % (self.d, self.K)

def evaluate_one(Alg, params, env, n, period_size):
    """One run of a bandit algorithm."""
    alg = Alg(env, n, params)

    regret = np.zeros(n // period_size)
    for t in range(n):

        # generate state
        env.randomize()

        # take action
        arm = alg.get_arm(t)

        # update model and regret
        alg.update(t, arm, env.reward(arm))
        regret_at_t = env.regret(arm)        
        regret[t // period_size] += regret_at_t

        # print('Round: ', t, ' Regret: ', np.cumsum(regret))        

    return regret, alg

def evaluate(Alg, params, envs, n=1000, period_size=1, printout=True):
    """Multiple runs of a bandit algorithm."""
    if printout:
        print("Evaluating %s" % Alg.print(), end="")
    start = time.time()

    num_exps = len(envs)
    regret = np.zeros((n // period_size, num_exps))
    alg = num_exps * [None]

    dots = np.linspace(0, num_exps - 1, 100).astype(int)

    for i, env in enumerate(envs):

        print('Env number:', i)

        env.reset_random()
        
        output = evaluate_one(Alg, params, env, n, period_size)
        regret[:, i] = output[0]
        alg[i] = output[1]

        if i in dots and printout:
            print(".", end="")

    if printout:
        print(" %.1f seconds" % (time.time() - start))
        
        total_regret = regret.sum(axis=0)
        print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f)" %
            (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
            np.median(total_regret), total_regret.max()))

    return regret, alg

class LinBanditAlg:
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.n = n
        self.sigma0 = 1.0
        self.sigma = 0.5
        self.crs = 1.0 # confidence region scaling

        for attr, val in params.items():
            setattr(self, attr, val)

        # sufficient statistics
        self.Gram = 1e-4 * np.eye(self.d) / np.square(self.sigma0)
        self.B = np.zeros(self.d)

    def update(self, t, arm, r):
        x = self.X[arm, :]
        self.Gram += np.outer(x, x) / np.square(self.sigma)
        self.B += x * r / np.square(self.sigma)

class LinUCB(LinBanditAlg):
    def __init__(self, env, n, params):
        LinBanditAlg.__init__(self, env, n, params)
        self.cew = self.crs * self.confidence_ellipsoid_width(n)

    def confidence_ellipsoid_width(self, t):
        # Theorem 2 in Abassi-Yadkori (2011)
        # Improved Algorithms for Linear Stochastic Bandits
        delta = 1 / self.n
        L = np.amax(np.linalg.norm(self.X, axis = 1))
        Lambda = 1 / np.square(self.sigma0)
        R = self.sigma
        S = np.sqrt(self.d)
        width = np.sqrt(Lambda) * S + R * np.sqrt(self.d * np.log((1 + t * np.square(L) / Lambda) / delta))
        return width

    def get_arm(self, t):
        Gram_inv = np.linalg.inv(self.Gram)
        theta = Gram_inv.dot(self.B)
        # UCBs
        self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "LinUCB"

def randomize_confidence(M = 20,  pdist = "Normal", pnormal_std = "0.125", pfixed = None, is_optimistic = True, is_coupled = True, K=None):
    if pdist == 'Dirac':
        # for greedy
        if is_coupled == True:
            a = 0
        else:
            a = np.zeros(self.K)
    else:
        if is_optimistic == True:
            lb = 0
        else:
            lb = -1
            
        ub = +1
        
        x = np.linspace(lb, ub, M)

        if pdist == 'Normal':
            probs = norm.pdf(x, loc = 0, scale = pnormal_std)
            
        elif pdist == 'Uniform':
            probs = uniform.pdf(x, loc = -lb, scale = ub - lb)
            
        elif pdist == 'Fixed':
            probs = pfixed


        probs[-1] = 1e-6 # constant probability of choosing the last confidence interval            

        probs = probs / np.sum(probs)      
                
        if is_coupled == True:
            m = np.random.choice(M, p=probs)
        else:
            m = np.random.choice(M, size=K, p=probs)
        
        a = (lb + m * (ub - lb) / (M-1)) 

    return a

class RandLinUCB(LinUCB):
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.n = n
        self.sigma0 = 1.0
        self.sigma = 0.5
        self.crs = 1.0 # confidence region scaling

        self.pnormal_std, self.pfixed = None, None
        for attr, val in params.items():
            setattr(self, attr, val)

        # sufficient statistics
        self.Gram = 1e-4 * np.eye(self.d) / np.square(self.sigma0)
        self.B = np.zeros(self.d)    

        self.cew = self.crs * self.confidence_ellipsoid_width(n)   

    def get_arm(self, t):
        Gram_inv = np.linalg.inv(self.Gram)
        theta = Gram_inv.dot(self.B)
    
        # randomize the UCB interval    
        a = randomize_confidence(self.M, self.pdist, self.pnormal_std, self.pfixed, self.is_optimistic, self.is_coupled)    
   
        # UCBs    
        self.mu = self.X.dot(theta) + a * self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))

        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "RandLinUCB"

class LinGreedy(LinBanditAlg):
    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if np.random.rand() < 0.05 * np.sqrt(self.n / (t + 1)) / 2:
          self.mu[np.random.randint(self.K)] = np.Inf
        else:
          theta = np.linalg.solve(self.Gram, self.B)
          self.mu = self.X.dot(theta)

        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "Lin e-greedy"

class LinTS(LinBanditAlg):
    def __init__(self, env, n, params):
                
        self.inflated, self.delta, self.epsilon = False, None, None

        super().__init__(env, n, params)

        if self.inflated:
            assert self.epsilon != None and self.delta != None
    
    def get_arm(self, t):

        Gram_inv = np.linalg.inv(self.Gram)        

        if self.inflated:            
            # params for theoretically inflated LinTS
            # see http://proceedings.mlr.press/v28/agrawal13.pdf Alg.1 + Def. 3            
            inflation = (24 / self.epsilon) * self.d * np.log(1/self.delta)

        else:
            inflation = 1.0
        
        thetabar = Gram_inv.dot(self.B)

        # posterior sampling
        # theta = np.random.multivariate_normal(thetabar, inflation*Gram_inv)
        z = np.random.multivariate_normal(np.zeros(d), np.eye(d))
        # print(z)        
        theta = thetabar + np.dot( scipy.linalg.sqrtm(inflation * Gram_inv), z)
        self.mu = self.X.dot(theta)
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "LinTS"

class LinPHE:
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.a = 2

        for attr, val in params.items():
            setattr(self, attr, val)

        # sufficient statistics
        self.pulls = np.zeros(self.K, dtype = int) # number of pulls
        self.reward = np.zeros(self.K) # cumulative reward
        self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
        self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
        for k in range(self.K):
            self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

    def update(self, t, arm, r):
        self.pulls[arm] += 1
        self.reward[arm] += r

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t < self.d:
            self.mu[t] = np.Inf
        else:
            # history perturbation
            pseudo_pulls = np.ceil(self.a * self.pulls).astype(int)
            pseudo_reward = np.random.binomial(pseudo_pulls, 0.5)
            Gram = np.tensordot(self.pulls + pseudo_pulls, self.X2, axes = ([0], [0]))
            B = self.X.T.dot(self.reward + pseudo_reward)

            reg = 1e-4 * np.eye(self.d)
            theta = np.linalg.solve(Gram + reg, B)
            self.mu = self.X.dot(theta) + self.tiebreak

        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "LinPHE"

if __name__ == "__main__":
    base_dir = os.path.join(".", "Results", "Lin")

    num_runs = 50
    n = 20000
    K = 100

    algorithms = [
        (RandLinUCB, {"M": 20, "pdist": "Normal", "pnormal_std":0.125, "is_optimistic":True, "is_coupled":True}, "RandLinUCB"),
        (LinUCB, {}, "LinUCB"),
        (LinTS, {}, "LinTS"),
        (LinTS, {"inflated": True, "epsilon": 1/np.log(n), "delta": 0.1}, "LinTS-Inflated"),
        (LinGreedy, {}, "e-greedy"),
    #   (LinPHE, {"a": 2}, "LinPHE (a = 2)"),
    #   (LinPHE, {"a": 1}, "LinPHE (a = 1)"),
        (LinPHE, {"a": 0.5}, "LinPHE (a = 0.5)")
    ]
   

    environments = [
        (LinBandit, {"noise": "bernoulli", "sigma": 0.5}, 5, "Bernoulli (d=5)"),
        (LinBandit, {"noise": "bernoulli", "sigma": 0.5}, 10, "Bernoulli (d=10)"),        
        (LinBandit, {"noise": "bernoulli", "sigma": 0.5}, 20, "Bernoulli (d=20)"),
        # (LinBandit, {"noise": "bernoulli", "sigma": 0.5}, 100, "Bernoulli (d=100)")
    ]

    for env_def in environments:
        env_class, env_params, d, env_name = env_def[0], env_def[1], env_def[2], env_def[-1]
        print("================== running environment", env_name, "==================")
        
        envs = []
        for run in range(num_runs):     

            np.random.seed(run)       

            # standard d-dimensional basis (with a bias term)
            basis = np.eye(d)
            basis[:, -1] = 1

            # arm features in a unit (d - 2)-sphere
            X = np.random.randn(K, d - 1)
            X /= np.sqrt(np.square(X).sum(axis=1))[:, np.newaxis]
            X = np.hstack((X, np.ones((K, 1))))  # bias term
            X[: basis.shape[0], :] = basis

            # parameter vector in a (d - 2)-sphere with radius 0.5
            theta = np.random.randn(d - 1)
            theta *= 0.5 / np.sqrt(np.square(theta).sum())
            theta = np.append(theta, [0.5])

            # create environment
            envs.append(env_class(X, theta, seed=run, **env_params))
            print("%3d: %.2f %.2f | " % (envs[-1].best_arm,
                envs[-1].mu.min(), envs[-1].mu.max()), end="")
            if (run + 1) % 10 == 0:
                print()
        
        res_dir = os.path.join(base_dir, env_name)
        os.makedirs(res_dir, exist_ok=True)
        
        for alg_def in algorithms:
            alg_class, alg_params,alg_name = alg_def[0], alg_def[1], alg_def[-1]            
            
            fname = os.path.join(res_dir, alg_name)        
            if os.path.exists(fname):
                print('File exists. Will load saved file. Moving on to the next algorithm')
            else:
                regret, _ = evaluate(alg_class, alg_params, envs, n)                
                cum_regret = regret.cumsum(axis=0)                           
                np.savetxt(fname, cum_regret, delimiter=",")
