
import itertools
import numpy as np
import time

import os

from scipy.stats import norm
import scipy

class LogBandit(object):
    """Logistic bandit."""

    def __init__(self, X, theta, seed = None):
        self.X = np.copy(X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.theta = np.copy(theta)

        self.mu = 1 / (1 + np.exp(- self.X.dot(self.theta)))
        self.best_arm = np.argmax(self.mu)
        
        self.seed = seed
        self.random = np.random.RandomState(seed)
    
    def reset_random(self):
        self.random = np.random.RandomState(self.seed)
        
    def randomize(self):
        # generate random rewards
        self.rt = (np.random.rand(self.K) < self.mu).astype(float)

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
        return "Logistic bandit: %d dimensions, %d arms" % (self.d, self.K)

    @staticmethod
    def ball_env(d=3, K=10, num_env=100):
        """Arm features and theta are generated randomly in a ball."""
        env = []
        for env_id in range(num_env):
            # standard d-dimensional basis (with a bias term)
            basis = np.eye(d)
            basis[:, -1] = 1

            # arm features in a unit (d - 2)-sphere
            X = np.random.randn(K, d - 1)
            X /= np.sqrt(np.square(X).sum(axis=1))[:, np.newaxis]
            X = np.hstack((X, np.ones((K, 1))))  # bias term
            X[: basis.shape[0], :] = basis

            # parameter vector in a (d - 2)-sphere with radius 1.5
            theta = np.random.randn(d - 1)
            theta *= 1.5 / np.sqrt(np.square(theta).sum())
            theta = np.append(theta, [0])

            # create environment
            env.append(LogBandit(X, theta))
            print("%3d: %.2f %.2f | " % (env[-1].best_arm, env[-1].mu.min(), env[-1].mu.max()), end="")
            if (env_id + 1) % 10 == 0:
                print()
        return env

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

class LogBanditAlg:
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.n = n
        self.sigma0 = 1.0
        self.a = 1.0
        self.crs = 1.0 # confidence region scaling

        for attr, val in params.items():
            setattr(self, attr, val)

        # sufficient statistics
        self.pos = np.zeros(self.K, dtype = int) # number of positive observations
        self.neg = np.zeros(self.K, dtype = int) # number of negative observations
        self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
        for k in range(self.K):
            self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

    def update(self, t, arm, r):
        self.pos[arm] += r
        self.neg[arm] += 1 - r

    def sigmoid(self, x):
        return 1 / (1 + np.exp(- x))

    def solve(self):
        # iterative reweighted least squares for Bayesian logistic regression
        # Sections 4.3.3 and 4.5.1 in Bishop (2006)
        # Pattern Recognition and Machine Learning
        theta = np.zeros(self.d)
        num_iter = 0
        while num_iter < 100:
            theta_old = np.copy(theta)

            Xtheta = self.X.dot(theta)
            R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
            pulls = self.pos + self.neg
            Gram = np.tensordot(R * pulls, self.X2, axes = ([0], [0])) +   1e-4 * np.eye(self.d) / np.square(self.sigma0)
            Rz = R * pulls * Xtheta -         self.pos * (self.sigmoid(Xtheta) - 1) -         self.neg * (self.sigmoid(Xtheta) - 0)
            theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

            if np.linalg.norm(theta - theta_old) < 1e-3:
                break;
            num_iter += 1

        return theta, Gram

class LogUCB(LogBanditAlg):
    def __init__(self, env, n, params):
        super().__init__(env, n, params)
        self.cew = self.crs * self.confidence_ellipsoid_width(n)

    def confidence_ellipsoid_width(self, t):
        # Section 4.1 in Filippi (2010)
        # Parametric Bandits: The Generalized Linear Case
        delta = 1 / self.n
        c_m = np.amax(np.linalg.norm(self.X, axis = 1))
        c_mu = 0.25 # minimum derivative of the mean function
        k_mu = 0.25
        kappa = np.sqrt(3 + 2 * np.log(1 + 2 * np.square(c_m / self.sigma0)))
        R_max = 1.0
        width = (2 * k_mu * kappa * R_max / c_mu) * np.sqrt(2 * self.d * np.log(t) * np.log(2 * self.d * self.n / delta))
        return width

    def get_arm(self, t):
        pulls = self.pos + self.neg
        Gram = np.tensordot(pulls, self.X2, axes = ([0], [0])) +  1e-4 * np.eye(self.d) / np.square(self.sigma0)
        Gram_inv = np.linalg.inv(Gram)
        theta, _ = self.solve()

        # UCBs
        self.mu = self.sigmoid(self.X.dot(theta)) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))

        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "GLM-UCB (log)"

class UCBLog(LogBanditAlg):
    def __init__(self, env, n, params):
        super().__init__(env, n, params)
        self.cew = self.crs * self.confidence_ellipsoid_width(n)

    def confidence_ellipsoid_width(self, t):
        # Theorem 2 in Li (2017)
        # Provably Optimal Algorithms for Generalized Linear Contextual Bandits
        delta = 1 / self.n
        sigma = 0.5
        kappa = 0.25 # minimum derivative of a constrained mean function
        width = (sigma / kappa) * np.sqrt((self.d / 2) * np.log(1 + 2 * self.n / self.d) + np.log(1 / delta))
        return width

    def get_arm(self, t):
        pulls = self.pos + self.neg
        Gram = np.tensordot(pulls, self.X2, axes = ([0], [0])) +  1e-4 * np.eye(self.d) / np.square(self.sigma0)
        Gram_inv = np.linalg.inv(Gram)
        theta, _ = self.solve()

        # UCBs
        self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))

        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "UCB-GLM (log)"

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
            m = np.random.choice(M, size=self.K, p=probs)
        
        a = (lb + m * (ub - lb) / (M-1)) 

    return a

class RandLogUCB(LogUCB):
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.n = n
        self.sigma0 = 1.0
        self.a = 1.0
        self.crs = 1.0 # confidence region scaling
    
        self.pnormal_std, self.pfixed = None, None
        for attr, val in params.items():
            setattr(self, attr, val)

        # sufficient statistics
        self.pos = np.zeros(self.K, dtype = int) # number of positive observations
        self.neg = np.zeros(self.K, dtype = int) # number of negative observations
        self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
        for k in range(self.K):
            self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])
    
        self.cew = self.crs * self.confidence_ellipsoid_width(n)

    def get_arm(self, t):
        pulls = self.pos + self.neg
        Gram = np.tensordot(pulls, self.X2, axes = ([0], [0])) +  1e-4 * np.eye(self.d) / np.square(self.sigma0)
        Gram_inv = np.linalg.inv(Gram)
        theta, _ = self.solve()

        # UCBs
        a = randomize_confidence(self.M, self.pdist, self.pnormal_std, self.pfixed, self.is_optimistic, self.is_coupled)            
        
        self.mu = self.sigmoid(self.X.dot(theta)) + a * self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))
        
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "RandLogUCB"

class RandUCBLog(UCBLog):
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.n = n
        self.sigma0 = 1.0
        self.a = 1.0
        self.crs = 1.0 # confidence region scaling
        
        self.pnormal_std, self.pfixed = None, None
        for attr, val in params.items():
          setattr(self, attr, val)

        # sufficient statistics
        self.pos = np.zeros(self.K, dtype = int) # number of positive observations
        self.neg = np.zeros(self.K, dtype = int) # number of negative observations
        self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
        for k in range(self.K):
          self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])
        
        self.cew = self.crs * self.confidence_ellipsoid_width(n)

    def get_arm(self, t):
        pulls = self.pos + self.neg
        Gram = np.tensordot(pulls, self.X2, axes = ([0], [0])) +  1e-4 * np.eye(self.d) / np.square(self.sigma0)
        Gram_inv = np.linalg.inv(Gram)
        theta, _ = self.solve()

        # randomize the UCB interval    
        a = randomize_confidence(self.M, self.pdist, self.pnormal_std, self.pfixed, self.is_optimistic, self.is_coupled)    
    
        # UCBs
        self.mu = self.X.dot(theta) + a * self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))

        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "RandUCBLog"

class LogGreedy(LogBanditAlg):
    def __init__(self, env, n, params):
        self.epsilon = 0.05
        super().__init__(env, n, params)

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if np.random.rand() < self.epsilon * np.sqrt(self.n / (t + 1)) / 2:
          self.mu[np.random.randint(self.K)] = np.Inf
        else:
          theta, _ = self.solve()
          self.mu = self.sigmoid(self.X.dot(theta))
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "Log e-greedy"

class LogTS(LogBanditAlg):
    def get_arm(self, t):

        theta, Gram = self.solve()

        Gram_inv = np.square(self.a) * np.linalg.inv(Gram)

        # posterior sampling
        # thetatilde = np.random.multivariate_normal(theta, Gram_inv)
        z = np.random.multivariate_normal(np.zeros(d), np.eye(d))
        # print(z)        
        thetatilde = theta + np.dot( scipy.linalg.sqrtm(Gram_inv), z)        

        self.mu = self.sigmoid(self.X.dot(thetatilde))

        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "GLM-TS (log)"

class LogFPL(LogBanditAlg):
    def solve(self):
        # normal noise perturbation
        pulls = self.pos + self.neg
        z = self.a * np.sqrt(pulls) * np.minimum(np.maximum(np.random.randn(self.K), -6), 6)

        # iterative reweighted least squares for Bayesian logistic regression
        # Sections 4.3.3 and 4.5.1 in Bishop (2006)
        # Pattern Recognition and Machine Learning
        theta = np.zeros(self.d)
        num_iter = 0
        while num_iter < 100:
            theta_old = np.copy(theta)

            Xtheta = self.X.dot(theta)
            R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
            Gram = np.tensordot(R * pulls, self.X2, axes = ([0], [0])) +  1e-4 * np.eye(self.d) / np.square(self.sigma0)
            Rz = R * pulls * Xtheta - (pulls * self.sigmoid(Xtheta) - (self.pos + z))
            theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

            if np.linalg.norm(theta - theta_old) < 1e-3:
                break;
            num_iter += 1

        return theta, Gram

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t < self.d:
            self.mu[t] = np.Inf
        else:
            # history perturbation
            theta, _ = self.solve()
            self.mu = self.sigmoid(self.X.dot(theta))
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "GLM-FPL (log)"

class LogPHE:
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.sigma0 = 1
        self.a = 2

        for attr, val in params.items():
            setattr(self, attr, val)

        # sufficient statistics
        self.pos = np.zeros(self.K, dtype = int) # number of positive observations
        self.neg = np.zeros(self.K, dtype = int) # number of negative observations
        self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
        self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
        for k in range(self.K):
            self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

    def update(self, t, arm, r):
        self.pos[arm] += r
        self.neg[arm] += 1 - r

    def sigmoid(self, x):
        return 1 / (1 + np.exp(- x))

    def solve(self):
        # iterative reweighted least squares for Bayesian logistic regression
        # Sections 4.3.3 and 4.5.1 in Bishop (2006)
        # Pattern Recognition and Machine Learning
        theta = np.zeros(self.d)
        num_iter = 0
        while num_iter < 100:
            theta_old = np.copy(theta)

            Xtheta = self.X.dot(theta)
            R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
            pulls = self.posp + self.negp
            Gram = np.tensordot(R * pulls, self.X2, axes = ([0], [0])) +   1e-4 * np.eye(self.d) / np.square(self.sigma0)
            Rz = R * pulls * Xtheta -         self.posp * (self.sigmoid(Xtheta) - 1) -         self.negp * (self.sigmoid(Xtheta) - 0)
            theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

            if np.linalg.norm(theta - theta_old) < 1e-3:
                break;
            num_iter += 1

        return theta, Gram

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t < self.d:
            self.mu[t] = np.Inf
        else:
            # history perturbation
            pulls = self.pos + self.neg
            pseudo_pulls = np.ceil(self.a * pulls).astype(int)
            pseudo_reward = np.random.binomial(pseudo_pulls, 0.5)
            self.posp = self.pos + pseudo_reward
            self.negp = self.neg + pseudo_pulls - pseudo_reward
      
            theta, _ = self.solve()
            self.mu = self.sigmoid(self.X.dot(theta)) + self.tiebreak
            
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "LogPHE"

if __name__ == "__main__":
    base_dir = os.path.join(".", "Results", "GenLin")

    algorithms = [
        # baselines
        (LogUCB, {}, "GLM-UCB"),
        (UCBLog, {}, "UCB-GLM"),
        (LogTS, {}, "GLM-TS"),
        (LogGreedy, {}, "e-greedy"),
        # sota
        (LogPHE, {"a": 0.5}, "LogPHE (a = 0.5)"),        
        (RandUCBLog, {"M": 20, "pdist": "Normal", "pnormal_std":0.125, "is_optimistic":True, "is_coupled":True}, "RandUCBLog"),    
        (RandLogUCB, {"M": 20, "pdist": "Normal", "pnormal_std":0.125, "is_optimistic":True, "is_coupled":True}, "RandLogUCB"),      
    ]

    num_runs = 50
    n = 20000
    K = 100

    environments = [
        (LogBandit, {}, 5, "Bernoulli (d=5)"),
        (LogBandit, {}, 10, "Bernoulli (d=10)"),      
        (LogBandit, {}, 20, "Bernoulli (d=20)"),
        # (LogBandit, {}, 100, "Bernoulli (d=100)")
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
            alg_class, alg_params, alg_name = alg_def[0], alg_def[1], alg_def[-1]        

            fname = os.path.join(res_dir, alg_name)        
            if os.path.exists(fname):
                print('File exists. Will load saved file. Moving on to the next algorithm')
            else:
                regret, _ = evaluate(alg_class, alg_params, envs, n)
                cum_regret = regret.cumsum(axis=0)
                np.savetxt(fname, cum_regret, delimiter=",")
