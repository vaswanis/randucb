import numpy as np
import time

import os

from scipy.stats import norm, uniform, truncnorm

class Bandit:
    def __init__(self, mu, seed):
        self.mu = np.copy(mu)
        self.K = self.mu.size
        self.best_arm = np.argmax(self.mu)
        
        self.seed = seed
    
    def reset_random(self):
        self.random = np.random.RandomState(self.seed)
    
    def reward(self, arm):
        # instantaneous reward of the arm
        return self.rt[arm]

    def regret(self, arm):
        # instantaneous regret of the arm
        return self.rt[self.best_arm] - self.rt[arm]

    def pregret(self, arm):
        # expected regret of the arm
        return self.mu[self.best_arm] - self.mu[arm]

class BerBandit(Bandit):
    """Bernoulli bandit."""
    def __init__(self, mu, seed=None):
        super().__init__(mu, seed)

    def randomize(self):
        # generate random rewards
        self.rt = (self.random.rand() < self.mu).astype(float)

    def print(self):
        return "Bernoulli bandit with arms (%s)" % ", ".join("%.3f" % s for s in self.mu)

class GaussBandit(Bandit):
    """Bernoulli bandit."""
    def __init__(self, mu, seed=None):
        super().__init__(mu, seed)

    def randomize(self):
        # generate random rewards
        self.rt = (np.minimum(np.maximum(self.random.normal(self.mu, 0.1), 0), 1)).astype(float)  

    def print(self):
        return "Gaussian bandit with arms (%s)" % ", ".join("%.3f" % s for s in self.mu)

class BetaBandit(Bandit):
    """Beta bandit."""

    def __init__(self, mu, a_plus_b=4, seed=None):
        super().__init__(mu, seed)
        self.a_plus_b = a_plus_b

    def randomize(self):
        # generate random rewards
        self.rt = self.random.beta(self.a_plus_b * self.mu, self.a_plus_b * (1 - self.mu))

    def print(self):
        return "Beta bandit with arms (%s)" % ", ".join("%.3f" % s for s in self.mu)

def evaluate_one(Alg, params, env, n, period_size):
    """One run of a bandit algorithm."""
    alg = Alg(env, n, params)

    regret = np.zeros(n // period_size)
    for t in range(n):
        # generate state
        env.randomize()
        # print("episode", t, "-- rewards:", env.rt)

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

class UCB1:
    def __init__(self, env, n, params):
        self.K = env.K
        self.crs = 1.0 # confidence region scaling

        for attr, val in params.items():
            setattr(self, attr, val)

        self.pulls = 1e-6 * np.ones(self.K) # number of pulls
        self.reward = 1e-6 * np.random.rand(self.K) # cumulative reward
        self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
  
    def confidence_interval(self, t):
        ct = self.crs * np.sqrt(2 * np.log(t))
        return ct * np.sqrt(1 / self.pulls)

    def update(self, t, arm, r):
        self.pulls[arm] += 1
        self.reward[arm] += r

    def get_arm(self, t):
        t += 1 # time starts at one
        mu_hat = self.reward / self.pulls
        self.ucb = mu_hat + self.confidence_interval(t) + self.tiebreak 
        arm = np.argmax(self.ucb)
        return arm

    @staticmethod
    def print():
        return "UCB1"

class EpsilonGreedy:
    def __init__(self, env, n, params):
        self.K = env.K        
        self.epsilon = None
        for attr, val in params.items():
            setattr(self, attr, val)
        assert self.epsilon != None

        self.pulls = 1e-6 * np.ones(self.K) # number of pulls
        self.reward = 1e-6 * np.random.rand(self.K) # cumulative reward

    def update(self, t, arm, r):
        self.pulls[arm] += 1
        self.reward[arm] += r

    def get_arm(self, t):
        if np.random.rand() < self.epsilon * np.sqrt(n / (t + 1)) / 2:
            # random exploration
            arm = np.random.choice(self.K)
        else:
            mu_hat = self.reward / self.pulls
            arm = np.argmax(mu_hat)
        return arm

    @staticmethod
    def print():
        return "EpsilonGreedy"

def randomize_confidence(M = 20,  pdist = "Normal", pnormal_std = "0.125", pfixed = None, is_optimistic = True, is_coupled = True, K = None):
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


        probs[-1] = 1e-6# constant probability of choosing the last confidence interval            

        probs = probs / np.sum(probs)              
                
        if is_coupled == True:
            m = np.random.choice(M, p=probs)
        else:
            m = np.random.choice(M, size=K, p=probs)
        
        a = (lb + m * (ub - lb) / (M-1)) 

    return a

class UCB_Laplace(UCB1):
    def confidence_interval(self, t):
        return np.sqrt((1+1/self.pulls)*np.log(self.K*np.sqrt(self.pulls+1)/self.delta)/(2*self.pulls))

    @staticmethod
    def print():
        return "UCB_Laplace"

class RandUCB(UCB1):
    def __init__(self, env, n, params):
        self.K = env.K
        self.crs = 1.0 # confidence region scaling
        
        self.pnormal_std, self.pfixed = None, None
        for attr, val in params.items():
            setattr(self, attr, val)

        self.pulls = 1e-6 * np.ones(self.K) # number of pulls
        self.reward = 1e-6 * np.random.rand(self.K) # cumulative reward
        self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

    def update(self, t, arm, r):
        self.pulls[arm] += 1
        self.reward[arm] += r

    def get_arm(self, t):
        t += 1 # time starts at one

        a = randomize_confidence(self.M, self.pdist, self.pnormal_std, self.pfixed, self.is_optimistic, self.is_coupled, self.K)

        mu_hat = self.reward / self.pulls
        self.ucb = mu_hat + a * self.confidence_interval(t) + self.tiebreak 

        arm = np.argmax(self.ucb)
        return arm

    @staticmethod
    def print():
        return "RandUCB"

class RandUCB_Laplace(RandUCB):
    def confidence_interval(self, t):
        return np.sqrt((1+1/self.pulls)*np.log(self.K*np.sqrt(self.pulls+1)/self.delta)/(2*self.pulls))

    @staticmethod
    def print():
        return "RR_UCB_Laplace"

class KLUCB:
    def __init__(self, env, n, params):
        self.K = env.K
        self.crs = 1.0 # confidence region scaling

        for attr, val in params.items():
            setattr(self, attr, val)

        self.pulls = 1e-6 * np.ones(self.K) # number of pulls
        self.reward = 1e-6 * np.random.rand(self.K) # cumulative reward
        self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

    def UCB(self, p, N, t):
        C = np.square(self.crs) * (np.log(t) + 3 * np.log(np.log(t) + 1e-6)) / N

        qmin = np.minimum(np.maximum(p, 1e-6), 1 - 1e-6)
        qmax = (1 - 1e-6) * np.ones(p.size)
        for i in range(16):
            q = (qmax + qmin) / 2
            ndx = (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))) < C
            qmin[ndx] = q[ndx]
            qmax[~ndx] = q[~ndx]
        return q

    def update(self, t, arm, r):
        if (r > 0) and (r < 1):
            r = (np.random.rand() < r).astype(float)
        self.pulls[arm] += 1
        self.reward[arm] += r

    def get_arm(self, t):
        t += 1 # time starts at one
        self.ucb = self.UCB(self.reward / self.pulls, self.pulls, t) + self.tiebreak
        arm = np.argmax(self.ucb)
        return arm

    @staticmethod
    def print():
        return "KL-UCB"

class BTS:
    def __init__(self, env, n, params):
        self.K = env.K

        for attr, val in params.items():
          setattr(self, attr, val)

        self.alpha = np.ones(self.K) # positive observations
        self.beta = np.ones(self.K) # negative observations

    def update(self, t, arm, r):
        if (r > 0) and (r < 1):
            r = (np.random.rand() < r).astype(float)
        self.alpha[arm] += r
        self.beta[arm] += 1 - r

    def get_arm(self, t):
        # posterior sampling
        self.mu = np.random.beta(self.alpha, self.beta)
        arm = np.argmax(self.mu)
        return arm

    @staticmethod
    def print():
        return "B-TS"

class OBTS(BTS):
    def get_arm(self, t):
        # posterior sampling
        mu_hat = self.alpha / (self.alpha + self.beta)
        mu = []
        for m, a, b in zip(mu_hat, self.alpha, self.beta):
            sample = np.random.beta(a, b)
            while sample < m:
                sample = np.random.beta(a, b)
            mu.append(sample)
        self.mu = np.array(mu)

        arm = np.argmax(self.mu)
        return arm

    @staticmethod
    def print():
        return "O-B-TS"

class GTS:
    def __init__(self, env, n, params):
        self.K = env.K

        for attr, val in params.items():
            setattr(self, attr, val)

        self.sums = 0.5 * np.ones(self.K)
        self.pulls = np.ones(self.K) # number of pulls per arm
    
    def get_arm(self, t):
        mu_hat = self.sums / self.pulls
        sigma_hat = self.sigma / np.sqrt(self.pulls)
        samples = np.random.normal(mu_hat, sigma_hat)
        # print(np.argmax(samples))
        return np.argmax(samples)
    
    def update(self, t, arm, r):
        self.pulls[arm] += 1
        self.sums[arm] += r

    @staticmethod
    def print():
        return "G-TS"

class OGTS(GTS):
    def get_arm(self, t):
        mu_hat = self.sums / self.pulls
        sigma_hat = self.sigma / np.sqrt(self.pulls)
        samples = np.random.normal(mu_hat, sigma_hat)
        # transform into optimistic sample
        samples = mu_hat + np.abs(mu_hat - samples)
        return np.argmax(samples)

    @staticmethod
    def print():
        return "O-G-TS"

class FPL:
    def __init__(self, env, n, params):
        self.K = env.K
        self.eta = np.sqrt((np.log(self.K) + 1) / (self.K * n))

        for attr, val in params.items():
            setattr(self, attr, val)

        self.loss = 1e-6 * np.random.rand(self.K) # cumulative loss

    def update(self, t, arm, r):
        # estimate the probability of choosing the arm
        wait_time = 0
        while True:
            wait_time += 1
            ploss = self.loss + np.random.exponential(1 / self.eta, self.K)
            if np.argmin(ploss) == arm:
                break;
        self.loss[arm] += (1 - r) * wait_time

    def get_arm(self, t):
        # perturb cumulative loss
        ploss = self.loss + np.random.exponential(1 / self.eta, self.K)
        arm = np.argmin(ploss)
        return arm

    @staticmethod
    def print():
        return "FPL"

class Giro:
    def __init__(self, env, n, params):
        self.K = env.K
        self.a = 1

        for attr, val in params.items():
            setattr(self, attr, val)

        self.pulls = np.zeros(self.K, dtype=int) # number of pulls
        self.reward = np.zeros((n, self.K)) # rewards
        self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

    def update(self, t, arm, r):
        self.reward[self.pulls[arm], arm] = r
        self.pulls[arm] += 1

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t < self.K:
            # each arm is pulled once in the first K rounds
            self.mu[t] = np.Inf
        else:
            # bootstrapping
            for k in range(self.K):
                pseudo_pulls = self.a * self.pulls[k]
                floor_pulls = np.floor(pseudo_pulls).astype(int)
                rounded_pulls = floor_pulls +           (np.random.rand() < pseudo_pulls - floor_pulls)
                H = np.concatenate((self.reward[: self.pulls[k], k],           np.zeros(rounded_pulls), np.ones(rounded_pulls)))
                sub = np.random.randint(0, H.size, H.size)
                self.mu[k] = H[sub].mean() + self.tiebreak[k]
        arm = np.argmax(self.mu)
        return arm

    @staticmethod
    def print():
        return "Giro"

class PHE:
    def __init__(self, env, n, params):
        self.K = env.K
        self.a = 2

        for attr, val in params.items():
            setattr(self, attr, val)

        self.pulls = np.zeros(self.K, dtype=int) # number of pulls
        self.reward = np.zeros(self.K) # cumulative reward
        self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

    def update(self, t, arm, r):
        self.pulls[arm] += 1
        self.reward[arm] += r

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t < self.K:
            # each arm is pulled once in the first K rounds
            self.mu[t] = np.Inf
        else:
            # history perturbation
            pseudo_pulls = np.ceil(self.a * self.pulls).astype(int)
            pseudo_reward = np.random.binomial(pseudo_pulls, 0.5)
            self.mu = (self.reward + pseudo_reward) /         (self.pulls + pseudo_pulls) + self.tiebreak
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "PHE"


if __name__ == "__main__":
    base_dir = os.path.join(".", "Results", "MAB")

    # For epsilon-greedy variant of RUCB
    # epsilon = 0.001
    # pfixed = np.zeros(20)
    # pfixed[-1] = epsilon
    # pfixed[:5] = (1 - epsilon) / 5

    algorithms = [
        # baselines
        (GTS, {"sigma": 0.5}, "G-TS"),
        # (BTS, {}, "B-TS"),
        # (OBTS, {}, "O-B-TS"),
        (UCB1, {}, "UCB1"),
     
        # (UCB_Laplace, {"delta": 0.1}, "UCB-Laplace"),
        #(EpsilonGreedy, {"epsilon": 0.05}, "EpsilonGreedy"),
    
        # this next line shall not be changed!
        #   (RandUCB, {"M": 20, "pdist": "Normal", "pnormal_std":0.1, "is_optimistic":True, "is_coupled":True}, "RandUCB"),        
    
        # another variant that works
        (RandUCB, {"M": 20, "pdist": "Normal", "pnormal_std": 1/8, "is_optimistic": True, "is_coupled": True}, "RandUCB"),        
        #   (RR_UCB_Laplace, {"M": 20, "pdist": "Normal", "pnormal_std":0.125, "is_optimistic":True, "is_coupled":True, "delta": 0.1}, "RR-UCB-Laplace"),
        
        # ablation study
        # (RandUCB, {"M": 20, "pdist": "Normal", "pnormal_std": 1/8, "is_optimistic": False, "is_coupled": True}, "RandUCB-Not_Optimistic"),
        # (RandUCB, {"M": 20, "pdist": "Normal", "pnormal_std": 1/8, "is_optimistic": True, "is_coupled": False}, "RandUCB-Not_Coupled"),
        # (RandUCB, {"M": 20, "pdist": "Normal", "pnormal_std": 1, "is_optimistic": True, "is_coupled": True}, "RandUCB-High_Sigma"),
        # (RandUCB, {"M": 20, "pdist": "Normal", "pnormal_std": 1/16, "is_optimistic": True, "is_coupled": True}, "RandUCB-Low_Sigma"),
        # (RandUCB, {"M": 5, "pdist": "Normal", "pnormal_std": 1/8, "is_optimistic": True, "is_coupled": True}, "RandUCB-Small_M"),
        # (RandUCB, {"M": 100, "pdist": "Normal", "pnormal_std": 1/8, "is_optimistic": True, "is_coupled": True}, "RandUCB-Large_M"),
        # (RandUCB, {"M": 20, "pdist": "Uniform", "is_optimistic": True, "is_coupled": True}, "RandUCB-Uniform"),
  
        # tighter UCB
        #   (RandUCB, {"M": 20, "pdist": "Fixed", "pnormal_std": None, "pfixed": pfixed, "is_optimistic":True, "is_coupled":True},"RandUCB-tighter"),
  
        # link 2 TS
        # (RandUCB, {"M": 2000, "pdist": "Normal", "pnormal_std":0.125, "is_optimistic":True, "is_coupled":False}, "RandUCB_2_O-TS"),
        # (RandUCB, {"M": 4000, "pdist": "Normal", "pnormal_std":0.125, "is_optimistic":False, "is_coupled":False}, "RandUCB_2_TS"),
  
        # working not great but satifying variant of epsilon-greedy
        # (RandUCB, {"M": 2, "pdist": "Fixed", "pfixed":[0.95, 0.05], "is_optimistic":True, "is_coupled":True}, "RandUCB_2_e-Greedy"),
                     
        # (KLUCB, {}, "KL-UCB"),
        # (Giro, {}, "Giro"),
        #   (FPL, {},"FPL"),
        #   (PHE, {"a": 2.1}, "PHE (a = 2.1)"),
        #   (PHE, {"a": 1.1}, "PHE (a = 1.1)"),
        # (PHE, {"a": 0.5}, "PHE (a = 0.5)")
    ]
    num_runs = 50
    n = 20000
    K = 100

    environments = [    
        # (GaussBandit, {}, 0.5, "Gaussian (easy)"),
        # (GaussBandit, {}, 0.1, "Gaussian (hard)"),
        (BerBandit, {}, 0.5, "Bernoulli (easy)"),
        (BerBandit, {}, 0.1, "Bernoulli (hard)"),    
        (BetaBandit, {"a_plus_b": 4}, 0.5, "Beta (easy)"),
        (BetaBandit, {"a_plus_b": 16}, 0.1, "Beta (hard)")    
    ]

    for env_def in environments:
        env_class, env_params, max_gap, env_name = env_def[0], env_def[1], env_def[2], env_def[-1]
        print("================== running environment", env_name, "==================")
    
        envs = []
        for run in range(num_runs):

            np.random.seed(run)

            mu = max_gap * np.random.rand(K) + (0.5 - max_gap/2)
            envs.append(env_class(mu, seed=run, **env_params))
    
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
