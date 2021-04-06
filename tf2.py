import numpy as np 
import tensorflow_probability as tfp 
import tensorflow as tf

dtype = np.float32
step_size = 0.01

def kernel(x):
    return tf.reduce_sum(x)

class Wavefunction:
    def __init__(self, alpha, beta):
        self.alpha = alpha 
        self.beta = beta
    def __call__(self, r):
        # print(r)
        r1 = r[0]**2 + r[1]**2
        r2 = r[2]**2 + r[3]**2
        r12 = tf.sqrt((r[0]-r[2])**2 + (r[1]-r[3])**2)
        deno = r12/(1+self.beta*r12)
        return tf.math.log(tf.exp(-0.5*self.alpha*(r1+r2)+deno)**2)

def LocalEnergy(r, k):
    alpha = 1
    beta = 0
    r1 = (r[0]**2 + r[1]**2)
    r2 = (r[2]**2 + r[3]**2)
    r12 = tf.sqrt((r[0]-r[2])**2 + (r[1]-r[3])**2)
    deno = 1.0/(1+beta*r12)
    deno2 = deno*deno
    le = 0.5*(1-alpha*alpha)*(r1 + r2) +2.0*alpha + 1.0/r12+deno2*(alpha*r12-deno2+2*beta*deno-1.0/r12)
    print(le)
    return le

prob = tfp.distributions.Normal(loc = dtype(0), scale = dtype(1))
step_function = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(target_log_prob_fn = Wavefunction(0.1, 0), 
                                                step_size =step_size)

step_function = tfp.mcmc.RandomWalkMetropolis(Wavefunction(1,0))

samples = tfp.mcmc.sample_chain(num_results = 1000, 
                                current_state = tf.convert_to_tensor([0.1,0, -0.1, 2], dtype = dtype),
                                kernel = step_function,
                                trace_fn=LocalEnergy)


