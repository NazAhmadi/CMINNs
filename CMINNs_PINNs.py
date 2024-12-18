import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import optax
import sys
import jax.nn as jnn
import pandas as pd
import random
from scipy.integrate import odeint
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from jax.experimental.ode import odeint as jax_odeint

#PK simulation
V1 = 810


def simulate_system(dose_times, num=1900, dt=0.01, k10=0.868*24, k12=0.006*24, k21=0.0838*24):
    A1 = []
    A2 = []
    A1.append(0)  # Initial condition for A1
    A2.append(0)  # Initial condition for A2

    # dose_times = [13, 17, 21]

    for i in range(1, num + 1):
        product = i * dt
        if product.is_integer() and int(product) in dose_times:
            mu = 3e+7
        else:
            mu = 0

        A1i = A1[i-1] - dt * (k10 + k12) * A1[i-1] + dt * k21 * A2[i-1] + mu
        A2i = A2[i-1] + dt * k12 * A1[i-1] - dt * k21 * A2[i-1]
        A1.append(A1i)
        A2.append(A2i)

    return jnp.array(A1)

#Auxiliary functions
def normalize(x):
  x_min = jnp.min(x)
  x_max = jnp.max(x)
  x_scaled = (x - x_min)/(x_max - x_min)
  return x_scaled

def denormalize(x_scaled, y_dense):
  x_min = jnp.min(y_dense)
  x_max = jnp.max(y_dense)
  x= x_scaled*(x_max - x_min)+ x_min
  return x

#PD simulation
V1 = 810

#Expriment 2 , day 13 - 17 - 21
psi = 20
k2 = 6e-4
lambda_0_exp1 =0.273
lambda_1_exp1 = 0.814
w0_exp1 = 2.72
k1 = 0.3


# Defining the model
def pd(t, w0=w0_exp1, lambda_0=lambda_0_exp1, lambda_1=lambda_1_exp1, k1=k1, k2=k2):
    def func(y, t):
        dose_times = [0,17-13, 21-13]
        A1 = simulate_system(dose_times)
        t_dense = jnp.linspace(0, 19, 1901)
        c = jnp.interp(t, t_dense, A1)/V1

        x1, x2 = y
        w = x1 + x2

        dx1_dt = lambda_0 * x1 * (1 + (lambda_0 / lambda_1 * w)**(psi))** (-1/psi) - k2 *c* x1
        dx2_dt = k2 *c* x1 - k1 * x2

        return [dx1_dt, dx2_dt]

    y0 = jnp.array([0.75, 1.21], dtype=jnp.float32)
    return jax_odeint(func, y0, t)



t_dense = jnp.linspace(0, 19, 1901)

y_dense = pd(jnp.ravel(t_dense))

w= y_dense[:,0]+y_dense[:,1]

t_label2 = np.array([9, 10, 11, 13, 15, 17, 19, 23, 28, 31])
w_real2 = np.array([0.6, 0.76, 1.08, 1.92, 2.72, 2.64, 2.08, 2.32, 4.24, 6.44])

t_control2 = np.array([ 13, 15, 17, 19, 23, 28, 31])
w_control2 = np.array([ 1.96, 3.75, 4.65, 6.36, 9.93, 13.58, 17.11])

####### data ########
t_i  = jnp.array([[0]])
i1 = normalize(y_dense[:, [0]])
i2 = normalize(y_dense[:, [1]])
IC = np.array([i1[0], i2[0], [1.92],[10]])
IC = np.array(IC).astype(np.float32).flatten()

w_real = w_real2[3:]
t_data = t_label2[3:]-13
data = w_real


#############################
t_dense = np.linspace(0, 35, 3501)[:, None]
tmin, tmax = t_dense[0,0], t_dense[-1,0]

def init_params(layers,seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = jax.random.normal(key, shape=(n_in, n_out)) / jnp.sqrt(n_in) # random initialization
        B = jax.random.normal(key, shape=(n_out,))
        params.append({'W': W, 'B': B , 'k2' : 0.6 })
    return params


def feature_transform(t):
    t = 0.1 * t
    return jnp.concatenate(
        (t, jnp.exp(t), jnp.exp(2 * t), jnp.exp(3 * t), jnp.exp(4 * t)),
        axis=1,
    )


def fwd(params, t):
    X = feature_transform(t)  # Apply the feature_transform to input t
    inputs= X
    *hidden, last = params
    for layer in hidden:
        inputs = jax.nn.tanh(inputs @ layer['W'] + layer['B'])

    Y= inputs @ last['W'] + last['B']
    Y=jnn.softplus(Y)
    return Y


@jax.jit
def MSE(true,pred):
  return jnp.mean((true-pred)**2)

################################################
def ODE_loss(t, y1, y2,y5,kt, params):
    psi = 20
    l0 =0.273
    l1 = 0.814
    k1 = kt(t)
    k2 = (jnp.tanh(params[0]['k2'])*0.5 + 6.5)*1e-4

    t_dose=[0,17-13, 21-13]

    A1 = simulate_system(t_dose)
    t_dense = jnp.linspace(0,19 , 1901)
    c = jnp.interp(t, t_dense, A1)/V1
    c = c.reshape(-1,1)

    y_1= denormalize(y1(t),y_dense[:,[0]])
    y_2= denormalize(y2(t),y_dense[:,[1]])

    y1_t = lambda t: jax.grad(lambda t: jnp.sum(y1(t)))(t)
    y2_t = lambda t: jax.grad(lambda t: jnp.sum(y2(t)))(t)

    x1, x2 = y_1, y_2
    x5 = y5(t)

    x1_scale= jnp.max(y_dense[:,[0]]) - (jnp.min(y_dense[:,[0]]))
    x2_scale= jnp.max(y_dense[:,[1]]) - (jnp.min(y_dense[:,[1]]))


    ode1 = x1_scale * y1_t(t) - ((l0 * x1 * (1 + (l0 / l1 * x5)**(psi))** (-1/psi) - k2 *c* x1))
    ode2 = x2_scale * y2_t(t) - ((k2 *c* x1 - k1 * x2))
    ode5= x1+ x2 - x5

    return ode1, ode2, ode5




def loss_fun(params,l1, l2,l3, l4, t_i, t_d, t_c, data_IC, data):

    # l3, l4 = 1, 1 you can use them if you have multi comparment model
    y1_func = lambda t: fwd(params, t)[:, [0]]
    y2_func = lambda t: fwd(params, t)[:, [1]]

    y5_func = lambda t: fwd(params, t)[:, [2]]
    ft      = lambda t: fwd(params, t)[:, [3]]

    loss_y1, loss_y2, loss_y5 = ODE_loss(t_c, y1_func, y2_func,y5_func,ft, params)

    loss_y1 = l1*loss_y1
    loss_y2 = l2*loss_y2


    loss_ode1 = jnp.mean(loss_y1 ** 2)
    loss_ode2 = jnp.mean(loss_y2 ** 2)
    loss_ode5 = jnp.mean(loss_y5 ** 2)


    t_i = t_i.flatten()[:,None]
    pred_IC = jnp.concatenate([y1_func(t_i), y2_func(t_i),y5_func(t_i), ft(t_i) ],axis=1)
    loss_IC = MSE(data_IC, pred_IC)

    t_d    = t_d.flatten()[:,None]
    w = denormalize(y1_func(t_d), y_dense[:,[0]]) + denormalize(y2_func(t_d), y_dense[:,[1]])
    data = data.reshape(-1, 1)

    loss_data = MSE(data, w)


    return loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode5






def loss_fun_total(params, params_l1, params_l2,params_l3, params_l4, t_i, t_d, t_c, data_IC, data, loss_weight):

    loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode5 = loss_fun(params,params_l1, params_l2,params_l3, params_l4, t_i, t_d, t_c, data_IC, data)
    loss_total = loss_weight[0]*loss_IC+ loss_weight[1]*loss_data\
                + loss_weight[2]*loss_ode1+ loss_weight[3]*loss_ode2+ loss_weight[4]*loss_ode5

    return loss_total



@jax.jit
def update(params, params_l1, params_l2, params_l3, params_l4, opt_state, opt_state_l1, opt_state_l2,opt_state_l3,opt_state_l4, t_i, t_data, t_c, IC, data, loss_weight):
  grads=jax.grad(loss_fun_total, argnums=[0,1,2,3,4])(params, params_l1, params_l2,params_l3, params_l4, t_i, t_data, t_c, IC, data, loss_weight)

  #Update params
  updates, opt_state = optimizer.update(grads[0], opt_state)
  params = optax.apply_updates(params, updates)

  updates_l1, opt_state_l1 = optimizer.update(-grads[1], opt_state_l1)
  params_l1 = optax.apply_updates(params_l1, updates_l1)

  updates_l2, opt_state_l2 = optimizer.update(-grads[2], opt_state_l2)
  params_l2 = optax.apply_updates(params_l2, updates_l2)

  updates_l3, opt_state_l3 = optimizer.update(-grads[3], opt_state_l3)
  params_l3 = optax.apply_updates(params_l2, updates_l2)

  updates_l4, opt_state_l4 = optimizer.update(-grads[4], opt_state_l4)
  params_l4 = optax.apply_updates(params_l2, updates_l2)

  return  params, params_l1, params_l2, params_l3, params_l4, opt_state, opt_state_l1, opt_state_l2, opt_state_l3, opt_state_l4





seed = random.randint(1, 1000)
params = init_params([5] + [30]*6+[4], seed)

optimizer = optax.adam(1e-4)
opt_state = optimizer.init(params)
N_c= 100


keys = jax.random.split(jax.random.PRNGKey(seed), 10)
params_l1 = jax.random.uniform(keys[0], shape=(N_c + 1, 1))
params_l2 = jax.random.uniform(keys[1], shape=(N_c + 1, 1))
params_l3 = jax.random.uniform(keys[2], shape=(N_c + 1, 1))
params_l4 = jax.random.uniform(keys[3], shape=(N_c + 1, 1))


opt_state_l1 = optimizer.init(params_l1)
opt_state_l2 = optimizer.init(params_l2)
opt_state_l3 = optimizer.init(params_l3)
opt_state_l4 = optimizer.init(params_l4)
#################################################################
num_seeds = 1

all_C1 = []
k2_values_list = []
k1_values_list = []
l1_values_list = []
l2_values_list = []

loss_his, loss_indi_his, epoch_his = [], [], []

time_points = np.array([0, 3.99, 4.5, 7.99, 8.5 , 12,16,19])

loss_weight_phase1 = [1, 1, 0, 0, 0, 0]
loss_weight_phase2 = [1, 1, 1, 1, 1, 1]
iterations_per_interval = 30000

##############################
k11=[]
k22=[]
all_outputs = []
all_time_points = []

#################### Training #########################
for i in range(1, len(time_points)):
    total_desired_points = 100
    print(f"Interval {i}: {time_points[i-1]} to {time_points[i]}")
    print(f' Initial condition:   ti = {t_i} , f(ti) =  {IC}')
    t_c= jnp.linspace(time_points[i-1], time_points[i], total_desired_points+1)[:, None]


    epochs_phase = iterations_per_interval

    for ep in range(epochs_phase+ 1):
        loss_weight = loss_weight_phase2

        params, params_l1, params_l2, params_l3,\
         params_l4, opt_state, opt_state_l1, opt_state_l2,\
          opt_state_l3, opt_state_l4 = update(params, params_l1,\
                                              params_l2, params_l3, params_l4,\
                                              opt_state, opt_state_l1, opt_state_l2,\
                                              opt_state_l3,opt_state_l4, t_i, t_data, t_c,\
                                              IC, data, loss_weight)

        # print loss and epoch info
        if ep %(1000) ==0:
          loss_val = loss_fun_total(params,params_l1, params_l2,params_l3, params_l4, t_i, t_data, t_c, IC, data, loss_weight)
          loss_val_individual = loss_fun(params,params_l1, params_l2,params_l3, params_l4, t_i, t_data, t_c, IC, data)
          epoch_his.append(ep)
          loss_his.append(loss_val)
          loss_indi_his.append(loss_val_individual)

        if ep %(45000) ==0:
          k2_out =  k2 = (jnp.tanh(params[0]['k2'])*0.5 + 6.5)*1e-4#(jnp.tanh(params[0]['k2'])*7e-4 + 10e-4)

    print(f"k2 value in interval {i} = {k2_out:.4e}")
    k22.append(k2_out)

    t_i  = jnp.array([[time_points[i]]])
    IC  =  fwd(params,t_i)
    ############ saving the results#################################
    t_c_interval = jnp.linspace(time_points[i-1], time_points[i], total_desired_points+1)[:-1]
    # Calculate the outputs for this interval
    interval_outputs = fwd(params, t_c_interval[:, None])
    all_outputs.append(interval_outputs)
    all_time_points.extend(t_c_interval.tolist())  # Append the time points to the main list
    ################################################################

    np.savez(f'./params_{i}.npz', *params)

    t_dense = np.linspace(0, 19, 1901)[:, None]
    pred = fwd(params,t_dense)

######### saving ################ 
k22_array = jnp.array(k22)

weights = [4, 0.5, 3.5, 0.5, 3.5, 4, 3]
weighted_values = []

for i in range(7):
    weighted_values.append(weights[i] * k22_array[i])

k2 = sum(weighted_values)

# Normalize by dividing by 19
k2_normalized = k2 / 19

data = {
    'k22_values': k22_array,
    'weights': weights,
    'weighted_values': weighted_values
}

df = pd.DataFrame(data)
df.loc['Total'] = ['', '', k2]
df.loc['Normalized'] = ['', '', k2_normalized]

# Save the DataFrame to a CSV file
csv_filename = 'k22_values.csv'
df.to_csv(csv_filename, index=False)

print(f"Values have been saved to {csv_filename}")


########## plotting the results ######### 
# Data provided
k22_values = [
    k22_array[0],
    k22_array[1],
    k22_array[2],
    k22_array[3],
    k22_array[4],
    k22_array[5],
    k22_array[6]
]

weights = [4.0, 0.5, 3.5, 0.5, 3.5, 4.0, 3.0]

# Calculate the corresponding time points based on weights
time_points = [0]
for w in weights:
    time_points.append(time_points[-1] + w)

# Plotting the results
plt.figure(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(k22_values)))  # Use a colormap to get different colors

for i in range(len(k22_values)):
    plt.plot([time_points[i], time_points[i+1]], [k22_values[i], k22_values[i]],color='blue', linewidth=2)
    plt.axvline(x=time_points[i+1], color='grey', linestyle='--', alpha=0.5)  # Add vertical lines to separate time ranges



plt.xlim(0, 19)
plt.ylim(min(k22_values) * 0.95, max(k22_values) * 1.05)
plt.title('k2 Values Over Discrete Time Ranges')
plt.xlabel('Time')
plt.ylabel('k2')
plt.grid(True)
plt.savefig('./k2.png', dpi=300)
plt.show()


################ Saving ################ 
# Convert list of outputs to a JAX array (or NumPy if you prefer)
all_outputs_array = jnp.vstack(all_outputs)
all_time_points_array = jnp.array(all_time_points)

ft =pred[:,3]
k = jnn.softplus(ft)
t_pred = t_dense
df_f = pd.DataFrame({"t": np.ravel(all_time_points_array), "x1": np.ravel(all_outputs_array[:, [0]]),"x2": np.ravel(all_outputs_array[:, [1]]),"w": np.ravel(all_outputs_array[:, [2]]),  "k1": np.ravel(all_outputs_array[:, [3]]) })
df_f.to_csv("./output.csv", index=False)


df_f = pd.DataFrame({"t": np.ravel(all_time_points_array), "k1": np.ravel(jnn.softplus(all_outputs_array[:, [3]])) })
df_f.to_csv("./k_pharma.csv", index=False)
###############################


# Plotting the results _ Validation
V1 = 810
psi = 20
lambda_0_exp1 =0.273
lambda_1_exp1 = 0.814


def k2_func(t):
    return jax.lax.cond(t < 4,
                        lambda _: k22_array[0],  # Value for k2 in the range [0, 3.99]
                        lambda _: jax.lax.cond(t < 8,
                                               lambda _: k22_array[2],  # Value for k2 in the range [4, 7.99]
                                               lambda _: k22_array[5],  # Value for k2 in the range [8, 19]
                                               operand=None),
                        operand=None)


data = pd.read_csv("./output.csv")

# Defining the model
def pd(t, w0=w0_exp1, lambda_0=lambda_0_exp1, lambda_1=lambda_1_exp1):
    def func(y, t):
        dose_times = [0,17-13, 21-13]
        A1 = simulate_system(dose_times)
        t_dense = jnp.linspace(0, 19, 1901)
        c = jnp.interp(t, t_dense, A1)/V1
        t_d= np.array(data["t"])
        k_values = jnp.array(data["k1"])
        k1 = jnp.interp(t, t_d, k_values)

        k2 = k2_func(t)
        # k1 = 1
        # k1=0.98
        x1, x2 = y
        w = x1 + x2

        dx1_dt = lambda_0 * x1 * (1 + (lambda_0 / lambda_1 * w)**(psi))** (-1/psi) - k2 *c* x1
        dx2_dt = k2 *c* x1 - k1 * x2
        return [dx1_dt, dx2_dt]

    y0 = jnp.array([0.75, 1.21], dtype=jnp.float32)
    return jax_odeint(func, y0, t)




t_dense = jnp.linspace(0, 19, 1901)  # Assuming 10 time units, change as necessary

y_dense = pd(jnp.ravel(t_dense))# /scale_factor


w= y_dense[:,0]+y_dense[:,1]#+ y_dense[:,2]#+y_dense[:,3]

plt.figure(figsize=(12, 6))

t_label2 = np.array([13, 15, 17, 19, 23, 28, 31])
w_real2 = np.array([1.92, 2.72, 2.64, 2.08, 2.32, 4.55, 6.65])
t_control2 = np.array([ 13, 15, 17, 19, 23, 28, 31])
w_control2 = np.array([ 1.96, 3.75, 4.65, 6.36, 9.93, 13.58, 17.11])



plt.scatter(t_label2, w_real2, label='W treated (data)')
plt.plot(t_dense+13, w, '-',color='red', label='W solver')

plt.title('Pharmacodynamic model over time with PINNs and k1(t)')
plt.xlabel('Time')
plt.ylabel('Weight of tumor')
plt.legend()
plt.grid(True,which="both", ls="--")
plt.xscale('linear')
plt.ylim(-0.1,18)
plt.xlim(13,32)
# plt.grid(which="both")
plt.show()
plt.savefig('./num.png', dpi=300)
