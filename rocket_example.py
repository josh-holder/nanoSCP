import cvxpy as cp
import jax.numpy as jnp
import jax
import numpy as np
import time
import matplotlib.pyplot as plt

def rocket_dynamics_old(x,u, dt):
    g = 9.81
    rotated_u = jnp.array([u[0]*jnp.cos(x[4])- u[1]*jnp.sin(x[4]), u[0]*jnp.sin(x[4]) + u[1]*jnp.cos(x[4])])
    new_x = jnp.array([
        x[0] + x[2]*dt,
        x[1] + x[3]*dt,
        x[2] + rotated_u[0]*dt,
        x[3] + (rotated_u[1] - g)*dt,
        x[4] + x[5]*dt,
        x[5] + u[0]*dt
    ])

    return new_x

def rocket_dynamics(x,u, dt):
    g = 9.81
    new_x = jnp.array([
        x[0] + x[2]*dt,
        x[1] + x[3]*dt,
        x[2] + u[0]/x[4]*dt,
        x[3] + (2*u[1]/x[4] - g)*dt,
        x[4] - 1/100*jnp.linalg.norm(u)*dt,
    ])

    return new_x

def propagate_controls(u, x0, dt):
    x_traj = np.zeros((u.shape[0]+1, x0.shape[0]))

    x_traj[0,:] = x0
    for k in range(u.shape[0]):
        x_traj[k+1,:] = rocket_dynamics(x_traj[k,:], u[k,:], dt)

    return x_traj

def take_derivatives(dynamics, x, u, dt):
    A = jax.jacfwd(dynamics, argnums=0)(x, u, dt)
    B = jax.jacfwd(dynamics, argnums=1)(x, u, dt)

    C = dynamics(x, u, dt) - A@x - B@u

    return A, B, C

def init_optimization(T, As, Bs, Cs, x0, xf, x_size, u_size):
    # Define the problem
    x = cp.Variable((T+1,x_size))
    u = cp.Variable((T,u_size))

    # x0 = cp.Parameter(4)
    # xf = cp.Parameter(4)

    Q = jnp.array([[1,0,0,0,0],
                     [0,1,0,0,0],
                     [0,0,1,0,0],
                     [0,0,0,1,0],
                     [0,0,0,0,0]])
    R = jnp.eye(u_size)

    #inequality constraints
    max_control_input = 20

    equality_constraints = []
    inequality_constraints = []
    for i in range(T):
        #Initialize the constraints which ensure that the linearized dynamics are satisfied.
        equality_constraints.append(x[i+1,:] == As[i]@x[i,:] + Bs[i]@u[i,:] + Cs[i])

        #Iniatilize control constraints
        inequality_constraints.append(cp.norm(u[i,:]) <= max_control_input)

        inequality_constraints.append(x[i,1] >= 0)
        inequality_constraints.append(x[i,4] >= 1)
        inequality_constraints.append(u[i,1] >= 0)

    equality_constraints.append(x[0,:] == x0)
    inequality_constraints.append(x[T,1] >= 0)

    all_constraints = equality_constraints + inequality_constraints

    # state_cost_per_timestep = [(1/2)*cp.quad_form(x[i,:] - xf, Q) for i in range(1,T)]
    state_cost_per_timestep = [0]

    control_cost_per_timestep = [(1/2)*cp.quad_form(u[i,:], R) for i in range(T)]

    final_cost = (1/2)*cp.quad_form(x[T,:] - xf, 100*Q)

    total_cost = cp.sum(state_cost_per_timestep) + cp.sum(control_cost_per_timestep) + final_cost

    # Solve the problem
    prob = cp.Problem(cp.Minimize(total_cost), all_constraints)

    return prob, x, u

def find_optimal_action(x0, xf, controls_guess, T, dt=0.1, tolerance=0.01, max_iwoi=5, verbose=False):
    """
    1. Roll out a trajectory, T_curr, using the dynamics model and random actions
    While T_curr does not end in the desired state, within some tolerance:
        1. Discretize the trajectory, linearize at each point to get A, B, C matrices
        2. Solve the constrained optimization problem using these A, B, and C matrices to get a new trajectory, T_new
        3. T_curr = T_new
    """
    x_size = x0.shape[0]
    u_size = controls_guess.shape[1]

    print(controls_guess)
    x_traj_curr = propagate_controls(controls_guess, x0, dt)
    u_traj_curr = controls_guess
    
    iter = 0
    best_value = np.inf
    best_x_traj = None
    best_u_traj = None
    x_traj_hist = []
    x_traj_hist.append(x_traj_curr)
    iters_without_improvement = 0
    while iters_without_improvement < max_iwoi:
        As = []
        Bs = []
        Cs = []
        
        start_linearization = time.time()
        
        for i, (state, control) in enumerate(zip(x_traj_curr, u_traj_curr)):
            # print(state.shape, state)
            # print(control.shape, control)
            if i % 2 == 0: #only linearize every 1 timesteps, and just use the same linearization for each
                A, B, C = take_derivatives(rocket_dynamics, state, control, dt)
            As.append(A)
            Bs.append(B)
            Cs.append(C)

        end_linearization = time.time()

        start_optimization = time.time()

        optimization_prob, opt_states, opt_controls = init_optimization(T, As, Bs, Cs, x0, xf, x_size, u_size)

        optimization_prob.solve(solver=cp.SCS)

        print(f"Optimization status: {optimization_prob.status}")

        if verbose:
            print(f"Linearization time: {end_linearization - start_linearization} seconds, optimization time: {time.time() - start_optimization} seconds")

        x_traj_curr = propagate_controls(opt_controls.value, x0, dt)
        u_traj_curr = opt_controls.value

        if optimization_prob.value + tolerance < best_value:
            best_value = optimization_prob.value
            iters_without_improvement = 0

            best_x_traj = x_traj_curr
            best_u_traj = u_traj_curr
            x_traj_hist.append(x_traj_curr)
        else:
            iters_without_improvement += 1

        print(f"Iteration {iter} complete. Value: {optimization_prob.value}, iwoi: {iters_without_improvement}")
        iter += 1
    
    for i, xt in enumerate(x_traj_hist):
        # if i != len(x_traj_hist) - 1:
        plt.plot(xt[:,0], xt[:,1],'k--', alpha=(i+1)/len(x_traj_hist))

    for k in range(T):
        print(np.linalg.norm(best_u_traj[k,:]))
        if k % 5 == 0:
            plt.quiver(best_x_traj[k,0], best_x_traj[k,1], -best_u_traj[k,0], -best_u_traj[k,1], color='g')
            # plt.quiver(best_x_traj[k,0]-u[0], best_x_traj[k,1]-u[1], u[0], u[1], color='g')
    
    plt.plot([-15,5], [0,0], 'r', label='Ground constraint')
    plt.quiver(-100, -100, 1, 1, color='g', label='Thrust vector')
    plt.plot([-100,-101], [0,0], 'k--', label='Trajectories')
    plt.scatter(xf[0], xf[1], color='b', marker='x', label='Landing Target')

    plt.xlim(-15, 5)
    plt.ylim(-5, 20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return best_x_traj, best_u_traj
    
# def optimize_trajectory(x0, xf, dt=0.01, tolerance=0.01, verbose=False):
#     final_state_error = np.inf

#     T = 50
    
#     x_size = 4
#     u_size = 2

#     #Initialize a random guess at the trajectory
#     control_choices = np.array([0.0,5.0,10.0])
#     controls_guess = np.random.choice(control_choices, size=(T,u_size))

#     action_num = 1

#     start = time.time()

#     actions_to_take_btwn_opt = 3

#     iterations_since_improvement = 0
#     best_final_state_error = np.inf

#     while final_state_error > tolerance and iterations_since_improvement < 20:
#         end_char = "\r" if not verbose else "\n"
#         print(f"Finding optimal action {action_num}, with final error {final_state_error:.2f}:", end=end_char)

#         max_iter = 1 if action_num > 1 else 10

#         state_traj, control_traj = find_optimal_action(vehicle_copy, x0, xf, controls_guess, T, dt=dt, tolerance=0.01, max_iter=max_iter, verbose=verbose)

#         final_state_error = np.linalg.norm(state_traj[-1,:] - xf)
#         if final_state_error < best_final_state_error:
#             iterations_since_improvement = 0
#             best_final_state_error = final_state_error
#         else:
#             iterations_since_improvement += 1
#             if verbose: print(f"Did not improve - iterations without improvement: {iterations_since_improvement}")


#         for action_to_take in range(actions_to_take_btwn_opt):
#             on_off_control = simpleOnOffControlsConvert(control_traj[action_to_take,:], threshold=0.1)
#             # on_off_control = control_traj[action_to_take,:]
#             vehicle.propagateVehicleState(on_off_control)

#             if verbose and action_to_take == 0:
#                 print(f"Final state {state_traj[-1,:]}\n Final State error: {final_state_error}")
#                 print(f"Selected action: {on_off_control}")

#         x0 = vehicle.state

#         controls_guess = jnp.vstack((control_traj[1:,:], np.zeros((actions_to_take_btwn_opt,u_size))))

#         action_num += actions_to_take_btwn_opt

#     print(f"Total optimization time: {time.time() - start} seconds")

#     return vehicle

if __name__ == "__main__":
    x0 = np.array([-10, 10, -2, -2, 2])
    xf = np.array([0, 0, 0, 0, 0])
    T = 50

    controls_guess = np.zeros((T, 2))
    for k in range(T):
        dir = (k%10) - 5

        controls_guess[k,:] = np.array([dir, 12])

    xs, us = find_optimal_action(x0, xf, controls_guess, T, dt=0.1, verbose=True, max_iwoi=5)

    # plt.plot(xs[:,0], xs[:,1])
    # # for i in range(T):
    # #     norm_u = us[i,:]/np.linalg.norm(us[i,:])
    # #     plt.quiver(xs[i,0]-norm_u[0], xs[i,1]-norm_u[1], us[i,0], us[i,1])
    # # plt.quiver(xs[:T,0], xs[:T,1], us[:,0], us[:,1], color='g')
    # plt.show()
    