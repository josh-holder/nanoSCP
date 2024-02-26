import cvxpy as cp
import jax.numpy as jnp
import jax
import numpy as np
import time

def rocket_dynamics(x,u):
    new_x = jnp.zeros_like(x)

    dt = 0.01
    g = 9.81

    new_x[0] = x[0] + x[3]*dt
    new_x[1] = x[1] + x[4]*dt

    new_x[2] = x[2] + u[0]*dt
    new_x[3] = x[3] + (u[1] - g)*dt

    return new_x

def take_derivatives(dynamics, x, u):
    A = jax.jacfwd(dynamics, argnums=0)(x, u)
    B = jax.jacfwd(dynamics, argnums=1)(x, u)

    C = dynamics(x, u) - A@x - B@u

    return A, B, C

def init_optimization(x0, xf, time_horizon, x_size, u_size):
    time_horizon = 50
    # Define the problem
    x = cp.Variable((time_horizon,x_size))
    u = cp.Variable((time_horizon,u_size))

    # x0 = cp.Parameter(4)
    # xf = cp.Parameter(4)

    As = cp.Parameter((time_horizon,x_size,x_size))
    Bs = cp.Parameter((time_horizon,x_size,u_size))
    Cs = cp.Parameter((time_horizon,x_size))

    x0 = jnp.array([-10, 10, 2, -2])
    xf = jnp.array([0, 0, 0, 0])

    Q = 10*jnp.eye(x_size)
    R = jnp.eye(u_size)

    # Define the linearized dynamics
    A, B, C = take_derivatives(rocket_dynamics, x, u)

    #inequality constraints
    max_control_input = 20
    min_control_input = 0

    equality_constraints = []
    inequality_constraints = []
    for i in range(time_horizon):
        #Initialize the constraints which ensure that the linearized dynamics are satisfied.
        equality_constraints.append(x[i+1,:] == As[i]@x[i,:] + Bs[i]@u[i,:] + Cs[i])

        #Iniatilize control constraints
        inequality_constraints.append(cp.norm(u) <= max_control_input)

        inequality_constraints.append(x[i,1] >= 0)
        inequality_constraints.append(u[i,1] >= 0)

    equality_constraints.append(x[0,:] == x0)

    all_constraints = equality_constraints + inequality_constraints

    # Define the cost
    cost = cp.quad_form(u, R)

    # state_cost_per_timestep = [(1/2)*cp.quad_form(opt_states[i,:] - desired_state, state_costs) for i in range(1,time_horizon)]

    control_cost_per_timestep = [(1/2)*cp.quad_form(u[i,:], R) for i in range(time_horizon)]

    final_cost = (1/2)*cp.quad_form(x[time_horizon-1,:] - xf, Q)

    total_cost = cp.sum(control_cost_per_timestep) + final_cost

    # Solve the problem
    prob = cp.Problem(cp.Minimize(cost), all_constraints)

    return prob, x, u

def find_optimal_action(vehicle, initial_state, desired_state, controls_guess, time_horizon, dt=0.01, tolerance=0.01, max_iter=1, verbose=False):
    """
    1. Roll out a trajectory, T_curr, using the dynamics model and random actions
    While T_curr does not end in the desired state, within some tolerance:
        1. Discretize the trajectory, linearize at each point to get A, B, C matrices
        2. Solve the constrained optimization problem using these A, B, and C matrices to get a new trajectory, T_new
        3. T_curr = T_new
    """
    state_size = initial_state.shape[0]
    control_size = vehicle.action_space_size

    vehicle.propogateVehicleTrajectory(controls_guess)

    state_traj_curr = vehicle.state_trajectory[-50:]
    control_traj_curr = vehicle.control_trajectory[-50:]
    
    iter = 0
    last_value = np.inf
    curr_value = np.inf
    value_diff = -np.inf
    while value_diff < -tolerance and iter < max_iter:
        As = []
        Bs = []
        Cs = []
        
        start_linearization = time.time()
        
        for i, (state, control) in enumerate(zip(state_traj_curr, control_traj_curr)):
            # print(state.shape, state)
            # print(control.shape, control)
            if i % 5 == 0: #only linearize every 5 timesteps, and just use the same linearization for each
                A, B, C = vehicle.calculateLinearControlMatrices(state, control)
            As.append(A)
            Bs.append(B)
            Cs.append(C)

        end_linearization = time.time()

        start_optimization = time.time()

        optimization_prob, opt_states, opt_controls = initializeOptimizationProblem(time_horizon, As, Bs, Cs, initial_state, desired_state, state_size, control_size)

        optimization_prob.solve()

        if verbose:
            print(f"Linearization time: {end_linearization - start_linearization} seconds, optimization time: {time.time() - start_optimization} seconds")

        last_value = curr_value
        curr_value = optimization_prob.value
        value_diff = curr_value - last_value

        vehicle.resetVehicle()
        vehicle.propogateVehicleTrajectory(opt_controls.value)

        state_traj_curr = vehicle.state_trajectory
        control_traj_curr = vehicle.control_trajectory

        # print(f"Iteration {iter} complete. Value: {optimization_prob.value}")
        iter += 1

        # final_state_error = np.linalg.norm(state_traj_curr[-1,:] - desired_state)
    
    return opt_states.value, opt_controls.value
    
def optimize_trajectory(initial_state, desired_state, dt=0.01, tolerance=0.01, verbose=False):
    final_state_error = np.inf

    time_horizon = 50
    
    x_size = 4
    u_size = 2

    #Initialize a random guess at the trajectory
    control_choices = np.array([0.0,5.0,10.0])
    controls_guess = np.random.choice(control_choices, size=(time_horizon,u_size))

    action_num = 1

    start = time.time()

    actions_to_take_btwn_opt = 3

    iterations_since_improvement = 0
    best_final_state_error = np.inf

    while final_state_error > tolerance and iterations_since_improvement < 20:
        end_char = "\r" if not verbose else "\n"
        print(f"Finding optimal action {action_num}, with final error {final_state_error:.2f}:", end=end_char)
        vehicle_copy = copy.deepcopy(vehicle)

        max_iter = 1 if action_num > 1 else 10

        state_traj, control_traj = find_optimal_action(vehicle_copy, initial_state, desired_state, controls_guess, time_horizon, dt=dt, tolerance=0.01, max_iter=max_iter, verbose=verbose)

        final_state_error = np.linalg.norm(state_traj[-1,:] - desired_state)
        if final_state_error < best_final_state_error:
            iterations_since_improvement = 0
            best_final_state_error = final_state_error
        else:
            iterations_since_improvement += 1
            if verbose: print(f"Did not improve - iterations without improvement: {iterations_since_improvement}")


        for action_to_take in range(actions_to_take_btwn_opt):
            on_off_control = simpleOnOffControlsConvert(control_traj[action_to_take,:], threshold=0.1)
            # on_off_control = control_traj[action_to_take,:]
            vehicle.propagateVehicleState(on_off_control)

            if verbose and action_to_take == 0:
                print(f"Final state {state_traj[-1,:]}\n Final State error: {final_state_error}")
                print(f"Selected action: {on_off_control}")

        initial_state = vehicle.state

        controls_guess = jnp.vstack((control_traj[1:,:], np.zeros((actions_to_take_btwn_opt,control_size))))

        action_num += actions_to_take_btwn_opt

    print(f"Total optimization time: {time.time() - start} seconds")

    return vehicle

if __name__ == "__main__":
    print("hi")