""" This module provides functions to simulate the Delta-Hes model: initialisation, simulation (for fully coupled cells and cells with an external Delta source) 
and calculation of nullclines. """

import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

# Initialisation functions of the system
def get_initial(lattice, params, initial_type = 'checkerboard', initial_val2 = None):
    '''Get initial condition for the system
    
    Input: 
    lattice: information about the structure of the system
    params: parameters of the system
    initial_type: type of initial condition, either 'uniform' or 'checkerboard'
     
    Returns:
     h0: initial condition for h
     m_h0: initial condition for m_h
     d0: initial condition for d
     m_d0: initial condition for m_d
     '''
    
    # Set random uniform distribution for delta
    U = np.random.uniform(0,1,[lattice.P, lattice.Q]) - 1/2      #a uniform random distribution
    epsilon = 1e-5
    sigma = 0.2
                                                        #multiplicative factor of Delta initial condition
    d0 = epsilon*(1 + sigma*U) 

    # depending on initial_type, set the initial condition for h0
    if initial_type == 'uniform':
        h0 = np.zeros([lattice.P, lattice.Q]) 

    elif initial_type == 'checkerboard':
        h0 = create_checkerboard(lattice.P, lattice.Q, val1=0, val2=initial_val2) # checkerboard pattern of Hes initial condition

    # set the initial condition for m_h0 and m_d0    
    m_h0 = np.zeros([lattice.P, lattice.Q])
    m_d0 = np.zeros([lattice.P, lattice.Q])

    return h0, m_h0, d0, m_d0

def create_checkerboard(P, Q, val1=0, val2=5):
    """
    Create a P x Q checkerboard pattern with alternating values.
    
    Parameters:
    - P: Number of rows (length of the lattice)
    - Q: Number of columns (width of the lattice)
    - val1: Value for the first set of squares (default 0)
    - val2: Value for the alternating set of squares (default 5)
    
    Returns:
    - checkerboard: array of shape (P, Q)
    """
    checkerboard = np.zeros((P, Q))
    
    # Assign values based on checkerboard pattern
    for i in range(P):
        for j in range(Q):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = val1
            else:
                checkerboard[i, j] = val2
    
    return checkerboard

def get_params(gamma_h, gamma_d, gamma_m, p_h, p_d, T_h, T_coupling, w_h, w_coupling, l, n, lattice, grad_hes = True, grad_coup = True, grad_hes_strength = 0, grad_coup_strength = 0):
    """Returns a SimpleNamespace containing model parameters calculated from physical parameters.

    Input:
    gamma_h: decay rate of Hes
    gamma_d: decay rate of Delta
    gamma_m: decay rate of mRNA

    p_h: threshold concentration of Hes Hill function repression
    p_d: threshold concentration of Delta Hill function activation

    T_h: time delay of Hes oscillations
    T_coupling: time delay of Delta coupling

    w_h: strength of Hes oscillations
    w_coupling: strength of Delta coupling

    l: Hill function coefficient for Hes repression
    n: Hill function coefficient for Delta activation

    grad_hes: boolean to apply gradient to Hes time delay
    grad_coup: boolean to apply gradient to Delta coupling time delay
    grad_hes_strength: strength of gradient for Hes time delay
    grad_coup_strength: strength of gradient for Delta coupling time delay

    lattice: SimpleNamespace containing lattice properties (P, Q)
    
    Returns: 
    params: SimpleNamespace containing model parameters

    """
    
    params = SimpleNamespace(
        gamma_h = gamma_h,
        gamma_d = gamma_d,
        gamma_m = gamma_m,
        p_h = p_h,
        p_d = p_d,
        T_h = T_h,
        T_coupling = T_coupling,
        w_h = w_h,
        w_coupling = w_coupling,
        l = l,
        n = n, 
        grad_hes = grad_hes,
        grad_coup = grad_coup,
        grad_hes_strength = grad_hes_strength,
        grad_coup_strength = grad_coup_strength
    )
    
    # apply gradient to correct part of model (Hes or coupling)
    if grad_hes:
        params.T_h = params.grad_hes_strength * np.arange(0, lattice.P) + params.T_h
    else:
        params.T_h = np.ones(lattice.P) * params.T_h
    
    if grad_coup:
        params.T_coupling = params.grad_coup_strength * np.arange(0, lattice.P) + params.T_coupling
    else:
        params.T_coupling = np.ones(lattice.P) * params.T_coupling

    return params

def get_lattice(P, Q):
    '''Creates a SimpleNamespace object with the lattice parameters of the system, so in 2D dimensions, connectivity matrix and weights
    Input:
    P : length of the PSM in cells
    Q : width of the PSM in cells (Q = 1 for 1D system)
    
    Returns:
    lattice : SimpleNamespace object with the lattice parameters of the system

    (lattice.w : weights of the cells in the system, 2D array of size P x Q
    lattice.connectivity : connectivity matrix of the system, 2D array containing cells neighbours)
    '''

    lattice = SimpleNamespace()

    # get sizes of the system 
    lattice.P = P            # length of PSM + tailbud
    lattice.Q = Q            # diameter (2D so width) of PSM and tailbud, if Q = 1 then we have 1D system

    # define the weights, connectivity matrix of the system, for 1D and 2D systems
    if lattice.Q > 1:
        lattice.w = 4*np.ones([P, Q])    # weight for a square lattice of cells, with each 4 neighbours
        lattice.w[0, :] = 3              # weights of the edges is 3
        lattice.w[-1, :] = 3
        lattice.w[:, 0] = 3
        lattice.w[:, -1] = 3
        lattice.w[0, 0] = 2              # weights of the corners is 2
        lattice.w[-1, 0] = 2 
        lattice.w[0, -1] = 2
        lattice.w[-1, -1] = 2     

    else:
        lattice.w = 2*np.ones(P)
        lattice.w[0] = 1            # weight of first and last cell is only 1
        lattice.w[-1] = 1        

    lattice.connectivity = connectivitymatrix(lattice.P, lattice.Q)  

    return lattice


def connectivitymatrix(P, Q):
    '''Calculates the connectivity matrix of the P x Q system

    Input:

    P, Q : dimensions of the system
    
    Returns: 
    
    M : connectivity matrix of the system
    '''
    k = P*Q                 # number of cells 
    M = np.zeros([k,k])

    # calculating the connectivity matrix

    for i in range(k):
        i_neighbour = findneigbours(i, P, Q)

        # check if the neighbours are within the range of the system
        for j in i_neighbour:
            if 0 <= j:
                M[i, j] = 1
    
    return M   

def findneigbours(index, P, Q):
    '''Finds neighbours of cell with index for system of P, Q cells
    Input :
    index: index of the cell
    P: number of cells in the system in the length direction
    Q: number of cells in the system in the width direction
    
    Returns:
    neighbours: list of indices of the neighbouring cells for cell with index   
    '''
    
    # finding neighbours for 1D system of cells
    if Q == 1:
        left = int(index - 1)
        right = int(index + 1)

        # check if the cell is on the edge of the system
        if index % P == 0:
            left = -1
        if index % P == P - 1:
            right = -1

        neighbours = [left, right]   
        
    # finding neighbours for 2D system of cells
    elif Q > 1: 

        left = index - 1
        right = index + 1
        up = index - P
        down = index + P

        # check if the cell is on the edge of the system
        if index % P == 0:
            left = -1
        if index % P == P - 1:
            right = -1
        if index < P:
            up = -1
        if index >= P*(Q-1):
            down = -1
        neighbours = [left, right, up, down]
                
    else: 
        print('Q should be at least 1')

    return neighbours

# Iteration functions of the system
def simulate(num_tsteps, dt, lattice, params, coupling_type = 'Delta', initial_type = 'checkerboard', initial_val2 = None):
    '''Runs the simulation of the model for a given number of time steps and time step size, other variables are the type of coupling in the system and the initialisation type.

    Inputs:
    num_tsteps: number of time steps to simulate
    dt: size of time step
    lattice: SimpleNamespace object containing the structure of the system
    params: SimpleNamespace object containing the parameters of the system
    coupling_type: type of coupling in the system, either 'Delta or 'Averaging' 
    initial_type: type of initialisation, either 'checkerboard' or 'uniform'
    initial_val2: second value for the checkerboard initialisation, if None then it is set to 5

    Returns:
    h: Hes values for each cell at each time step (num_tsteps, P, Q)
    m_h: Hes mRNA values for each cell at each time step (num_tsteps, P, Q)
    d: Delta values for each cell at each time step (num_tsteps, P, Q)
    m_d: Delta mRNA values for each cell at each time step (num_tsteps, P, Q)
    '''
    
    # ensure the time step size is small enough to capture the delay gradients
    if params.grad_hes_strength > 0 and dt > params.grad_hes_strength:
        raise ValueError('dt must be smaller than the Hes gradient strength')
    elif params.grad_coup_strength > 0 and dt > params.grad_coup_strength:
        raise ValueError('dt must be smaller than the Hes gradient strength')
    
    # set up the initial conditions
    if initial_type == 'checkerboard':

        h_init, m_h_init, d_init, m_d_init = get_initial(lattice, params, initial_type, initial_val2)

    elif initial_type == 'uniform':
        h_init, m_h_init, d_init, m_d_init = get_initial(lattice, params, initial_type)

    #intrinsic oscillator components
    h = np.zeros([num_tsteps, lattice.P, lattice.Q])
    m_h = np.zeros([num_tsteps, lattice.P, lattice.Q])
    h[0] = h_init
    m_h[0] = m_h_init

    #coupling components
    if coupling_type == 'Delta':
        m_d = np.zeros([num_tsteps, lattice.P, lattice.Q])
        d = np.zeros([num_tsteps, lattice.P, lattice.Q])
        m_d[0] = m_d_init
        d[0] = d_init

    elif coupling_type == 'Averaging':
        m_d = np.zeros([num_tsteps, lattice.P, lattice.Q])
        d = np.zeros([num_tsteps, lattice.P, lattice.Q])
        m_d[0] = m_d_init
        d[0] = d_init

    # iterate through the time steps and calculate the values of the next time step
    for i in tqdm(range(int(num_tsteps-1))):

        # calculate delayed values of Hes
        params.T_h_steps = np.round(params.T_h / dt).astype(int)
        h_delay = get_delayed_value(h, i, params.T_h_steps, lattice, params)

        # calculate delay in the coupling (no delay gives a value of 0)
        params.T_coup_steps = np.round(params.T_coupling / dt).astype(int)
        d_delay = get_delayed_value(d, i, params.T_coup_steps, lattice, params)

        # coupling dynamics 
        if coupling_type == 'Delta':
            d_tau_neighbours = d_neighbours(d_delay[i,:,:], lattice)
            d[i+1,:,:] = Euler(d[i,:,:], dd_dt(d[i,:,:], m_d[i,:,:], params), dt)
            m_d[i+1,:,:] = Euler(m_d[i,:,:], dmd_dt(m_d[i,:,:], h_delay[i,:,:], params), dt)
            couple_component = hill_function_positive(d_tau_neighbours, int(params.n), params.p_d)
        
        elif coupling_type == 'Averaging':
            h_average = np.mean(h_delay[i,:,:], axis=0) # average of the delayed values of Hes
            couple_component = h_average - h[i,:,:]
            # print('Error: Averaging coupling not implemented yet')

        # calculate the values of the next time step for Hes and Hes mRNA
        h[i+1,:,:] = Euler(h[i,:,:], dh_dt(h[i,:,:], m_h[i,:,:], params), dt)
        m_h[i+1,:,:] = Euler(m_h[i,:,:], dmh_dt(m_h[i,:,:], h_delay[i,:,:], couple_component, params, lattice), dt)


    return h, m_h, d, m_d   

def simulate_fixed_delta(num_tsteps, dt, external_delta, lattice, params):
    # h_init, m_h_init, d_init, m_d_init = get_initial(lattice, params, initial_type = 'checkerboard', initial_val2=300)
    h_init, m_h_init, d_init, m_d_init = get_initial(lattice, params, initial_type = 'uniform')

    #intrinsic oscillator components
    h = np.zeros([num_tsteps, lattice.P, lattice.Q])
    m_h = np.zeros([num_tsteps, lattice.P, lattice.Q])
    d = np.zeros([num_tsteps, lattice.P, lattice.Q])
    m_d = np.zeros([num_tsteps, lattice.P, lattice.Q])

    h[0] = h_init
    m_h[0] = m_h_init
    d[0] = d_init
    m_d[0] = m_d_init

    # iterate through the time steps and calculate the values of the next time step
    for i in tqdm(range(int(num_tsteps-1))):

        # calculate delayed values of Hes
        params.T_h_steps = np.round(params.T_h / dt).astype(int)
        h_delay = get_delayed_value(h, i, params.T_h_steps, lattice, params)

        d[i+1,:,:] = Euler(d[i,:,:], dd_dt(d[i,:,:], m_d[i,:,:], params), dt)
        m_d[i+1,:,:] = Euler(m_d[i,:,:], dmd_dt(m_d[i,:,:], h_delay[i,:,:], params), dt)

        couple_component = external_delta/params.w_coupling
        
        # calculate the values of the next time step for Hes and Hes mRNA
        h[i+1,:,:] = Euler(h[i,:,:], dh_dt(h[i,:,:], m_h[i,:,:], params), dt)
        m_h[i+1,:,:] = Euler(m_h[i,:,:], dmh_dt(m_h[i,:,:], h_delay[i,:,:], couple_component, params, lattice), dt)

    return h, m_h, d, m_d

def simulate_oscillating_delta(num_tsteps, dt, external_delta, coupling_delay, lattice, params):
    # h_init, m_h_init, d_init, m_d_init = get_initial(lattice, params, initial_type = 'checkerboard', initial_val2=300)
    h_init, m_h_init, d_init, m_d_init = get_initial(lattice, params, initial_type = 'uniform')

    #intrinsic oscillator components
    h = np.zeros([num_tsteps, lattice.P, lattice.Q]) 
    m_h = np.zeros([num_tsteps, lattice.P, lattice.Q])
    d = np.zeros([num_tsteps, lattice.P, lattice.Q])
    m_d = np.zeros([num_tsteps, lattice.P, lattice.Q])

    h[0] = h_init
    m_h[0] = m_h_init
    d[0] = d_init
    m_d[0] = m_d_init

    # iterate through the time steps and calculate the values of the next time step
    for i in tqdm(range(int(num_tsteps-1))):

        # calculate delayed values of Hes
        params.T_h_steps = np.round(params.T_h / dt).astype(int)

        h_delay = get_delayed_value(h, i, params.T_h_steps, lattice, params)

        d[i+1,:,:] = Euler(d[i,:,:], dd_dt(d[i,:,:], m_d[i,:,:], params), dt)
        m_d[i+1,:,:] = Euler(m_d[i,:,:], dmd_dt(m_d[i,:,:], h_delay[i,:,:], params), dt)

        couple_components = external_delta/params.w_coupling

        # calculate delay in the coupling (no delay gives a value of 0)
        if coupling_delay == 0:
            couple_component = couple_components[i]
        else:
            T_coup_steps = np.round(coupling_delay / dt).astype(int)
       
            if i < T_coup_steps:
                couple_component = couple_components[0]  
            else:
                couple_component = couple_components[i - T_coup_steps]       
    
        # calculate the values of the next time step for Hes and Hes mRNA
        h[i+1,:,:] = Euler(h[i,:,:], dh_dt(h[i,:,:], m_h[i,:,:], params), dt)
        m_h[i+1,:,:] = Euler(m_h[i,:,:], dmh_dt(m_h[i,:,:], h_delay[i,:,:], couple_component, params, lattice), dt)

    return h, m_h, d, m_d

def external_delta_oscillator(num_tsteps, dt, period, external_strength):
    """Generates an oscillating external Delta source for the simulation.
    
    Inputs:
    num_tsteps: number of time steps to generate the oscillation for
    dt: time step size
    period: period of the oscillation in minutes
    external_strength: strength of the oscillation, amplitude of the sine wave
    
    Returns:
    array of shape num_tsteps containing the oscillating external Delta source values
    """

    return external_strength*(np.sin(np.arange(num_tsteps) * dt * 2 * np.pi / period) + 1)

def Euler(x, dx_dt, dt):
    """ Performs the Euler method for numerical integration.
    Parameters:
    x : array-like, current state of the system
    dx_dt : array-like, derivative of the state
    dt : float, time step size

    Returns:
    x_new : array-like, updated state of the system after time step dt
    """
    x_new = x + dx_dt*dt
    
    return x_new

def dh_dt(h, m_h, params):
    """Computes the time derivative of Hes based on the model equations and the current state of the system
    Parameters:
    h: array containing current state of Hes for all cells
    m_h: array containing current state of Hes mRNA for all cells
    params: SimpleNamespace containing parameters of the model
    
    Returns:
    dh/dt : array containing time derivation of Hes for all cells
    """
    return m_h - params.gamma_h*h

def dmh_dt(m_h, h_delay, couple_component, params, lattice):
    """Computes the time derivative of Hes mRNA based on the model equations, the current state of the system, the type of coupling implemented and the delay of the system
    Parameters:
    m_h: array containing current state of Hes mRNA for all cells
    h_delay : array containing (delayed) state of Hes for all cells
    couple_component: array containing coupling component for all cells
    params: SimpleNamespace containing parameters of the model
    
    Returns:
    dmh/dt : array containing time derivation of Hes mRNA for all cells
    """
    return params.w_h * hill_function_negative(h_delay, int(params.l), params.p_h) + params.w_coupling * couple_component - params.gamma_m*m_h

def dd_dt(d, m_d, params):
    """Computes the time derivative of Delta based on the model equations and the current state of the system
    Parameters:
    m: array containing current state of Delta for all cells
    m_d: array containing current state of Delta mRNA for all cells
    params: SimpleNamespace containing parameters of the model
    
    Returns:
    dm/dt : array containing time derivation of Delta for all cells
    """
    return m_d - params.gamma_d*d

def dmd_dt(m_d, h_delay, params):
    """Computes the time derivative of Delta mRNA based on the model equations, the current state of the system, the type of coupling implemented and the delay of the system
    Parameters:
    m_d: array containing current state of Delta mRNA for all cells
    h_delay : array containing (delayed) state of Hes for all cells
    params: SimpleNamespace containing parameters of the model
    
    Returns:
    dmd/dt : array containing time derivation of Delta mRNA for all cells
    """
    return hill_function_negative(h_delay, params.l, params.p_h) - params.gamma_m*m_d

def d_neighbours(values, lattice):
    '''Calculates the values of the neighouring Delta for each cell in the lattice.
    Parameters: 
    values: the current values of the Delta variables in the lattice
    lattice: SimpleNamespace object containing the structur of the system
    
    Returns:
    d_neighbours: the values of the neighbouring Delta variables for each cell in the lattice'''

    # set the axis over which the sum is to be taken
    axis = 1

    d_neighbours = np.sum(lattice.connectivity * values.flatten() / lattice.w.flatten(), axis=axis)
    d_neighbours = d_neighbours.reshape(lattice.P, lattice.Q)

    return d_neighbours

def get_delayed_value(values_array, current_index, delay_steps, lattice, params):
    """Retrieves the value from a past step in the simulation, to compute delayed systems
    
    Parameters:
    values_array: array of values of which to take a delayed value
    current_index: index of the current time step
    delay_steps: array of timesteps to delay the value by (array for each value of P)
    lattice: SimpleNamespace containing the structure of the system
    
    Returns:
    values_delayed: array of values with the delayed value at the current index, for delayes that are too short, the value is set to 0
    """

    values_delayed = values_array.copy()

    # Calculate the delayed values of the current index for each P
    
    for p in range(lattice.P):
        delayed_index = max(0, current_index - delay_steps[p]) # If the delay is larger than the current index, set the value to 0
        values_delayed[current_index, p, :] = values_array[delayed_index, p, :]

    return values_delayed


def hill_function_positive(x, coeff, threshold):
    """Computes a value using the activation Hill function, dependent on value of x, threshold concentration and Hill coefficient.
    
    Parameters:
    - x: Value to be transformed by the Hill function.
    - coeff: Hill coefficient, which determines the steepness of the curve.
    - threshold: Threshold concentration, which determines the x-value at which the function is half-maximal.
    
    Returns:
    - Computed value based on the Hill function."""

    return x**coeff / (threshold**coeff + x**coeff)

def hill_function_negative(x, coeff, threshold):
    """Computes a value using the repressing Hill function, dependent on value of x, threshold concentration and Hill coefficient.
    
    Parameters:
    - x: Value to be transformed by the Hill function.
    - coeff: Hill coefficient, which determines the steepness of the curve.
    - threshold: Threshold concentration, which determines the x-value at which the function is half-maximal.
    
    Returns:
    - Computed value based on the Hill function."""
    return threshold**coeff / (threshold**coeff + x**coeff)

def nullclines_Hes(m_h, h, p_h, l, gamma_h, gamma_m):
    """Calculate the nullclines for Hes network for the internal oscillator equation.
    
    Inputs : 
    m_h: array of Hes mRNA concentrations
    h: array of Hes protein concentrations
    p_h: threshold concentration of Hes Hill function repression
    l: Hill function coefficient for Hes repression
    gamma_h: decay rate of Hes protein
    gamma_m: decay rate of Hes mRNA
    
    Returns:
    h_null: array of Hes protein nullcline values
    m_h_null: array of Hes mRNA nullcline values
    """
    # Hes protein nullcline
    h_null = gamma_h**(-1)*m_h
    
    # Hes mRNA nullcline
    m_h_null = gamma_m**(-1) * p_h**l / (p_h**l + h**l)

    return h_null, m_h_null

def nullclines_Hes_deltabath(m_h, h, D_ext, p_h, l, gamma_h, gamma_m):
    """Calculate the nullclines for Hes network for the internal oscillator with external Delta source equation.
    
    Inputs : 
    m_h: array of Hes mRNA concentrations
    h: array of Hes protein concentrations
    D_ext: external Delta source value
    p_h: threshold concentration of Hes Hill function repression
    l: Hill function coefficient for Hes repression
    gamma_h: decay rate of Hes protein
    gamma_m: decay rate of Hes mRNA
    
    Returns:
    h_null: array of Hes protein nullcline values
    m_h_null: array of Hes mRNA nullcline values
    """
    # Hes protein nullcline
    h_null = gamma_h**(-1)*m_h
    
    # Hes mRNA nullcline
    m_h_null = gamma_m**(-1) * p_h**l / (p_h**l + h**l) + gamma_m**(-1)*D_ext

    return h_null, m_h_null