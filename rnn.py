import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def rnn_cell_forward_prop(xt, a_prev, params):
    """
    Single forward step of a RNN-cell (without LSTM or GRU).

    Arguments:
    xt -- input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    # Retrieve the weights and biases
    Wax, Waa, Wya = params["Wax"], params["Waa"], params["Wya"]
    ba, by = params["ba"], params["by"]

    # compute next activation state and output of the current cell
    a_next = np.tanh(Waa.dot(a_prev)+Wax.dot(xt)+ba)
    yt_pred = softmax(Wya.dot(a_next)+by)

    # store values needed for backward propagation in cache
    cache = (a_next, a_prev, xt, params)

    return a_next, yt_pred, cache




def rnn_forward_prop(x, a0, params):
    """
    Forward propagation of the RNN.

    Arguments:
    x -- input data for every time-step, of shape (n_x, m, T_x).
    a0 -- initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """

    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = params["Wya"].shape


    # initialize "a" and "y_pred" with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next
    a_next = a0

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        a_next, yt_pred, cache = rnn_cell_forward_prop(x[:,:,t], a_next, params)
        # Save the value of the new "next" hidden state in "a"
        a[:,:,t] = a_next
        # Save the value of the prediction in "y"
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches"
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches







def lstm_cell_forward_prop(xt, a_prev, c_prev, params):
    """
    Single forward step of a LSTM-cell.

    Arguments:
    xt -- input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    # Retrieve the weights and biases
    Wf, Wi, Wc, Wo, Wy = params["Wf"], params["Wi"], params["Wc"], params["Wo"], params["Wy"]
    bf, bi, bc, bo, by = params["bf"], params["bi"], params["bc"], params["bo"], params["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.zeros((n_a+n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next
    ft = sigmoid(Wf.dot(concat)+bf)
    it = sigmoid(Wi.dot(concat)+bi)
    cct = np.tanh(Wc.dot(concat)+bc)
    c_next = ft*c_prev+it*cct
    ot = sigmoid(Wo.dot(concat)+bo)
    a_next = ot*np.tanh(c_next)

    # Compute prediction of the LSTM cell
    yt_pred = softmax(Wy.dot(a_next)+by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, params)

    return a_next, c_next, yt_pred, cache


def lstm_forward_prop(x, a0, params):
    """
    Forward propagation of the RNN using an LSTM-cell.

    Arguments:
    x -- input data for every time-step, of shape (n_x, m, T_x).
    a0 -- initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy']
    n_x, m, T_x = x.shape #V
    n_y, n_a = params['Wy'].shape #V

    # initialize "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next
    a_next = a0
    c_next = np.zeros((n_a, m))

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache
        a_next, c_next, yt, cache = lstm_cell_forward_prop(x[:,:,t], a_next, c_next, params)
        # Save the value of the new "next" hidden state in "a"
        a[:,:,t] = a_next
        # Save the value of the prediction in "y"
        y[:,:,t] = yt
        # Save the value of the next cell state
        c[:,:,t]  = c_next
        # Append the cache into caches
        caches.append(cache)


    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches







# RNN test with expected values
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
params = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward_prop(xt, a_prev, params)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape)

#a_next[4] =  [ 0.59584544  0.18141802  0.61311866  0.99808218  0.85016201  0.99980978
# -0.18887155  0.99815551  0.6531151   0.82872037]
#a_next.shape =  (5, 10)
#yt_pred[1] = [ 0.9888161   0.01682021  0.21140899  0.36817467  0.98988387  0.88945212
#  0.36920224  0.9966312   0.9982559   0.17746526]
#yt_pred.shape =  (2, 10)

print('\n\n')

np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward_prop(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))

#a[4][1] =  [-0.99999375  0.77911235 -0.99861469 -0.99833267]
#a.shape =  (5, 10, 4)
#y_pred[1][3] = [ 0.79560373  0.86224861  0.11118257  0.81515947]
#y_pred.shape =  (2, 10, 4)
#caches[1][1][3] = [-1.1425182  -0.34934272 -0.20889423  0.58662319]
#len(caches) =  2



print('\n\n\n')

# LSTM test with expected values
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy,
    "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward_prop(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))

#a_next[4] =  [-0.66408471  0.0036921   0.02088357  0.22834167 -0.85575339  0.00138482
#  0.76566531  0.34631421 -0.00215674  0.43827275]
#a_next.shape =  (5, 10)
#c_next[2] =  [ 0.63267805  1.00570849  0.35504474  0.20690913 -1.64566718  0.11832942
#  0.76449811 -0.0981561  -0.74348425 -0.26810932]
#c_next.shape =  (5, 10)
#yt[1] = [ 0.79913913  0.15986619  0.22412122  0.15606108  0.97057211  0.31146381
#  0.00943007  0.12666353  0.39380172  0.07828381]
#yt.shape =  (2, 10)
#cache[1][3] = [-0.16263996  1.03729328  0.72938082 -0.54101719  0.02752074 -0.30821874
#  0.07651101 -1.03752894  1.41219977 -0.37647422]
#len(cache) =  10

print('\n\n')

np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy,
    "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward_prop(x, a0, parameters)
print("a[4][3][6] = ", a[4][3][6])
print("a.shape = ", a.shape)
print("y[1][4][3] =", y[1][4][3])
print("y.shape = ", y.shape)
print("caches[1][1[1]] =", caches[1][1][1])
print("c[1][2][1]", c[1][2][1])
print("len(caches) = ", len(caches))

#a[4][3][6] =  0.172117767533
#a.shape =  (5, 10, 7)
#y[1][4][3] = 0.95087346185
#y.shape =  (2, 10, 7)
#caches[1][1[1]] = [ 0.82797464  0.23009474  0.76201118 -0.22232814 -0.20075807  0.18656139
#  0.41005165]
#c[1][2][1] -0.855544916718
#len(caches) =  2
