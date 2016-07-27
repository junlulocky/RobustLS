import numpy as np

def compute_ls_solution(x, t, p=0):
    """ Computes the solution for the optimization problem

      min_{W, b} 1/2K*\sum_{k=1}^K \lVert W*\tilde{x}_k + b - t_k \rVert^2

    where \tilde{x} is a the input samples x corrupted by binary masking noise
    (aka Dropout) with probability of corruption p.

    The solution is similar to the normal LS solution but with a
    weighting of C_xx and a scaling of W by (1-p)

    Input :
    -------
    `x` : contains inputs of shape `K x d1 x d2 x ...`. `K` gives the number
          of samples.
    `t` : contains targets of shape `K x d1' x d2' x ...`
    `p` : the corruption probability - when p=0, classical LS

    Output : Returns a list [W, b, r]
    --------
    `W` : weights of shape `d1 x d2 x ... x d1' x d2' x ...`
    `b` : bias of shape `d1' x d2' x ...`
    `r` : residual error
    """
    num_samples = x.shape[0]
    inshape = x.shape[1:]
    outshape = t.shape[1:]

    # Reshape into Matlab-style: `Features x Samples`
    x = np.transpose(np.reshape(x, (num_samples, np.prod(inshape)), 'F'))
    t = np.transpose(np.reshape(t, (num_samples, np.prod(outshape)), 'F'))

    # Compute statistics of input/target vectors
    mu_t = np.mean(t, axis=1, keepdims=True)
    mu_x = np.mean(x, axis=1, keepdims=True)
    print 't has shape: {}'.format(t.shape)
    print 'mu_t has shape: {}'.format(mu_t.shape)
    print 'x has shape: {}'.format(x.shape)
    print 'mu_x has shape: {}'.format(mu_x.shape)

    C_tx = np.dot(t - mu_t, np.transpose(x - mu_x)) / num_samples
    C_xx = np.dot(x - mu_x, np.transpose(x - mu_x)) / num_samples
    C_tt = np.dot(t - mu_t, np.transpose(t - mu_t)) / num_samples

    # If we use dropout-like corruption, then we need to adapt the data
    # statistics here
    if p > 0:
        C_tx = (1-p) * C_tx
        A = np.identity(np.prod(inshape))*p*(1-p) + (1-p)**2
        C_xx = A * (C_xx + np.dot(mu_x, np.transpose(mu_x))) \
          - (1-p)**2 * np.dot(mu_x, np.transpose(mu_x))
        mu_x = (1-p) * mu_x

    # Compute pseudo-inverse of C_xx
    # (using `np.linalg.eigh` is faster than `np.linalg.pinv`)
    d, u = np.linalg.eigh(C_xx)
    cutoff = 1e-15 * np.maximum.reduce(d)
    for i in range(np.prod(inshape)):
        if d[i] > cutoff:
            d[i] = 1./d[i]
        else:
            d[i] = 0.0

    # Compute weight and bias term
    W = np.dot(C_tx, np.dot(np.dot(u, np.diag(d)), u.transpose()))
    b = mu_t - np.dot(W, mu_x)

    # compute residual error
    r = 0.5 * np.trace(np.reshape(C_tt - np.dot(W, np.transpose(C_tx)),
                                (np.prod(outshape), np.prod(outshape)), 'F'))

    # If we use dropout-like corruption, then we re-scale the weight matrix here
    # to match the first-moment of the data
    if p > 0:
        W = W * (1-p)

    # reshape parameters into correct size
    W = np.reshape(np.transpose(W), inshape + outshape, 'F')
    b = np.reshape(b, (outshape), 'F')

    return [W, b, r]

def mainTest():
    #### load data
    X = np.load('data/X.npy')
    y = np.load('data/y.npy')

    p = 0.025

    [W, b, r] = compute_ls_solution(X, y, p)
    print "Residual: {}".format(r)


if __name__ == '__main__':
    mainTest()
