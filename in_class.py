import numpy as np

N = 100
K = 3  # this would be 784 for 28x28 images

data = np.random.randn( K, N )

X = np.random.randn( K, 1 )

# ss = 1.0

sigma = np.eye( K ) # gives you an identity matrix

prob = 0.0

# c = (2 * pi) ^ (-k / 2)

for i in range( N ):
    mu_i = data[:, i:i+1] # you do i:i+1 instead of just i so that it will
    # keep the 2 dimensionality for linear algebra purposes

    # diff = np.zeros( K, 1 )
    # for d in range( K ):
    #     diff[d] = x[d] - mu_i[d]
    # This is the bad way to do it

    diff = X - mu_i

    prob += c * np.exp( -0.5 * np.dot( diff.T, np.dot( sigma, diff ) ) )
    # p(x | mu, sigma) = 2pi^-(k/2) * sigma * exp( -1/2 * (x - mu)T * sigma(x - mu))

prob /= N
 # The above code is pretty good, but it's still slow

# Because one of the dimensions is one, np knows to subtract the vector
# from each column in the matrix
diff = X - data

# np.dot( sigma, diff )

# np.dot( diff.T, np.dot( sigma, diff ) ) # this works
# but it calculates and nxn matrix with a lot of values we don't need

# diff * np.dot( sigma, diff )
# This pointwise multiplies each column in diff by each column
# in the sigma diff matrix

result = np.sum( diff * np.dot( sigma, diff ), axis=0, keepdims=True )
# the keepdims keeps the original size of the matrix because otherwise np
# will auto get rid of dimensions it thinks you don't want/need

result = np.exp( -0.5 * result )
np.mean( result )
