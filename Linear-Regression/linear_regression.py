import numpy as np
import matplotlib.pyplot as plt

def get_data(point_number):
    data = []
    for i in range(point_number):
        x = np.random.uniform(-10.0, 10.0)         # Draw samples from a uniform distribution
       	# mean = 0 (centre), std = 0.1 (width)  
       	eps = np.random.normal(0.0, 0.01)           # Draw random samples from a normal (Gaussian) distribution.
       	y = 1.477 * x + 0.089 + eps
       	data.append([x, y])
      	
    data = np.array(data)
    return data
    
    # Plot data
    # plt.plot(data[:,0], data[:,1], 'or')
    # plt.show() 

# ******************* Compute Loss function *****************
# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # computer mean-squared-error
        totalError += ((w * x + b) - y) ** 2
    # average loss for each point
    return totalError / float(len(points))

# ******************* Compute gradient and update *****************
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(wx+b-y)
        b_gradient += ((2 * ((w_current * x + b_current) - y)) / N
        # grad_w = 2(wx+b-y)*x
        w_gradient += ((2 * x * ((w_current * x + b_current) - y)) / N
    # update w'
    new_b = b_current - (learningRate * b_gradient)        # learningRate = Step length
    new_w = w_current - (learningRate * w_gradient)
    
    return [new_b, new_w]

# ******************* Set w = w' and loop *****************
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
	
    b = starting_b
    w = starting_w
    # update for several times
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]    

# ********************* Run function *******************

def run(points):
	
    # points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.001                         # learningRate = Step length
    initial_b = 0                                 # initial y-intercept guess
    initial_w = 0                                 # initial slope guess
    num_iterations = 10000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
          
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )
    
if __name__ == '__main__':
   
   data = get_data(10000)
   run(data)