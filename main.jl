using Distributions
using Turing, Flux
include("./ntm.jl")
using .ntm


# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)

"""						  										   ---> fully connected write head network (hidden size X output size)
															       |	
	xt(1 X input size)---> controller (hidden size X input size)---|
						 										   |
																   ---> fully connected write head network (hidden size X output size)
"""

# Specify the probabalistic model.
@model bayes_nn(xs, ts, input_size, hidden_size, max_shift, num_rows, num_cols) = begin
    """
    	Args : xs - input data
    		   ts - label data
    		   input_size - dimension of input data vector
    		   hidden_size - dimension of hidden layer
    		   max_shift - allowed maximum shift while Circular Shift
    		   num_rows, num_cols - dimension of memory block
    """
	# Create the weight and bias vector.
    nn_params_con ~ MvNormal(zeros(hidden_size * (hidden_size + input_size + num_cols) + hidden_size), sig .* ones(hidden_size * (hidden_size + input_size + num_cols) + hidden_size))

    output_size_rd = num_cols+3+2*max_shift+2
    nn_params_read ~ MvNormal(zeros(hidden_size * output_size_rd + output_size_rd), sig .* ones(hidden_size * output_size_rd + output_size_rd))

    nn_params_mem ~ MvNormal(zeros(num_rows * num_cols), sig .* ones(num_rows * num_cols))

    output_size_wt = num_cols+3+2*max_shift+3+num_cols+1+num_cols
    
    nn_params_write ~ MvNormal(zeros(hidden_size * output_size_wt + output_size_wt), sig .* ones(hidden_size * output_size_wt + output_size_wt))
    
    nn_params_fc ~ MvNormal(zeros(num_cols*input_size + input_size), sig .*ones(num_cols*input_size + input_size))
    
    # Calculate predictions for the inputs given the weights
    # and biases in theta.
    
    preds = NTM(xs, input_size, hidden_size, max_shift, num_rows, num_cols, nn_params_con, nn_params_read, nn_params_write, nn_params_mem, nn_params_fc)
    
    # Observe each prediction.
    for i = 1:length(ts)
            ts[i] ~ Bernoulli(preds[i])
    end
    
    
end;


input_size=1
hidden_size = 10

# Generating binary data.
p_true = 0.5
Ns = 0:10;
data = rand(Bernoulli(p_true), last(Ns))

# Perform inference.
N = 1
ch = sample(bayes_nn(data, data, input_size, hidden_size, 2, 32, 32), HMC(0.05, 4), N);

# Extract all weight and bias parameters.
theta_con = ch[:nn_params_con].value.data;
theta_read = ch[:nn_params_read].value.data;
theta_write = ch[:nn_params_write].value.data;
theta_mem = ch[:nn_params_mem].value.data;
theta_fc = ch[:nn_params_fc].value.data;

# Find the index that provided the highest log posterior in the chain.
_, i = findmax(ch[:lp].value.data)

# Extract the max row value from i.
i = i.I[1]


Z = NTM(data, 1, hidden_size, 2, 4, 5, theta_con[i, :], theta_read[i, :], theta_write[i, :], theta_mem[i, :], theta_fc[i, :])
println(data)
for z in Z
	println(z)
end 
