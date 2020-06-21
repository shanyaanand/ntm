
module con
    using Distributions
    using Turing, Flux
	function unpack_controller(hidden_size, input_size, num_cols, nn_params::AbstractVector)

	    W1 = reshape(nn_params[1:hidden_size * (hidden_size + input_size + num_cols)], hidden_size, (hidden_size + input_size + num_cols));   
	    b1 = reshape(nn_params[hidden_size + input_size + num_cols + 1:hidden_size + input_size + num_cols + hidden_size], hidden_size)
	 
	    return W1, b1
	end
	export unpack_controller

	function controller(xt, ht_1, rt_1, hidden_size, input_size, num_cols, nn_params::AbstractVector)
		"""
			Args : xt - current input vector
				   ht_1 - previous hidden state
				   rt_1 - previous read
				   hidden_size - dimension of hidden layer 
				   input_size - dimension of input vector
				   num_cols - number of columns in the memory block/matrix 

			Ops : controller takes the concatenation of the current inputs xt and the previous read-vectors rt_1 as input and returns a hidden state h of shape (hidden_size)
		"""
	    W1, b1 = unpack_controller(hidden_size, input_size, num_cols, nn_params)
	    nn = Chain(Dense(W1, b1, σ))
	    return nn([xt;ht_1;rt_1])
	end
	export controller
end