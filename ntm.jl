
module ntm

    using Distributions
    using Turing, Flux
	include("./con.jl")
	using .con
	include("./heads.jl")
	using .heads
    function NTM(xs, input_size, hidden_size, max_shift, num_rows, num_cols, nn_params_con::AbstractVector, nn_params_read::AbstractVector, nn_params_write::AbstractVector, nn_params_mem::AbstractVector, nn_params_fc::AbstractVector)
    	"""
    		Args : xs - input data
    			   input_size - dimension of input data vector
    			   hidden_size - dimension of hidden layer
    			   max_shift - allowed maximum shift while Circular Shift
    			   num_rows, num_cols - dimension of memory block
    			   nn_params_*** - weights and bias of networks

    		Ops :  encapsulate all the modules
    	"""
    	# Copy data
        ht_1 = ones(hidden_size)
        rt_1 = ones(num_cols)
        prev_wt = ones(num_rows)    
        for j = length(xs)
            xt = xs[j]
            ht = controller(xt, ht_1, rt_1, hidden_size, input_size, num_cols, nn_params_con)
            ht_ = readhead(ht, hidden_size, num_cols, max_shift, nn_params_read)
            rt_1, prev_wt = reading(ht_, max_shift, num_rows, num_cols, prev_wt, nn_params_mem)
            mem = writehead(ht, hidden_size, prev_wt, num_rows, num_cols, max_shift, nn_params_write, nn_params_mem)
            ht_1 = ht 
        end

        # Perform recovery 
        cp = []
        ht_1 = ones(hidden_size)
        rt_1 = ones(num_cols)
        prev_wt = ones(num_rows)    

        for i = 1:length(xs)
            x = zeros(input_size) 
            ht = controller(x, ht_1, rt_1, hidden_size, input_size, num_cols, nn_params_con)
            ht_ = readhead(ht, hidden_size, num_cols, max_shift, nn_params_read)
            rt_1, prev_wt = reading(ht_, max_shift, num_rows, num_cols, prev_wt, nn_params_mem)
            ht_1 = ht
            ot = fc(rt_1, input_size, num_cols, nn_params_fc)
            append!(cp, ot) 
        end    
        return reshape(cp, 1, length(xs))
    end
    export NTM
end