
module heads
	include("./utils.jl")
	using .utils

    using Distributions
    using Turing, Flux
    function unpack_readhead(hidden_size, output_size, nn_params::AbstractVector)

        W1 = reshape(nn_params[1:hidden_size * output_size], output_size, hidden_size);   
        s = (hidden_size * output_size) + 1 
        e = (hidden_size * output_size) + output_size
        b1 = reshape(nn_params[s:e], output_size)
     
        return W1, b1
    end
    export unpack_readhead

    function readhead(ht, hidden_size, num_cols, max_shift, nn_params::AbstractVector)
    	"""
    		Ops : extract instructions from controller's output
    	"""
        output_size = num_cols+3+2*max_shift+2
        W1, b1 = unpack_readhead(hidden_size, output_size, nn_params)
        nn = Chain(Dense(W1, b1, σ))###
        return nn(ht)
    end 
    export readhead

    function reading(ht, max_shift, num_rows, num_cols, prev_wt, nn_params::AbstractVector)
        """
        	Args : ht - hidden layer output(instriction from controller)
        		   max_shift - maximum shift allow in circular shifting 
        		   num_rows, num_cols - dimensions of memory block/matrix
        		   prev_wt - previous step final weight

        	Ops : access memory block/matrix
        """
        k, beta, g, s, gamma_ = ht[1:num_cols], ht[num_cols+1], ht[num_cols+2], ht[num_cols+3:num_cols+2+2*max_shift+1], ht[num_cols+3+2*max_shift+1]
        gamma = 1.001 + relu(gamma_)
        mem = reshape(nn_params[1:num_rows * num_cols], num_rows, num_cols);
        W = focus_head(k, beta, mem, prev_wt, g, s, gamma)
        rt = transpose(mem) * W
        return rt, W
    end
    export reading

    function erase(W, e, mem, rows, cols)

        temp = (ones(rows, cols) - W.*e')
        return mem .*temp
    end
    export erase

    function write(mem, W, a)
        return mem .+ W .* a'
    end
    export write

    function unpack_writehead(hidden_size, output_size, nn_params::AbstractVector)

        W1 = reshape(nn_params[1:hidden_size * output_size], output_size, hidden_size);   
        b1 = reshape(nn_params[hidden_size * output_size + 1 : hidden_size * output_size + output_size], output_size)
     
        return W1, b1
    end
    export unpack_writehead

    function writehead(h, hidden_size, prev_wt, num_rows, num_cols, max_shift, nn_params::AbstractVector, nn_params_mem::AbstractVector)
    	"""
    		Ops : calculate weightings w, erase vectors e and add vectors a and updates the memory contents
    	"""
        output_size = num_cols+3+2*max_shift+3+num_cols+1+num_cols
        W1, b1 = unpack_writehead(hidden_size, output_size, nn_params)
        nn = Chain(Dense(W1, b1, σ))
        ht =  nn(h)  
        mem = reshape(nn_params_mem[1:num_rows * num_cols], num_rows, num_cols);
        k, beta, g, s, gamma_, e, a = ht[1:num_cols], ht[num_cols+1], ht[num_cols+2], ht[num_cols+3:num_cols+2+2*max_shift+1], ht[num_cols+2+2*max_shift+2], ht[num_cols+2+2*max_shift+3:num_cols+2+2*max_shift+2+num_cols], ht[num_cols+2+2*max_shift+2+num_cols+1:num_cols+2+2*max_shift+2+num_cols+num_cols]
        gamma = 1 + relu(gamma_)
        W = focus_head(k, beta, mem, prev_wt, g, s, gamma)
        mem = erase(W, e, mem, num_rows, num_cols)
        mem = write(mem, W, a)
    end
    export writehead

    function unpack_fc(num_cols, input_size, nn_params::AbstractVector)
        W1 = reshape(nn_params[1:num_cols*input_size], input_size, num_cols);   
        b1 = reshape(nn_params[num_cols*input_size + 1 : num_cols*input_size + input_size], input_size)
        return W1, b1

    end
    export unpack_fc

    function fc(rt , input_size, num_cols, nn_params::AbstractVector)
        W, b = unpack_fc(num_cols, input_size, nn_params)
        nn = Chain(Dense(W, b, σ))
        return nn(rt)
    end
    export fc

end 