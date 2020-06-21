
module utils
    using Distributions
    using Turing, Flux
    function focus_head(k, beta, mem, prev_wt, g, s, gamma)
    	"""
    		Ops : find the weight
    	"""
        Wc = content_weight(k, beta, mem)
        Wg = gated_interpolation(Wc, prev_wt, g)
        W = sharpen(Wg, gamma)
        return W
    end
    export focus_head

    function content_weight(k, beta, mem)
    	"""
    		Ops : measure the similarity between k and memory block/matrix
    	"""

        mem_norm = Flux.normalise(mem, dims = 2)
        k_norm = Flux.normalise(k)
        similarity_scores = (mem_norm * k_norm)
        Wc = softmax(beta .* similarity_scores)
        return Wc
    end
    export content_weight

    function gated_interpolation(Wc, prev_wt, g)
    	"""
    		Ops : Gated interpolation controls to what extent the content-based addressing mechanism should be used
    	"""
        Wg = g*Wc + (1 - g)*prev_wt
        return Wg    

    end
    export gated_interpolation

    function sharpen(Wg, gamma)
        W = Wg
        W = W ./( sum(W) .+ 0.000001) 
        return W    
    end
    export sharpen
end        