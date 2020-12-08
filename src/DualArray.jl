module DualArray

import TensorOperations: IndexTuple
import TensorOperations: contract!

# TODO: Assert size consistency.
struct DualTensor{N, T}
    A::Array{T}
    ∂::NTuple{N, Array{T}}
end

include("tensor_opr.jl")

end
