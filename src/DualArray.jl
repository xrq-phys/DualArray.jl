module DualArray

import TensorOperations: IndexTuple
import TensorOperations: contract!
import ForwardDiff: Dual
import ForwardDiff

struct DualTensor{N, T}
    A::Array{T}
    ∂::NTuple{N, Array{T}}

    "Construct an instance of `DualTensor` from `Array` components."
    DualTensor(A::Array{T},
               ∂::NTuple{N, Array{T}}) where {N, T} = begin
        all(map(sz -> sz == size(A), size.(∂))) ||
            throw(DimensionMismatch("Tensor data and its derivatives must have the same size."))

        new{N, T}(A, ∂)
    end
end

"Convert `Array` of `Dual`s into `DualTensor`."
DualTensor(A::Array{Dual{Tag, T, N}}) where {N, T, Tag} = begin
    AA = ForwardDiff.value.(A)
    ∂A = (( (x -> x[i∂]).(ForwardDiff.partials.(A)) for i∂=1:N )...,)

    DualTensor(AA, ∂A)
end

include("tensor_opr.jl")

end
