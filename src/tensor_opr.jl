# TensorOperations.jl interface.
#

contract!(α,
          A::DualTensor{N, T}, conjA::Symbol,
          B::DualTensor{N, T}, conjB::Symbol,
          β,
          C::DualTensor{N, T}, 
          oindA::IndexTuple, cindA::IndexTuple, 
          oindB::IndexTuple, cindB::IndexTuple,
          indleft::IndexTuple,
          indright::IndexTuple, syms = nothing) where {N, T} =
    contract!(α,
              A, conjA, B, conjB,
              β, C,
              oindA, cindA, 
              oindB, cindB, 
              (indleft..., indright...), syms)

contract!(α,
          A::DualTensor{N, T}, conjA::Symbol,
          B::DualTensor{N, T}, conjB::Symbol,
          β,
          C::DualTensor{N, T}, 
          oindA::IndexTuple, cindA::IndexTuple, 
          oindB::IndexTuple, cindB::IndexTuple,
          tindC::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing) where {N, T} = begin

    # Value part.
    contract!(α,
              A.A, conjA, B.A, conjB,
              β, C.A,
              oindA, cindA,
              oindB, cindB,
              tindC, syms)

    # Loop over derivatives.
    # ∂A part.
    for i=1:N
        contract!(α,
                  A.∂[i], conjA, B.A, conjB,
                  β, C.∂[i],
                  oindA, cindA,
                  oindB, cindB,
                  tindC, syms)
    end
    # ∂B part.
    for i=1:N
        contract!(α,
                  A.A, conjA, B.∂[i], conjB,
                  true, C.∂[i],
                  oindA, cindA,
                  oindB, cindB,
                  tindC, syms)
    end

    C
end

