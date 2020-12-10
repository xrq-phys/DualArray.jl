# Test error-handling.
#

@testset "error-handling" begin
    @testset "in DualTensor creation" begin
        try
            DualTensor( rand(3, 3), (rand(3, 3), (rand(3, 2))) )
        catch err
            @test err isa DimensionMismatch
        end
    end
end

