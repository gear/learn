function C = FactorMat( A )
% FactorMat Function creates a matrix of assignment and value for better
% visualization
C = [IndexToAssignment(1:prod(A.card), A.card) A.val'];
end

