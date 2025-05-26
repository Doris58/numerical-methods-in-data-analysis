function [upLeft, bottomRight] = extractNeighborhood(patchCenters, numPatchesinDim)
% CREATEEMBEDDINGIMAGE Computes image-space windows for patch-based similarity.
% INPUTS:
%   patchCenters      - Matrix of size (numPatches x 2), each row is [row, col]
%   numPatchesinDim   - Scalar; half window size around each center
%
% OUTPUTS:
%   upLeft       - Top-left corner of each spatial window
%   bottomRight  - Bottom-right corner of each spatial window

    topMost    = patchCenters(1,:);
    bottomMost = patchCenters(end,:);

    % For each patch center, compute square neighborhood boundaries
    numPatches = size(patchCenters, 1);
    radiusVec = repmat([numPatchesinDim, numPatchesinDim], numPatches, 1);
    upLeft = patchCenters - radiusVec;
    bottomRight = patchCenters + radiusVec;

    addToBottomRight = max(0, 1 - upLeft); 
    subFromUpleft(:,1) = max(0, bottomRight(:,1) - bottomMost(1));
    subFromUpleft(:,2) = max(0, bottomRight(:,2) - bottomMost(2));

    upLeft(:,1) = min(max(upLeft(:,1), topMost(1)), bottomMost(1));
    upLeft(:,2) = min(max(upLeft(:,2), topMost(2)), bottomMost(2));
    bottomRight(:,1) = min(max(bottomRight(:,1), topMost(1)), bottomMost(1));
    bottomRight(:,2) = min(max(bottomRight(:,2), topMost(2)), bottomMost(2));

    upLeft = upLeft - subFromUpleft;
    bottomRight = bottomRight + addToBottomRight;

end