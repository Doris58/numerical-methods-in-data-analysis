function [topLeft, bottomRight] = extractNeighborhood(patchCenters, numPatchesinDim)
    topMost    = patchCenters(1,:);
    bottomMost = patchCenters(end,:);

    % For each patch center, compute square neighborhood boundaries
    numPatches = size(patchCenters, 1);
    radiusVec = repmat([numPatchesinDim, numPatchesinDim], numPatches, 1);
    topLeft = patchCenters - radiusVec;
    bottomRight = patchCenters + radiusVec;

    addToBottomRight = max(0, 1 - topLeft);

    subFromTopleft(:,1) = max(0, bottomRight(:,1) - bottomMost(1));
    subFromTopleft(:,2) = max(0, bottomRight(:,2) - bottomMost(2));

    topLeft(:,1) = min(max(topLeft(:,1), topMost(1)), bottomMost(1));
    topLeft(:,2) = min(max(topLeft(:,2), topMost(2)), bottomMost(2));

    bottomRight(:,1) = min(max(bottomRight(:,1), topMost(1)), bottomMost(1));
    bottomRight(:,2) = min(max(bottomRight(:,2), topMost(2)), bottomMost(2));

    topLeft = topLeft - subFromUpleft;
    bottomRight = bottomRight + addToBottomRight;

end




