function detection = anomalyDetection(full_diffusion_map, ...
    patchIndices, patchCenters, upLeft, bottomRight, imageLinIndices, sigma_bar)
% Row-based anomaly detection using spatial neighborhoods in diffusion space
% Each row of full_diffusion_map is a patch embedding
% Each column is a diffusion coordinate

nPatches = size(upLeft, 1);
similarity = zeros(nPatches, 1);
normDiffusionCoords = sum(full_diffusion_map.^2, 2);  

maskDim = max(2, ceil(0.2 * (bottomRight(1,1) - upLeft(1,1) + 1) / 2));


parfor i = 1 : nPatches
    rows = upLeft(i,1):bottomRight(i,1);
    cols = upLeft(i,2):bottomRight(i,2);


    mask = true(length(rows), length(cols));

    masked_inds = [patchCenters(i,1) + [-maskDim; maskDim], ...
                       patchCenters(i,2) + [-maskDim; maskDim]];
    masked_inds = max(masked_inds, 1);
    masked_inds(:,1) = min(masked_inds(:,1), max(bottomRight(:,1)));
    masked_inds(:,2) = min(masked_inds(:,2), max(bottomRight(:,2)));
    masked_inds = masked_inds - [upLeft(i,:); upLeft(i,:)] + 1;
    mask(masked_inds(1,1):masked_inds(2,1), masked_inds(1,2):masked_inds(2,2)) = false;

    winIdx = imageLinIndices(rows, cols) .* mask;
    

    winIdx = winIdx(winIdx ~= 0);
    [~, winIdxPatches, ~] = intersect(patchIndices, winIdx);

    diffusionMapCoords = full_diffusion_map(winIdxPatches, :);  
    patchCoords = full_diffusion_map(i, :);                     

    dists = normDiffusionCoords(winIdxPatches) ...
          - 2 * (diffusionMapCoords * patchCoords') ...
          + normDiffusionCoords(i);

    W = exp(-dists / (sigma_bar + eps));
    W(W < 0.5) = 0;

    if all(W == 0)
        similarity(i) = 0;
    else
        similarity(i) = mean(W);
    end
end

detection = 1 - similarity;

end
