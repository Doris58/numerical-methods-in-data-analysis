function detection = anomalyDetection(full_diffusion_map, ...
    patchIndices, patchCenters, upLeft, bottomRight, imageLinIndices, sigma_bar)

nPatches = size(upLeft, 1);
similarity = zeros(nPatches, 1);
% ovo je vektor-stupac
normDiffusionCoords = sum(full_diffusion_map.^2, 2);  % squared norms of embeddings

% Define mask size as ~20% of window width
maskDim = max(2, ceil(0.2 * (bottomRight(1,1) - upLeft(1,1) + 1) / 2));

parfor i = 1 : nPatches

    rows = upLeft(i,1):bottomRight(i,1);
    cols = upLeft(i,2):bottomRight(i,2);

    mask = true(length(rows), length(cols));


    center_row = patchCenters(i,1) - upLeft(i,1) + 1;
    center_col = patchCenters(i,2) - upLeft(i,2) + 1;

    row_range = max(center_row - maskDim, 1):min(center_row + maskDim, length(rows));
    col_range = max(center_col - maskDim, 1):min(center_col + maskDim, length(cols));
    mask(row_range, col_range) = false;


    winIdx = imageLinIndices(rows, cols);
    winIdx = winIdx(mask);
    winIdx = winIdx(winIdx ~= 0);


    [~, winIdxPatches, ~] = intersect(patchIndices, winIdx);


    % ovo ispod je matrica
    diffusionMapCoords = full_diffusion_map(winIdxPatches, :);
    % ovo je vektor-redak! ==> transp. je vektor-stupac!
    patchCoords = full_diffusion_map(i, :);

    % srednji clan ovdje je vektor-stupac!
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
