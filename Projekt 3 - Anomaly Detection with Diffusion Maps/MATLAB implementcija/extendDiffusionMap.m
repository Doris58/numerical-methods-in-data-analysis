function full_diffusion_map = extendCoordinates_LP(X_subset, X, diffusion_map, Dist_squared_subset, nonTrivialIndices_subset, sigma0, maxLevels, tolerance)
    % EXTENDCOORDINATES_LP Extends diffusion coordinates from subset to full set
    % using the Laplacian Pyramid extension method.
    %
    % INPUTS:
    %   X_subset        - Sampled subset (m x d)
    %   X_full          - Full dataset (n x d)
    %   subsetEmbedding - Diffusion embedding of subset (m x k)
    %   sigma0          - Initial kernel scale (scalar)
    %   maxLevels       - Maximum number of pyramid levels
    %   tolerance             - Tolerance for residual norm
    %
    % OUTPUT:
    %   fullEmbedding   - Extended embedding (n x k)
    
    if nargin < 6
        sigma0 = 25;
        %sigma0 = estimateSigma(X_subset);
    end
    if nargin < 7
        % maksimalni broj iteracija / razina Laplaceove piramide
        maxLevels = 10;
    end
    if nargin < 8
        tolerance = 1e-4;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % racunanje rijetke matrice (kvadrata) udaljenosti 
    % do KNN najblizih susjeda u X_subset
    % za svaki patch iz X 
    % ===> sparse matrica Dist_squared
   
    numPatches = size(X, 1);
    numSamples = size(X_subset, 1);

    % naci kNN najblizih susjeda (+ sebe)
    % za svaki patch iz X medu patch-evima u X_subset
    [Idx, Dist] = knnsearch(X_subset, X, 'K', kNN + 1); 

    % broj nenul elemenata rijetke matrice tezina
    numNonTrivial = numPatches * (kNN + 1);

    % alokacija memorije za tri vektora za sparse matricu 
    rowIndices = zeros(1, numNonTrivial);
    colIndices = zeros(1, numNonTrivial);
    distances = zeros(1, numNonTrivial);

    % popunjavanje tih vektora - bilo kojim redoslijedom, samo da si medusobno odgovaraju,
    % tj da je redoslijed isti za sva tri vektora
    i = 1;
    % na ovaj nacin - da iskoristimo strukturu matrica Idx i Dist
    for patchIndex = 1 : numPatches

        rowIndices(i : i + kNN) = Idx(patchIndex, :); 
        colIndices(i : i + kNN) = patchIndex;
        distances(i : i + kNN) = Dist(patchIndex, :);

        i = i + (kNN + 1);

    end

    distances = distances .^2;

    Dist_squared = sparse(rowIndices, colIndices, distances, numSamples, numPatches);

    nonTrivialIndices = sub2ind( [numSamples, numPatches] , rowIndices, colIndices);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % broj koordinata difuzijskog preslikavanja
    % = broj stupaca matrice dif. presl.
    newDim = size(diffusion_map, 2);
   
    % alokacija matrice za difuzijsko preslikavanje na cijelom dataset-u
    % tj. na svim patch-evima
    full_diffusion_map = zeros(numPatches, newDim);

    % prosirujemo koordinatu po koordinatu (tj. komponentu) difuzijskog preslikavanja
    for dmCoord = 1 : newDim
        % funkcija koju prosirujemo = koordinata dif. presl. = odgovarajuci stupac matrice dif. presl.
        f = diffusion_map(:, dmCoord);
        f_approx = zeros(numSamples, 1);
        % alokacija vektora za prosirenje te koordinate
        f_ext = zeros(numPatches, 1);

        %sigma = sigma0;
        
        for level = 1 : maxLevels

            d_l = f - f_approx; % rezidual
                
            % POPRAVITI OVO !!
            if norm( d_l ) < tolerance 
                if level == 1
                    [rowIndices, colIndices] = ind2sub([ numSamples, numPatches ], nonTrivialIndices);
                    weights = exp( - ( Dist_squared(nonTrivialIndices) / ( sigma0/(2^level) ) ) );
                    W = sparse(rowIndices, colIndices, weights, numSamples, numPatches);
                    one_over_Q = spdiags( ( 1 ./ (sum(W) + eps) )' , 0, size(W, 2), size(W, 2));

                    s_l_ext = (one_over_Q * W') * d_l;
                    f_ext = f_ext + s_l_ext;
                
                end

                break;

            end

            [rowIndices, colIndices] = ind2sub([ numSamples, numSamples ], nonTrivialIndices_subset);  
            weights_subset = exp( - ( Dist_squared_subset(nonTrivialIndices_subset) / ( sigma0/(2^level) ) ) );
            W_subset = sparse(rowIndices, colIndices, weights_subset, numSamples, numSamples);
            % osigurati da matrica W_subset zadovoljava zahtjev simetricnosti
            W_subset = (W_subset +  W_subset') / 2;  

            one_over_Q_subset = spdiags( ( 1 ./ (sum(W_subset, 2) + eps) ) , 0, numSamples, numSamples);


            s_l = (one_over_Q_subset * W_subset) * d_l; 

            f_approx = f_approx + s_l;      


            [rowIndices, colIndices] = ind2sub([ numSamples, numPatches ], nonTrivialIndices);
            weights = exp( - ( Dist_squared(nonTrivialIndices) / ( sigma0/(2^level) ) ) );
            W = sparse(rowIndices, colIndices, weights, numSamples, numPatches);
            one_over_Q = spdiags( ( 1 ./ (sum(W) + eps) )' , 0, size(W, 2), size(W, 2));


            s_l_ext = (one_over_Q * W') * d_l;

            f_ext = f_ext + s_l_ext;
                            
            %sigma = sigma / 2;
        end

        full_diffusion_map(:, dmCoord) = f_ext;
    end

    end






    function sigma = estimateSigma(X)
                
        [~, D] = knnsearch(X, X, 'K', 8);
        sigma = median(D(:, end));

    end



    
    
    