function [W, Dist_squared_subset, nonTrivialIndices] = computeAffinityMatrix(X, kNN, selfTuneNNIndex)
    % COMPUTEAFFINITYMATRIX Constructs a sparse affinity matrix using
    % self-tuned Gaussian kernel (Zelnik-Manor & Perona).
    %
    % INPUTS:
    %   X              - Data matrix (numSamples ¡Ñ patchDim^2)
    %   kNN            - Number of nearest neighbors to use
    %   selfTuneNNIndex  - Index of neighbor used to define £m_i (e.g., 7)
    %
    % OUTPUT:
    %   W                - Sparse affinity matrix (numSamples ¡Ñ numSamples)
    %   Dist_squared_subset - Sparse matrix of squared distances
    %   nonTrivialIndices   - Linear indices of non-zero elements in W

    fprintf('\n--- Computing affinity matrix ---\n');

    % da bi matrica W bila rijetka
    % za svaki patch/podatak racunamo tezine samo za kNN najblizih susjeda
    % ostale tezine smatrati da su = 0
    % ( pa ih tako mozemo i postaviti u matrici ==> sparse matrica )

    % ===> racunati i samog sebe uz kNN najblizih susjeda,
    % jer ne zelimo da je slicnost sa samim sobom 0, nego 1
    % ===> gledati kNN + 1 najblizih susjeda za svaki patch

    %fprintf('\nCalling knnsearch with %d samples and %d neighbors...\n', size(X,1), kNN);
    %fflush(stdout);
    %[Idx, Dist] = knnsearch(X, X, 'K', kNN + 1);
    % ===> ovo ne radi u Octave-u !

    [Dist, Idx] = pdist2(X, X, 'euclidean', 'Smallest', kNN + 1);
    % pdist2 daje matrice dimenzija (kNN + 1) x brojPodataka ===>
    Dist = Dist';
    Idx = Idx';

    fprintf('\n Distances to kNN computed. \n');

    disp('First few indices:');
    disp(Idx(1:5, :));
    disp('First few distances:');
    disp(Dist(1:5, :));

    % osigurati da je K-ti najblizi susjed x_K za sigma_i = dist(x_i, x_K)
    % medu najbl. susjedima za koje smo racunali udaljenosti (tj. ciju slicnost necemo postaviti na 0)
    % da iskoristimo vec izracunate vrijednosti
    selfTuneNNIndex = min( selfTuneNNIndex + 1, size(Dist, 2) );  % ili (..., kNN + 1), ali ovo je sigurnije
    % preslikavanje sigma_i = dist(x_i, x_K)
    sigma = Dist(:, selfTuneNNIndex);

    fprintf('\n Scaling factors extracted. \n');

    % MATLAB funkcija sparse( ... )
    % ===> za konstrukciju rijetke matrice dovoljno je konstruirati:
    % - vektor indeksa redaka
    % - vektor indeksa stupaca
    % - vektor pripadnih vrijednosti
    % za sve elemente koji nisu = 0

    numSamples = size(X, 1);
    % broj nenul elemenata
    numNonTrivial = numSamples * (kNN + 1);

    % alokacija memorije za tri spomenuta vektora
    rowIndices = zeros(1, numNonTrivial);
    colIndices = zeros(1, numNonTrivial);
    weights = zeros(1, numNonTrivial);

    % spremiti rijetku matricu medusobnih udaljenosti podataka u podskupu
    % da bismo je mogli samo iskor. kasnije u Laplaceovoj piramidi
    % ===> sparse matrica Dist_subset (moze i kvadrate udaljenosti)

    % ==> spremiti odvojeno vektor netrivijalnih udaljenosti
    distances = zeros(1, numNonTrivial);
    %scales = zeros(1, numNonTrivial);

    fprintf('\n Vectors for sparse matrix allocated. \n');

    % popunjavanje tih vektora - bilo kojim redoslijedom, samo da si medusobno odgovaraju,
    % tj da je redoslijed isti za sva tri vektora
    i = 1;
    % na ovaj nacin - da iskoristimo strukturu matrica Idx i Dist
    for patchIndex = 1 : numSamples

        rowRange = i : (i + kNN);

        rowIndices(rowRange) = patchIndex;
        colIndices(rowRange) = Idx(patchIndex, :);

        distances(rowRange) = Dist(patchIndex, :);

        %s_i = sigma(patchIndex);
        %s_j = sigma(Idx(patchIndex, :));
        %d2  = Dist(patchIndex, :).^2;

        %weights(range) = exp( - d2 ./ (s_i * s_j + eps) );

        % vektor neke duljine = vektor te duljine ./ vektor te duljine
        % dodati + eps jer se moze dogoditi da je neki sigma_i = dist(x_i, x_K) = 0, pa bi nazivnik bio = 0 !!
        weights(rowRange) = exp( - ( Dist(patchIndex, :).^2 ) ./ ( sigma(patchIndex) .* ( sigma( Idx(patchIndex, :) ) )' + eps  ) );
        %denom = sigma(patchIndex) * sigma(Idx(patchIndex, :)) + eps;
        %weights(rowRange) = exp(-(Dist(patchIndex, :).^2) ./ denom);

        %scales(i : i + kNN) = sigma(patchIndex) * sigma( Idx(patchIndex, :) ); % vektor = skalar * vektor

        i = i + (kNN + 1);

        if mod(patchIndex, 1000) == 0
            fprintf('Processed %d / %d samples\n', patchIndex, numSamples);
            %fflush(stdout);
        end

    end

    fprintf('\n Building sparse matrices.\n');

    distances = distances .^2;

    Dist_squared_subset = sparse(rowIndices, colIndices, distances, numSamples, numSamples);

    W = sparse(rowIndices, colIndices, weights, numSamples, numSamples);

    % osigurati da matrica W zadovoljava zahtjev simetricnosti
    % NAPOMENA uz ovo
    W = (W + W') / 2;  % je li ovo efikasno na sparse matrici?

    nonTrivialIndices = sub2ind( [numSamples, numSamples] , rowIndices, colIndices);

    fprintf('Affinity matrix computation complete.\n');

    disp('First 10x10 block of W');
    disp(W(1:10, 1:10));

end

