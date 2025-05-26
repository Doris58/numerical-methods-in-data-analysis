function [W, Dist_squared_subset, nonTrivialIndices] = computeAffinityMatrix(X, kNN, selfTuneNNIndex)
    % COMPUTEAFFINITYMATRIX Constructs a sparse affinity matrix using
    % self-tuned Gaussian kernel (Zelnik-Manor & Perona).
    %
    % INPUTS:
    %   X              - Data matrix (numSamples × patchDim^2)
    %   kNN            - Number of nearest neighbors to use
    %   selfTuneNNIndex  - Index of neighbor used to define σ_i (e.g., 7)
    %
    % OUTPUT:
    %   W              - Sparse affinity matrix (numSamples × numSamples)
    
    
    % da bi matrica W bila rijetka
    % za svaki patch/podatak racunamo udaljenosti samo do kNN najblizih susjeda
    % ostale tezine smatrati da su = 0 ( pa ih tako mozemo i postaviti u matrici ==> sparse matrica )

    % ===> racunati i samog sebe uz kNN najblizih susjeda, 
    % jer ne zelimo da je slicnost sa samim sobom 0, nego 1
    % ===> gledati kNN + 1 najblzih susjeda za svaki patch

    [Idx, Dist] = knnsearch(X, X, 'K', kNN + 1); 

    % osigurati da je K-ti najblizi susjed x_K za sigma_i = dist(x_i, x_K)
    % medu najbl. susjedima za koje smo stvarno racunali udaljenosti
    % tj. ciju slicnost nismo postavili na 0 !!
    selfTuneNNIndex = min( selfTuneNNIndex, size(Dist, 2) );  % ili (..., kNN )
    % preslikavanje sigma_i = dist(x_i, x_K)
    sigma = Dist(:, selfTuneNNIndex); 

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
    % da bismo je mogli samo iskor. kasnije u Laplacevoj piramidi
    % ===> sparse matrica Dist_subset (moze i kvadrate udaljenosti)

    % ==> spremiti odvojeno vektor netrivijalnih udaljenosti
    distances = zeros(1, numNonTrivial);
    %scales = zeros(1, numNonTrivial);

    % popunjavanje tih vektora - bilo kojim redoslijedom, samo da si medusobno odgovaraju,
    % tj da je redoslijed isti za sva tri vektora
    i = 1;
    % na ovaj nacin - da iskoristimo strukturu matrica Idx i Dist
    for patchIndex = 1 : numSamples

        rowIndices(i : i + kNN) = patchIndex;
        colIndices(i : i + kNN) = Idx(patchIndex, :);

        distances(i : i + kNN) = Dist(patchIndex, :);

        % vektor neke duljine = vektor te duljine ./ vektor te duljine
        % dodati + eps jer se moze dogodiiti da je neki sigma_i = dist(x_i, x_K) = 0, pa bi nazivnik bio = 0 !!
        weights(i : i + kNN) = exp( - ( Dist(patchIndex, :).^2 ) ./ ( sigma(patchIndex) * sigma( Idx(patchIndex, :) ) + eps  ) );

        %scales(i : i + kNN) = sigma(patchIndex) * sigma( Idx(patchIndex, :) ); % vektor = skalar * vektor 

        i = i + (kNN + 1);

    end

    distances = distances .^2;

    Dist_squared_subset = sparse(rowIndices, colIndices, distances, numSamples, numSamples);

    W = sparse(rowIndices, colIndices, weights, numSamples, numSamples);

    % osigurati da matrica W zadovoljava zahtjev simetricnosti
    % NAPOMENA uz ovo
    W = (W + W') / 2;  % je li ovo efikasno na sparse matrici?

    nonTrivialIndices = sub2ind( [numSamples, numSamples] , rowIndices, colIndices);


    end
    