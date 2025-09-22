function [diffusion_map, lambda, permutation] = computeDiffusionMap(W, numEigs)
    % COMPUTEDIFFUSIONMAP Performs spectral decomposition of the symmetric
    % diffusion operator P_s = D^{-1/2} * W * D^{-1/2}, and returns the
    % leading eigenvectors and eigenvalues.
    %
    % INPUTS:
    %   W       - Sparse, symmetric affinity matrix (n x n)
    %   numEigs - Number of eigenvectors to compute (default: 10)
    %
    % OUTPUTS:
    %   embedding - Matrix (n x numEigs), each row is Psi(x_i)
    %   lambda    - Vector (numEigs x 1) of eigenvalues

    if nargin < 2
        numEigs = 10;
    end

    numSamples = size(W, 1);

    % racunanje matrice D^{-1/2}
    % ===> D^{-1/2} * D^{-1/2} = D^{-1}
    % ===> dijagonalna matrica s 1 / sqrt(D(i,i)) na dijagonali
    % ===> zelimo rijetku matricu (sparse)
    % ===> MATLAB funkcija spdiags( ... )

    % vektor-stupac suma redaka matrice W
    d = sum(W, 2);  % ne treba + eps zbog sigurne 1 na dijagonali matrice W ?
    D_sqrt_inv = spdiags(1 ./ sqrt(d), 0, numSamples, numSamples);

    D_inv = spdiags(1 ./ d, 0, numSamples, numSamples);
    P = D_inv * W;

    % racunanje simetricne matrice P_s = D^{-1/2} * W * D^{-1/2}
    P_s = D_sqrt_inv * W * D_sqrt_inv;

    % osigurati da je P_s sparse ?
    P_s = sparse(P_s);

    % racunanje spektralne dekompozicije matrice P_s :
    % P_s = U * Lambda * U^T
    % ===> MATLAB solver eigs( ... ) za rijetke matrice
    opts.isreal = true;
    opts.issym = true;
    % zelimo da eigs vrati numEigs vodecih svojstvenih parova
    [U, Lambda] = eigs(P_s, numEigs, 'LM', opts);

    % U je matrica dimenzija numSamples x numEigs
    % stupci su svojstveni vektori
    % Lambda je dijagonalna matrica dimnenzije numEigs x numeigs

    % vektor svojstvenih vrijednosti
    lambda = diag(Lambda);

    % za svaki slucaj sortirati svojstvene vrijednosti silazno
    % a onda i tako poredati pripadne svojstvene vektore
    [lambda, indices] = sort(lambda, 'descend');
    U = U(:, indices);

    Phi = D_sqrt_inv * U;

    % normirati svojstvene vektore ? tj. stupce matrice Phi
    %Optional but good practice �X makes eigenvectors have unit norm (for stability and consistency).
    % sum(Phi.^2) = sum(Phi.^2, 1) --> vektor-redak suma stupaca

    % 1. NACIN
    %Phi = Phi ./ repmat( sqrt(sum(Phi.^2)) , size(Phi, 1), 1);

    % 2. NACIN (automatski broadcast dimenzije)
    Phi = Phi ./ sqrt( sum(Phi.^2) );

    % izostavljamo trivijalni svojstveni par
    inds = 2 : length(lambda);

    % difuzijsko preslikavanje za t = 1
    % ===> mnozenje svakog stupca Psi odgovarajucim lambda
    % tj svojstvenog vektora pripadnom svojstvenom vrijednoscu

    % 1. NACIN
    %diffusion_map = Phi(:, inds) .* repmat( lambda(inds)', size(Phi, 1), 1 );

    % 2. NACIN (automatski broadcast dimenzije)
    diffusion_map = Phi(:, inds) .* lambda(inds)';

    % ===> diffusion_map je dimenzija numSamples x numEigs
    % svaki redak odgovara jednom patch-u, tj vektoru koordinata njegovog dif preslikavanja
    %  = Psi(x_i) = ( Psi_1(x_i). Psi_2(x_i), ..., Psi_numEigs(x_i) )
    % svaki stupac odgovara jednoj difuzijskoj koordinati = Psi_i u svakom patch-u redom
    % = (Psi_i(x_1), Psi_i(x_2), ... , Psi_i(x_numSamples))
    % Psi_i = Lambda_i * Phi_i

    %Now that you have the diffusion coordinates, you can:
    %1. Reorder the data points
    %Sort them by the first diffusion coordinate:
    [~, permutation] = sort(diffusion_map(:, 1));

