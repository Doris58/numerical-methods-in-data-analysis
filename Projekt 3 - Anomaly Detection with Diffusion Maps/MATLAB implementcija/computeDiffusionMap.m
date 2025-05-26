function [diffusion_map, lambda] = computeDiffusionMap(W, numEigs)
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
    % ===> dijagonalna matrica s 1/ sqrt(D(i,i)) na dijagonali
    % ===> zelimo rijetku matricu (sparse)
    % ===> MATLAB funkcija spdiags( ... )

    % vektor-stupac suma redaka matrice W
    d = sum(W, 2);  % ne treba + eps zbog sigurne 1 na dijagonali matrice W ? 
    D_sqrt_inv = spdiags(1 ./ sqrt(d), 0, numSamples, numSamples);  
    
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

    % za svaki slucaj sortirati svojstvee vrijednosti silazno
    % a onda i tako poredati pripadne svojstvene vektore
    [lambda, indices] = sort(lambda, 'descend'); 
    U = U(:, indices);


    % PROVJERITI OVO JOS !!!! (refactor/main : line 436 )
    Psi = D_sqrt_inv * U;  

    % normirati svojstvene vektore ? tj. stupce matrice Psi 
    % sum(Psi.^2) = sum(Psi.^2, 1) --> vektor-redak suma stupaca
    Psi = Psi ./ repmat( sqrt(sum(Psi.^2)) , size(Psi, 1), 1); 

    % izostavljamo trivijalni svojstveni par
    inds = 2 : length(lambda); 

    % difuzijsko preslikavanje za t = 1
    % ===> mnozenje svakog stupca Psi odgovarajucim lambda
    % tj svojstvenog vektora pripadnom sv0jstvenom vrijednoscu 
    diffusion_map = Psi(:, inds) .* repmat( lambda(inds)', size(Psi, 1), 1 );
    
    %V MOZDA JE DOVOLJNO SAMO OVO - Psi tj DM direktno iz U i lambda
    %diffusion_map = U .* lambda';  

    % ===> diffusion_map je dimenzija numSamples x numEigs
    % svaki redak odgovara jednom patch-u, tj vektoru kooridnata njegovog dif preslikavanja
    %  = Psi(x_i) = ( Psi_1(x_i). Psi_2(x_i), ..., Psi_numEigs(x_i) )
    % svaki stupac odgovara jeednoj difuzijskoj koordinati = Psi_i u svakom patch-u redom
    % = (Psi_i(x_1), Psi_i(x_2), ... , Psi_i(x_numSamples))
    % Psi_i = Lambda_i * Phi_i

    % Graf - svojstvene vrijednosti
    figure;
    % lambda sortiran silazno
    plot(lambda);
    title('\lambda');
    ylim([0 1]);
    saveas(gcf,'C:\Users\grgos\Downloads\napredne linearne\capsule-8971865\code\output\lambda','png');
    cd('C:\Users\grgos\Downloads\napredne linearne\capsule-8971865\code\output\');

    % Graf - prve tri koordinate/komponente dofuzijskog preslikavanja
    % u svim patch-evima/podacima
    figure;
    scatter(1 : size(diffusion_map, 1), diffusion_map(:, 1))
    hold on %hold all
    scatter(1 : size(diffusion_map, 1), diffusion_map(:, 2))
    scatter(1 : size(diffusion_map, 1), diffusion_map(:, 3))
    title('Prve tri koordinate dif. preslikavanja');
    saveas(gcf,'C:\Users\grgos\Downloads\napredne linearne\capsule-8971865\code\output\Diffusion Map Coordinates','png');
    cd('C:\Users\grgos\Downloads\napredne linearne\capsule-8971865\code\output\');

    % graf difuzijskog preslikavanja u R^3
    %figure(101);
    figure;
    scatter3(diffusion_map(:, 1), diffusion_map(:, 2), diffusion_map(:, 3))
    title('Difuzijsko preslikavanje');
    saveas(gcf,'C:\Users\grgos\Downloads\napredne linearne\capsule-8971865\code\output\Diffusion Map','png');
    cd('C:\Users\grgos\Downloads\napredne linearne\capsule-8971865\code\output\');
    
    end
    