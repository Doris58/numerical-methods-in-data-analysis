function sigma_bar = estimateGlobalSigma(full_diffusion_map, r)
    % procedura za racunanje sigma_bar tocno kakva je opisana u glavnoj literaturi seminara

    % nasumican odabir n_pair piksela originalne slike ciju cemo dif. udaljenost izracunati
    % <=> nasumican odabir n_pair patch-eva,
    % Aa kako je dif. udaljenost = eukl- udaljenost vrijednosti dif. prelikavanja
    % ==> odabrati omdah n_pair vrijednsoti dif. preslikavanja

    if nargin < 2
        r = 1.0;
    end

    numPatches = size(full_diffusion_map, 1);

    % recimo da je n_par = 1/4 broja patch-eva, ali maksimalno 1000
    n_pairs = min( 1000, round(numPatches/4) );

    n_elements = 2 * n_pairs;

    % standardni nasumican izbor - n_elements elemenata od numPatches vrijednosti dif. preslikavanja
    p = randperm(numPatches);
    p = p(1 : n_elements);

    distances = zeros(1, n_pairs);
    % racunanje euklidskih udaljenosti izmedu i-te i (n_pair + i)-te nasumicnoo
    % odabrane vrijednosti edifuzijskog preslikavanja
    % (i dalje su to "random" parovi)
    for i = 1 : n_pairs
        difference = full_diffusion_map(  p(i) , :) - full_diffusion_map( p(n_pairs + i) , :);
        distances(i) = sqrt( sum( difference.^2 ) );
    end

    % varijanca = std^2
    sigma_bar = r * ( std(distances)^2 );

end
