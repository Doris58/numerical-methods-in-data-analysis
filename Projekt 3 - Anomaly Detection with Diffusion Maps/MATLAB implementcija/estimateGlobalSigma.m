function sigma_bar = estimateGlobalSigma(full_diffusion_map, r)

    if nargin < 2
        r = 1.0;
    end

    numPatches = size(full_diffusion_map, 1);

    n_pairs = min( 1000, round(numPatches/4) );

    n_elements = 2 * n_pairs;

    % standardni nasumican izbor
    p = randperm(numPatches);
    p = p(1 : n_elements);

    distances = zeros(1, n_pairs);

    for i = 1 : n_pairs
        distances(i) = sqrt( sum( full_diffusion_map(p(i), :) - full_diffusion_map( p(n_pairs + i) , :) ) .^2 );
    end

    % varijanca = std^2
    sigma_bar = r * ( std(distances)^2 );

end