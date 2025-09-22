function [X_subset, dmIndices] = samplePatches(X, patchIndices, dmIndices, subsetRatio)

    % koliko podataka minimalno (ako ih ima toliko) uzeti za podskup ?
    minSubsetSize = 1000;

    numPatches = size(X, 1); % moze i length(patchIndices)

    % zeljeni broj patch-eva za koje cemo racunati egzaktno dif. preslikavanje
    subsetSize = round(subsetRatio * numPatches);

    % detekcija patch-eva u ovom skupu
    % koji odgovaraju patch-evima losije rezolucije
    % na prosloj razini koji su bili sumnjivi
    [intersectt, indicesInPatchIndices, indicesInDmIndices] = intersect(patchIndices, dmIndices);
    dmIndices = indicesInPatchIndices;

    % ako je broj sumnjivih s prosle razine manji od zeljenog broja
    % ostalo nadopuniti nasumicno
    if length(dmIndices) < subsetSize
        allIndices = (1 : numPatches)';
        % skupovna razlika - svi indeksi bez onih koji su vec odabrani
        allIndices = setdiff(allIndices, dmIndices);

        % standardni nacin za uzimanje nasumicnog podskupa odredene velicine
        p = randperm( length(allIndices) );
        allIndices = allIndices( p( 1 : (subsetSize - length(dmIndices)) ) );

        dmIndices  = sort([dmIndices; allIndices]);

    elseif length(dmIndices) > subsetSize  % ako je broj sumnjivih s prosle razine veci od zeljenog broja
        % smanji podskup na zeljenu velicinu
        % npr. uzmi toliki pocetni dio trenutnog podskupa ?
        dmIndices = dmIndices( round( linspace(1, length(dmIndices), subsetSize) ) );
        % PROVJERI JOS OVU LINIJU
    end

    % izdvoji odabrane patch-eve, tj. retke, u podskup
    X_subset = X(dmIndices, :);

end
