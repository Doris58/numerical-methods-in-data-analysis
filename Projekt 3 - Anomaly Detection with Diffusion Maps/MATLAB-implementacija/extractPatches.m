function [patches, patchCenters] = extractPatches(image, patchDim)
    % EXTRACTPATCHES Extract overlapping square patches from a grayscale image.
    %
    % INPUTS:
    %   image       - Grayscale image (double precision)
    %   patchDim  - Patch dimension (e.g., 8 for 8x8 patches)
    %
    % OUTPUTS:
    %   patches       - Matrix of shape (numPatches x patchDim^2)
    %                   Each row is a flattened patch.
    %   patchCenters  - Matrix of shape (numPatches x 2)
    %                   Each row is [row, col] location of the patch center.

    fprintf('\n--- Extracting patches of dimension %d ---\n', patchDim);

    % svakom pikselu slike - elementu matrice
    % zelimo pridruziti kvadratni patch dimenzije patchDim ciji je taj piksel centar

    % zbog svih piksela udaljenih od ruba slike za manje od patchDim/2
    % prosirit cemo sliku za polovinu patchDim
    halfPatchDim = floor(patchDim / 2);

    % MATLAB funkcija za prosirenje matrice sa svih strana
    paddedImage = padarray(image, [halfPatchDim, halfPatchDim], 'symmetric');

    % dimenzije neprosirene slike
    [H, W] = size(image);
    % ===> ukupan broj patch-eva (tj. podataka velike dimenzije) - jedan za svaki piksel slike
    numPatches = H * W;

    % alokacija spremnika za patch-eve
    % spremamo ih u retke matrice (kao retke zbog kasnije upotrebe MATLAB funkcija poput pdist2)
    patches = zeros(numPatches, patchDim^2);
    % alokacija spremnika - matrice za koordinate retka i stupca pripadnih centara na slici
    patchCenters = zeros(numPatches, 2);

    % indeks retka matrice u koju spremamo, tj. indeks patch-a
    k = 1;
    % krecemo se prvo po stupcima pa po retcima matrica image i paddedImage
    % promatramo centre patch-eva - elemente matrice Image
    for c_center = 1 : W
        for r_center = 1 : H
            % istovremeno pridruzujemo pripadni gornji lijevi vrh - on je iz paddedImage
            top = r_center;
            left = c_center;

            % izdvajanje pripadnog patch-a iz paddedImage
            patch = paddedImage(top : top + patchDim - 1, ...
                                left : left + patchDim - 1);

            % vektorizacija (flattening) patch-a (po njegovim stupcima!) u vektor duljine patchDim^2
            patch = patch(:);
            % spremanje patch-a kao retka matrice svih patch-eva
            patches(k, :) = patch';
            % spremanje koordinata retka i stupca pripadnog centra u originalnoj slici
            patchCenters(k, :) = [r_center, c_center];
            k = k + 1;
        end
    end

end
