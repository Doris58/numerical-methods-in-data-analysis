clear; clc; close all;

% dodati trenutni / radni direktorij u path, da se pronadu sve (pomocne) funkcije
addpath(pwd);

% ovo samo za Octave
%octave = true;
%if octave
if exist('OCTAVE_VERSION', 'builtin')
    pkg load image;       % za imread, imresize, rgb2gray, imfilter
    pkg load statistics;  % za knnsearch, randperm, quantile
end

% input image path - relativni path
imagePath = fullfile(pwd, 'data', 'bird.jpg');

outputDir = fullfile(pwd, 'output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% *** PARAMETRI ********************** %%
% broj razina u Gaussovoj piramidi
maxPyrLevel = 3;
% veli?幂ne uzorka/podskupa dataset-a za ra?憷nanje egzaktnog dif. preslikavanja za svaku razinu
% npr.
% 3. razina: 50% (jer je slika najmanja)
% 2. razina 33,33%
% 1. razina 10% (jer je slika najve?𡩣)
subsetRatio = [0.1, 1/3, 0.5];

% dimenzija patch-a na prvom levelu/originalnoj slici
fullPatchDim = 8;
% broj najblizih susjeda za RIJETKU matricu slicnosti
kNN = 16;
% kojeg najblizeg susjeda uzimamo za lokalni faktor sigma za tezine (self-tuning)
selfTuneNNIndex = 7;
% zeljena niza dimenzija, tj. broj svojstvenih parova
newDim = 7;  % ili npr. 10

% faktor za kontrolu skaliranja udaljenosti za slicnost u anomaly detection
r = 20;
%% ************************************ %%


%% U?䬠TAVANJE DIGITALNE SLIKE (grayscale type) U MATRICU
image = imread(imagePath);
% skaliranje vrijednosti piksela s [0, 255] na decimalne vrijednosti u [0, 1] (double precision)
image = im2double(image);

% mi cemo zasad promatrati samo grayscale digitalne slike
useColor = false;             % Set true to use L*a*b* patches
if size(image, 3) > 1 && ~useColor
    image = rgb2gray(image);
end

% mozda imwrite ... (line 21 u refactor/main)

%% KREIRANJE GAUSSOVE PIRAMIDE
% pyramid - spremnik, npr. MATLAB cell array - sadrzi slike (matrice) za svaku razinu
% alokacija
pyramid = cell(maxPyrLevel, 1);
pyramidDet = cell(maxPyrLevel, 1);
% slika za prvu razinu je bas originalna slika
pyramid{1} = image;
% koristenje ugradene MATLAB funckije impyramid
% mozemo staviti true za MATLAB, ali obavezno false za Octave
blackBoxGaussianPyramid = false;
if blackBoxGaussianPyramid
    for level = 2 : maxPyrLevel
        % Gaussov filter na sliku s preth. razine, pa redukcija dimenzije na ?嶤tvrtinu (downsampling)
        pyramid{level} = impyramid( pyramid{level - 1}, 'reduce' );
    end
else
    for level = 2 : maxPyrLevel
        % "rucno" Gaussov filter na sliku s preth. razine, pa redukcija dimenzije na cetvrtinu
        h = fspecial('gaussian', [5 5], 1);  % 5x5 Gaussian kernel with sigma=1
        blurred = imfilter(pyramid{level - 1}, h, 'replicate');
        reduced = blurred(1:2:end, 1:2:end);  % Downsample by 2

        pyramid{level} = reduced;
    end
end


dmIndices = [];

%pyrScores = cell(1, maxPyrLevel);  % anomaly scores at each level

%% ITERACIJE PO GAUSSOVOJ PIRAMIDI
% od najlosije rezolucije slike (najvise razine) do originalne (prve razine)
for level = maxPyrLevel : -1 : 1
level = maxPyrLevel;
    fprintf('\n--- Processing Pyramid Level %d ---\n', level);

    % dimenzija kvadratnih patch-eva za svaku razinu upola manja od dimenzije za prethodnu razinu
    patchDim = floor( fullPatchDim / level );
    % ===> dobiva li se to onda ovim izrazom ? ; ovako izracunati automatski za bilo koji level
    % to je dobro, jer ne znamo mozda unaprijed koliki je maxPyrLevel
    % ili na pocetku izracunati vektor dimenzija patcheva za svaku razinu !!

    % zeljena niza dimenzija, tj. broj svojstvenih parova
    % ===> svakako zelimo da je manja od trenutne = patchDim^2 - OBAVEZNO OVO
    newDim = min( patchDim^2, newDim );

    % slika (matrica) na ovoj razini
    image = pyramid{level};
    imshow(image);

    %% 1. IZDVAJANJE SVIH (!) PREKLAPAJUCIH (do na 1 piksel) PATCH-EVA ZELJENE VELICINE SA SLIKE
    % pamtimo podatke kao vektore-retke i odgovaraju?囻 piksele slike koji su centri
    [X, patchCenters] = extractPatches(image, patchDim);

    % zapamcene kooridnate retka i stupca centara nam sluze za ===>

    % redom kako su patch-evi spremljeni u retke matrice X (tj patchCenters)
    % redni brojevi tih patch-eva (tj. njihovih centara)
    % na slici krecuci se po stupcima pa po retcima matrice-slike
    patchIndices = sub2ind([ size(image, 1), size(image, 2) ], patchCenters(:, 1), patchCenters(:, 2));

    subsetRatio = subsetRatio(level);

    %% 2. IZDVAJANJE PODSKUPA PATCH-EVA ZELJENE VELICINE ZA RACUNANJE EGZ. DIF. PRESLIKAVANJA
    % uz uzimanje sumnjivih patch-eva iz prethodne razine
    [X_subset, dmIndices] = samplePatches(X, patchIndices, dmIndices, subsetRatio);

    %% 3. RACUNANJE MATRICE TEZINA ZA PODSKUP
    % sa skaliranjem udaljenosti: sigma_i = dist(x_i, x_k)
    % ali rijetka - netrivijalne tezine samo za kNN najblizih susjeda
    % vracamo i rijetku matricu (kvadrata) medusobnih udaljenosti za podskup
    [W, Dist_squared_subset, nonTrivialIndices_subset] = computeAffinityMatrix(X_subset, kNN, selfTuneNNIndex);

    % vizualizacija
    figure;
    imagesc(W);
    colorbar;
    title('Affinity Matrix W');
    xlabel('Column Index');
    ylabel('Row Index');
    saveas(gcf, fullfile('output', 'affinity_matrix_W.png'));

    %% 4. RACUNANJE DIFUZIJSKOG PRESLIKAVANJA ZA PODSKUP
    % postavili smo broj zeljenih koordinata kao parametar newDim
    [diffusion_map, lambda, permutation] = computeDiffusionMap(W, newDim);

    % vizualizacija
    d = sum(W, 2);
    D_inv = spdiags(1 ./ d, 0, size(W,1), size(W,1));
    P = D_inv * W;
    P_sorted = P(permutation, permutation);

    figure;
    imagesc(P_sorted);
    colorbar;
    title('Permuted Markov Matrix P (by first diffusion coordinate)');
    saveas(gcf, fullfile('output', 'permuted_P_matrix.png'));

    plotDiffusionEmbedding(diffusion_map, lambda);

    %% 5. PROSIRENJE DIF. PRESLIKAVANJA NA CIJELI SKUP - SVE PATCH-EVE
    % Laplaceova piramida
    % ako je stvarno uzet podskup, a ne cijeli skup
    if size(X_subset, 1) ~= size(X, 1)
        full_diffusion_map = extendDiffusionMap(X_subset, X, diffusion_map, Dist_squared_subset, nonTrivialIndices_subset, kNN);
    else % ako je vec promatran cijeli dataset
        full_diffusion_map = diffusion_map;
    end

    % parametar - velicina susjedstva za anomaly score

    % losija rezulocija, tj visi level ===> manji prozor za anomaly score
    numPatchesinDim = round(20/level);
    [topLeft, bottomRight] = extractNeighborhood(patchCenters, numPatchesinDim);

    [H, W] = size(image);
    imageLinIndices = reshape(1 : H * W, H, W);

    % parametar n_pairs cemo definirati unutar ove metode
	sigma_bar = estimateGlobalSigma(full_diffusion_map, r);

    % 6. ANOMALY DETECTION
    detection = anomalyDetection(full_diffusion_map, ...
    patchIndices, patchCenters, topLeft, bottomRight, imageLinIndices, sigma_bar);

     %% Save and show detection map
    figure;
    imshow(detection, []);
    title(sprintf('Anomaly Map - Level %d', level));
    colormap('hot'); colorbar;

    imwrite(im2uint8(mat2gray(detection)), colormap('hot'), ...
        fullfile(outputDir, sprintf('detect_level%d.png', level)));

    %% Find suspicious pixels using 95% quantile
    thresh = quantile(detection(:), 0.95);
    suspiciousMask = zeros(size(image));
    suspiciousMask(patchIndices) = detection(:) > thresh;

    imwrite(suspiciousMask, fullfile(outputDir, sprintf('suspicious_level%d.png', level)));

    %% Upsample suspicious locations to next (finer) level
    if level > 1
        finerSize = size(pyramid{level - 1});
        dmIndices = upsampleIndicesToFinerLevel(suspiciousMask, finerSize);
    end


end

%% Smooth finest level detection map
confidence = detection;  % from level = 1
smoothedConfidence = confidence;
smoothedConfidence(confidence < 0.3) = 0;
smoothedConfidence = imfilter(smoothedConfidence, fspecial('average', 3));

imwrite(im2uint8(smoothedConfidence), colormap('hot'), ...
    fullfile(outputDir, 'smooth_confidence.png'));

figure;
imshow(smoothedConfidence, []);
title('Smoothed Confidence Map');
colormap('hot'); colorbar;

