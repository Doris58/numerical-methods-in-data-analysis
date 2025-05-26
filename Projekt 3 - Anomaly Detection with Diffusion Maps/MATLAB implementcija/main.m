clear; clc; close all;


% input image path
imagePath = 'C:\\Users\\grgos\\Downloads\\napredne linearne\\capsule-8971865\\data\\bird.jpg';     

%% *** PARAMETRI ********************** %%
% broj razina u Gaussovoj piramidi
maxPyrLevel = 3;    
% veličine uzorka/podskupa datset-a za računanje egzaktnog dif. preslikavanja za svaku razinu
% npr.
% 3. razina: 50% (jer je slika najmanja)
% 2. razina 33,33%
% 1. razina 10% (jer je slika najveća)
subsetRatio = [0.1, 1/3, 0.5];

% dimenzija patch-a na prvom levelu/originalnoj slici
fullPatchDim = 8;                 
% broj najblizih susjeda za RIJETKU matricu slicnosti
kNN = 16;                      
% kojeg njablizeg susjeda uzimamo za lokalni faktor sigma za tezine (self-tuning)
selfTuneNNIndex = 7;    
% zeljena niza dimenzija, tj. broj svojstvenih parova
newDim = 7;  % ili npr. 10

% faktor za kontrolu skaliranja udaljenosti za slicnost u anomaly detection
r = 20;
%% ************************************ %%      


%% UČITAVANJE DIGITALNE SLIKE (grayscale type) U MATRICU 
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
blackBoxGaussianPyramid = true;
if blackBoxGaussianPyramid
    for level = 2 : maxPyrLevel
        % Gaussov filter na sliku s preth. razine, pa redukcija dimenzije na četvrtinu (downsampling)
        pyramid{level} = impyramid( pyramid{level - 1}, 'reduce' );
    end
%else
    % "rucno" Gaussov filter na sliku s preth. razine, pa redukcija dimenzije na cetvrtinu 
end


dmIndices = [];  

%pyrScores = cell(1, maxPyrLevel);  % anomaly scores at each level

%% ITERACIJE PO GAUSSOVOJ PIRAMIDI 
% od najlosije rezolucije slike (najvise razine) do originalne (prve razine)
for level = maxPyrLevel : -1 : 1
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

    %% 1. IZDVAJANJE SVIH (!) PREKLAPAJUCIH (do na 1 piksel) PATCH-EVA ZELJENE VELICINE SA SLIKE
    % pamtimo podatke kao vektore-retke i odgovarajuće piksele slike koji su centri
    [X, patchCenters] = extractPatches(image, patchDim);

    % zapamcene kooridnate retka i stupca centara nam sluze za ===>

    % redom kako su patch-evi spremljeni u retke matrice X (tj patchCenters)
    % redni brojevi tih patch-eva (tj. njihovih centara)
    % na slici krecuci se po stupcima pa po retcima matrice-slike
    patchIndices = sub2ind([ size(image, 1), size(image, 2) ], patchCenters(:, 1), patchCenters(:, 2));

    subsetRatio = subsetRatio(level);

    % 2. IZDVAJANJE PODSKUPA PATCH-EVA ZELJENE VELICINE ZA RACUNANJE EGZ. DIF. PRESLIKVANJA
    % uz uzimanje sumnjivih patch-eva iz prethodne razine
    [X_subset, dmIndices] = samplePatches(X, patchIndices, dmIndices, subsetRatio);

    % 3. RACUNANJE MATRICE TEZINA ZA PODSKUP
    % sa skaliranjem udaljenosti: sigma_i = dist(x_i, x_k)
    % ali rijetka - netrivijalne tezine samo za kNN najblizih susjeda
    % vracamo i rijetku matricu (kvadrata) medusobnih udaljenosti za podskup
    [W, Dist_squared_subset, nonTrivialIndices_subset] = computeAffinityMatrix(X_subset, kNN, selfTuneNNIndex);

    % 4. RACUNANJE DIFUZIJSKOG PRESLIKAVANJA ZA PODSKUP
    % postavili smo broj zeljenih koordinata kao parametar newDim
    [diffusion_map, lambda] = computeDiffusionMap(W, newDim);


    % 5. PROSIRENJE DIF. PRESLIKAVANJA NA CIJELI SKUP - SVE PATCH-EVE
    % Laplaceova piramida
    % ako je stvarno uzet poskup, a ne cijeli skup
    if size(X_subset, 1) ~= size(X, 1)
        full_diffusion_map = extendDiffusionMap(X_subset, X, diffusion_map, Dist_squared_subset, nonTrivialIndices_subset);
    else % ako je vec promatran cijeli dataset
        full_diffusion_map = diffusion_map;
    end

    % parametar - velicina susjedstva za anomaly score 
    numPatchesinDim = round(20/level);
    [upLeft, bottomRight] = extractNeighborhood(patchCenters, numPatchesinDim);

    [H, W] = size(image);
    imageLinIndices = reshape(1 : H * W, H, W);

    % parametar n_pairs cemo definirati unutar ove metode
	sigma_bar = estimateGlobalSigma(full_diffusion_map, r);

    % 6. ANOMALY DETECTION
    detection = anomalyDetection(full_diffusion_map, ...
    patchIndices, patchCenters, upLeft, bottomRight, imageLinIndices, sigma_bar);

    % 7. SUMNJVI PATCH-EVI - 0.05 quantile
    temp = zeros(size(image, 1), size(image, 2)); 
    thresh = quantile(detection(:), 0.95);
    temp(patchIndices) = detection(:) > thresh;
    dmIndices = find(temp);

    filename = [path_name image_name 'suspicious.png'];
    imwrite(temp, filename);

    detIm = nan(size(image, 1), size(image, 2));
    detIm(patchIndices) = detection(:);

    filename = [path_name image_name_pyr 'detect.png'];
    temp     = detIm;
    temp(isnan(temp)) = 0;
    temp     = im2uint8(temp);
    imwrite(temp, cmap, filename);

    pyramidDet{level} = detIm;

    % 8. UPSAMPLING SUMNJIVIH NA SLJEDECU RAZINU
    if level > 1
        dmIndices = upsampleIndicesToFinerLevel(dmIndices, size(pyramid{level}), size(pyramid{level - 1}));
    end


end



% smoothed anomaly score image of final level
confidence = pyramidDet{1};
smoothedConfidence = confidence;
smoothedConfidence(confidence < 0.3) = 0; 
smoothedConfidence = imfilter(smoothedConfidence, fspecial('average',3));

filename = [path_name image_name '_smoothConf.png'];
smoothedConfidence(isnan(smoothedConfidence)) = 0;
mwrite(im2uint8(smoothedConfidence), cmap, filename);



%visualizeResults(pyrScores, pyramid);