
% ***************** SVICARSKA ROLADA / PODACI ******************************** %
% SWISS ROLL DATA =
% { (x, y, z) = (t*sin(t), h, t*cos(t)) : t in [3pi/2, 9pi/2], h in [0, 20] }

% PARAMETRIZACIJA KRUZNICE RADIJUSA r -- x^2 + y^2 = r^2
% polarne koordinate: 
% r - duljina radijvektora ( za tocku (x, y) )
% phi - kut radijvektora s pozitivnim dijelom x-osi
% x = r*sin(phi), y = r*cos(phi)

% r = phi ==> 
% x = phi * sin(phi), y = phi * cos(phi)
% ==> SPIRALA u ravnini

% uvodenje trece koordinate ==> Svicarska rolada !!
% ovisno koju koordinatu odaberemo za tu, duz te osi se siri rolada

a_t = 3 * pi / 2;
b_t = 9 * pi / 2;
n_t = 1000;

% ovako generiram n UNIFORMNO DISTRIBUIRANIH pseudoslucajnih brojeva 
% u segmentu [a_t, b_t]
t = a_t + (b_t - a_t) .* rand(n_t, 1);  % rand(n_t) bi bila kvadratna matrica
% => da, ocito su svi gen. brojevi u tom segmentu !

a_h = 0;
b_h = 20;
n_h = 1000;

h = a_h + (b_h - a_h) .* rand(n_h, 1);

x = t .* sin(t);
y = h;
z = t .* cos(t);

c = linspace(1, 10, length(x));
% nakon x, y, z, prvo ide Marker Size pa Marker Color
% mogu i preko name, value?
% scatter3(x, y, z, [], c, 'filled')

% ovako bojam podatke s obzirom na velicinu varijable t, 
% od hladnije prema toplijoj boji
scatter3(x, y, z, [], t, 'filled')

% JOŠ: nacrtati cijelu mnogostrukost
% **************************************************************************** %

% **** 1. KORAK ***** k-NN *************************************************** %

% svi vektori gore su vektori-stupci!
X = [x, y, z];

% za medusobne udaljenosti 1000 tocaka/vektora/podataka imam ukupno oko 500 
% relevantnih brojeva/velicina

% tj, za n vektora ukupno n * (n - 1) / 2 velicina
% jer je svaka metrika simetricna i refleksivna

% ali, vazno mi je znati za svaku udaljenost za koja dva vektora je ona
% ==> moram spremiti udaljenosti u matricu dimenzije n x n ?
% ==> matrica ce sigurno biti simetricna, s nulama na dijagonali

D = zeros(n_t, n_t);

for i = 1 : n_t
    for j = 1: n_t
        D(i, j) = norm( X(i, :) - X(j, :) ); % norm je bas norma-2, tj. euklidska
    end
end

% provjera je li D simetricna
% disp(D)

% ILI OVAKO -- matlab sam radi ono sto sam ja:
% cak i ono racunanje samo n(n-1)/2 puta i spremanje u jedan vektor
D2 = squareform( pdist(X) );  % ==> ovakav postupak je efikasniji !
%disp(D2)

% sortiram retke matrice D uzlazno
[sortedD, Index] = sort(D, 2, 'ascend');

% provjera da se D ne mijenja sort-om
%disp(D)

k = 10; % hiperparametar -- mijenjati !

%D = D(:, 1 : k)  % NE ==> ne racunati da je netko susjed sam sebi, a taj bi svakako bio najblizi

% D = D(:, 2 : k + 1)

% k najblizih susjeda za svaki podatak
Index = Index(:, 2 : k + 1);

% ****************************************************************************** %

% **** 2. KORAK ***** racunanje W ********************************************** %

% pratiti izvod s predavanja do konacne formule
% x vektor ==> norma2(x)^2 = x' * x

% obratiti paznju na mnozenja:
% - vektora retka i vektora stupca (inner (skalarni) product ?)
% - vektora retka i matrice
% - vektora stupca i vektora retka (outer product ?)
% - matrice i vektora stupca

% ovo gore posebno ako su vektori jedinicni
% ==> ocekivano je da postoje odgovarajuce fje u MATLAB-u
% npr. repmat

% vektor-stupac * jedinicni vektor-redak =
% = matrica ciji su svi stupci pocetni vektor-stupac
% tj, u jednom retku su vi elementi jednaki
% odgovarajucem elementu pocetnog vektora-stupca
% ==> rezultat je ocito matrica ranga 1

% jedinicni vektor-redak * vektor-stupac
% = suma svih elemenata vektora-stupca (svakako skalar)

% matrica * jedinicni vektor-stupac
% = vektor-stupac ciji su elementi 
% sume svih elemenata odgovarajuceg retka matrice

% jedinicni vektor-redak * matrica =
% = vektor-redak ciji su elementi
% sume svih elemenata odgovarajuceg stupca matrice

% ==> jedinicni vektor-redak * matrica * jedinicni vektor-stupac
% = suma svih elemenata matrice ?

epsilon = 1e-3; % za regularizaciju matrica G_i (npr. d = 3 < 10 = k)
e = ones(k, 1); % ones(k) je matrica dimenzije k x k

% treba nam na kraju ukupno n x k vrijednosti w, zapravo n x n
% i trebamo znati koja vrijednost pripada kojem i i j
% ==> napraviti/alocirati memoriju za odmah cijelu matricu

W = zeros(n_t); % bas ovako je dobro i jer ce odmah preostale vrijednosti postaviti na 0

for i = 1 : n_t
    % sve ove ostale meduvrijednosti (matrice i vektore),
    % za pojedini i, ne moramo pamtiti,
    % samo konacne koeficijente w
    X_i = X(Index(i, :), :)';  % DA, dobro sam!
    x_i = X(i, :)'; % PAZI, da bude vektor-redak !
    C_i = repmat( x_i, 1, k );   % repmat(x_i, k) daje matricu
    % provjera jednakosti; sto je slozenije od ovo dvoje?
    B_i = x_i * e'; % * je standardno matricno mnozenje
    Z_i = C_i - X_i;
    G_i = Z_i' * Z_i;
    % regularizacija matrice ??
    G_i = G_i + epsilon * eye(k);

    G_i_inv = inv(G_i);

    vector_w_i = sum(G_i_inv, 2);
    scalar_w_i = sum(G_i_inv, "all");
    % ||==> ovo se moze jos pojednostavniti !

    W(i, Index(i, :)) = vector_w_i / scalar_w_i;  % ./
    % ==> ne treba transponiranje: elementi stupca 
    % se redom pohrane kao elementi retka
    % ovo je provjera:
    if i == 1
        disp(vector_w_i / scalar_w_i)
        disp( W(i, Index(i, :)) )
        size( W(i, Index(i, :)) )
        sum( W(i, Index(i, :)) )
        sum( W(i, :) )
    end
end
%disp(w_i)
%size(w_i)
%sum(w_i)

%disp(B_i == C_i) % disp djeluje na matrici, pa je ovo unutra matrica ciji su elementi rezultati logickog uvjeta
%disp(C_i)
%disp(B_i)

% NE OVAKO:
%vec_w_i = G_i_inv * e;
% NEGO OVAKO:
vec_w_i_1 = G_i \ e;

% PROVJERA: ---------
vec_w_i_2 = sum(G_i_inv, 2);

%disp(vec_w_i_1)
%disp(vec_w_i_2)
% ==> OK ------------

% % PROVJERA JE LI OVO SVE JEDNAKO: --------
% scal_w_i_1 = sum(vec_w_i_2, 1) 
% scal_w_i_2 = sum( sum(G_i_inv, 2) , 1)
% scal_w_i_3 = sum(G_i_inv, "all")
% scal_w_i_4 = e' / G_i * e  % ne / pa \, jer je prvi rez vektor; matlab mnozi slijeva nadesno, iako je asocijativno
% scal_w_i_5 = e' * G_i_inv * e  % ovo nije dobro raditi !
% % JEST ==> OK ----------------------------

% ==> koristit cu scal_w_i_3, tj. tu "formulu" !

% ****************************************************************************** %

% **** 3. KORAK ***** racunanje Y ********************************************** %

% pratiti izvod s predavanja
% stvarno posložiti vektore w_i kao retke matrice W

% Frobeniusova norma definicija ?

% M = ....














