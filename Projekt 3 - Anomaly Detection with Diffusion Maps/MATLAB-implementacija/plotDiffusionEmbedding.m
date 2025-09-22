function plotDiffusionEmbedding(diffusion_map, lambda)
    % Plots eigenvalues, 2D/3D diffusion map, and coordinates per sample
    %
    % INPUTS:
    %   diffusion_map - n x d matrix of diffusion coordinates
    %   lambda        - d+1 x 1 vector of eigenvalues (first one ignored)

    numSamples = size(diffusion_map, 1);
    numCoords = size(diffusion_map, 2);

    % === 1. Plot eigenvalues
    figure;
    plot(lambda, 'o-'); ylim([0, 1]); grid on;
    xlabel('Index'); ylabel('\lambda'); title('Eigenvalues of P_s');
    % gcf = get current figure
    saveas(gcf, fullfile('output', 'eigenvalues_plot.png'));

    % === 2. 3D scatter if 3+ dimensions
    if numCoords >= 3
        figure;
        scatter3(diffusion_map(:, 1), diffusion_map(:, 2), diffusion_map(:, 3), ...
                 12, diffusion_map(:, 1), 'filled');
        xlabel('\psi_1'); ylabel('\psi_2'); zlabel('\psi_3');
        title('3D Diffusion Map Embedding (t = 1)');
        grid on;
        saveas(gcf, fullfile('output', 'diffusion_map_3D.png'));
    end

    % === 3. 2D scatter with coloring
    figure;
    scatter(diffusion_map(:, 1), diffusion_map(:, 2), 12, diffusion_map(:, 1), 'filled');
    xlabel('\psi_1'); ylabel('\psi_2');
    title('2D Diffusion Map Embedding (t = 1)');
    axis equal; grid on; colorbar;
    saveas(gcf, fullfile('output', 'diffusion_map_2D.png'));

    % === 4. Line plot of each coordinate across samples
    figure;
    hold on;
    for j = 1 : numCoords
        scatter(1:numSamples, diffusion_map(:, j), 8, '.');
    end
    hold off;
    xlabel('Sample Index'); ylabel('\psi_j');
    title('Diffusion Coordinates over Samples');
    legend(arrayfun(@(j) sprintf('\\psi_{%d}', j), 1 : numCoords, 'UniformOutput', false), ...
       'Interpreter', 'tex', 'Location', 'best');
    grid on;
    saveas(gcf, fullfile('output', 'diffusion_coordinates_over_samples.png'));
end

