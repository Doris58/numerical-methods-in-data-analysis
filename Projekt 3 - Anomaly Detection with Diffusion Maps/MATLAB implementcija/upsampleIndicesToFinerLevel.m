function upsampledIndices = upsampleIndicesToFinerLevel(dmIndices, coarseSize, fineSize)
% UPSAMPLEINDICESTOFINERLEVEL maps suspicious patch indices from a coarse
    % pyramid level to the next finer resolution (2x upsampling).
    %
    % INPUTS:
    %   dmIndices     - linear indices (1D) of suspicious patches in coarse image
    %   coarseSize - [height, width] of the coarse image (e.g., [64, 64])
    %   fineSize   - [height, width] of the finer image (e.g., [128, 128])
    %
    % OUTPUT:
    %   upsampledIndices - linear indices in the finer image

    [rows, cols] = ind2sub(coarseSize, dmIndices);
    
    % 1 piksel ==> 4 piksela
    rows_up = [2*rows - 1; 2*rows;     2*rows - 1; 2*rows];
    cols_up = [2*cols - 1; 2*cols - 1; 2*cols;     2*cols];
     
    valid = rows_up <= fineSize(1) & cols_up <= fineSize(2);
    rows_up = rows_up(valid);
    cols_up = cols_up(valid);
     
    upsampledIndices = sub2ind(fineSize, rows_up, cols_up);
    
end




    
   
    