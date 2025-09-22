function finerIndices = upsampleIndicesToFinerLevel(mask, finerSize)
    [r, c] = find(mask);

    r_up = [r*2 - 1; r*2; r*2 - 1; r*2];
    c_up = [c*2 - 1; c*2 - 1; c*2; c*2];

    valid = r_up <= finerSize(1) & c_up <= finerSize(2);
    r_up = r_up(valid);
    c_up = c_up(valid);

    finerIndices = sub2ind(finerSize, r_up, c_up);
end

