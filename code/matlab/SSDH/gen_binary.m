function binary = gen_binary (feature)

[fea_len, fea_size] = size(feature)
binary = zeros(fea_len * 2, fea_size)
for index = 1 : fea_size
  for i = 1 : fea_len
    if feature[i, index] < 0.25
      binary[2 * i - 1] = 0;
      binary[2*i] = 0;
    else if feature[i, index] < 0.5
      binary[2 * i - 1] = 0;
      binary[2*i] = 1;
    else if feature[i, index] < 0.75
      binary[2 * i - 1] = 1;
      binary[2*i] = 0;
    else 
      binary[2 * i - 1] = 1;
      binary[2*i] = 1;
    end
  end
end