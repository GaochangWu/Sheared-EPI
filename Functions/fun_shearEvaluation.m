function scoreVolume=fun_shearEvaluation(modelPath, data)
load(modelPath);
data=single(data);
[H,W,~,depthRes]=size(data);
% Reconstruct cost volume
scoreVolume=zeros(H,W,depthRes);
parfor k=1:depthRes
    cur_slice = data(:,:,:,k);
    % conv 1
    layer = vl_nnconv(cur_slice, weight{1}, bias{1}, 'pad', 1, 'stride', 1, 'NoCudnn');
    conv_layer1 = vl_nnrelu(layer,[]);
    layer = vl_nnpool(conv_layer1, 1, 'Stride', [2, 2]);
    % conv 2
    layer = vl_nnconv(layer, weight{2}, bias{2}, 'pad', 1, 'stride', 1, 'NoCudnn');
    conv_layer2 = vl_nnrelu(layer,[]);
    layer = vl_nnpool(conv_layer2, 1, 'Stride', [2, 2]);
    % conv 3
    layer = vl_nnconv(layer, weight{3}, bias{3}, 'pad', 1, 'stride', 1, 'NoCudnn');
    conv_layer3 = vl_nnrelu(layer,[]);
    layer = vl_nnpool(conv_layer3, 1, 'Stride', [2, 2]);
    % conv 4
    layer = vl_nnconv(layer, weight{4}, bias{4}, 'pad', 1, 'stride', 1, 'NoCudnn');
    conv_layer4 = vl_nnrelu(layer,[]);
    
    % deconv 1
    layer = vl_nnconvt(conv_layer4, weight{5}, bias{5}, 'Upsample', [2 2], 'Crop', [0, 0, 0, 0]);
    layer = layer(1:size(conv_layer3,1),1:size(conv_layer3,2),:);
    deconv_layer1 = vl_nnrelu(layer,[]);
    layer = cat(3, deconv_layer1, conv_layer3);
    % deconv 2
    layer = vl_nnconvt(layer, weight{6}, bias{6}, 'Upsample', [2 2], 'Crop', [0, 0, 0, 0]);
    layer = layer(1:size(conv_layer2,1),1:size(conv_layer2,2),:);
    deconv_layer2 = vl_nnrelu(layer,[]);
    layer = cat(3, deconv_layer2, conv_layer2);
    % deconv 3
    layer = vl_nnconvt(layer, weight{7}, bias{7}, 'Upsample', [2 2], 'Crop', [0, 0, 0, 0]);
    layer = layer(1:size(conv_layer1,1),1:size(conv_layer1,2),:);
    deconv_layer3 = vl_nnrelu(layer,[]);
    layer = cat(3, deconv_layer3, conv_layer1);
    % conv 5
    layer = vl_nnconv(layer, weight{8}, bias{8}, 'pad', 1, 'stride', 1, 'NoCudnn');

    scoreVolume(:,:,k)=layer;
end
