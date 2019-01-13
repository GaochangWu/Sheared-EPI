function [lf_gt, lf_input] = fun_load4DLF(scenePath, s_res, t_res, ang_start, ang_res_out, ang_res_in, tone_coef, crop_rate, shear_range)
scale = round( (ang_res_out-1)/(ang_res_in-1) );
try
    input_lf = im2double(imread([scenePath,'.jpg']));
catch
    input_lf = im2double(imread([scenePath,'.png']));
end

input_lf = fun_adjustTone(input_lf,tone_coef);
h = size(input_lf, 1) / t_res;
w = size(input_lf, 2) / s_res;
lf_gt = zeros(h, w, 3, t_res, s_res);
for ax = 1 : s_res
    for ay = 1 : t_res
        lf_gt(:, :, :, ay, ax) = input_lf(ay:t_res:end, ax:s_res:end, :);
    end
end

lf_gt = lf_gt(:, :, :, ang_start:ang_start+ang_res_out-1, ang_start:ang_start+ang_res_out-1);
lf_input = lf_gt(:, :, :, [1:scale:ang_res_out], [1:scale:ang_res_out]);

% Crop for pyramid and encoder-decoder structure

[H,W,~,~,~]=size(lf_gt);
H=floor(H/(2^crop_rate))*(2^crop_rate);
W=floor((W-2*shear_range)/(2^crop_rate))*(2^crop_rate)+2*shear_range;
lf_gt=lf_gt(1:H,1+shear_range:W-shear_range,:,:,:);
lf_input=lf_input(1:H,1:W,:,:,:);



