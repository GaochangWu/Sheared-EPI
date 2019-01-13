clear;clc;close all;
sceneFile = '.\Scenes\30scenes\';  % .\Scenes\StanfordMicro05\   G:/Dataset/Extra/
im_list = dir(strcat(sceneFile,'/*.png'));
model_path = './Model/model.mat';
resultFile = strcat(sceneFile,'Results/');
%% Setting parameters
flag_write = 1;
shear_range = 9;
num_pym = 3;

s_res = 14;
t_res = 14;
ang_start = 5;
ang_res_out = 7;
ang_res_in = 2;
tone_coef = 2.25;

para.windowFilter=5;
para.sigmaFilter=15.5;
up_scale = (ang_res_out-1)/(ang_res_in-1);
idx_in = [1:up_scale:ang_res_out];

% prepare functions
addpath(genpath('Functions'));
addpath(genpath('./matconvnet/matlab'));
%%
PSNR_Batch = 0;
SSIM_Batch = 0;
fid = fopen([resultFile,'Log.txt'], 'wt');
for n = 1:length(im_list)
    sceneName = im_list(n).name;
    sceneName = sceneName(1:end-4);
%% Preparing
    MakeDir([resultFile, sceneName,'/images']);
    fidn = fopen([resultFile,sceneName,'/Log.txt'], 'wt');
%% Load input light field
    fprintf('Scene %d of %d, name: %s\n', n, length(im_list), sceneName);
    num_pool = 3;                                                     % A fixed parameter for encoder-decoder network
    crop_rate = max(num_pym,num_pool);

    [lf_gt, lf_input] = fun_load4DLF([sceneFile,sceneName], s_res, t_res, ang_start, ang_res_out, ang_res_in, 1.0, crop_rate, shear_range);
    W = size(lf_gt, 2);
%% Sheared EPI reconstruction
    lf_recon = fun_shearedEPIRecon(lf_input, model_path, num_pym, para, shear_range, ang_res_out);
%% Evaluation
    bodercut_Y=22;
    bodercut_X=fix((W-498)/2);
    PSNR=0;
    SSIM=0;
    K=0;
    for i=1:ang_res_out
        for j=1:ang_res_out
            im=lf_recon(bodercut_Y+1:end-bodercut_Y,1+bodercut_X:end-bodercut_X,:,i,j);
            gt=lf_gt(bodercut_Y+1:end-bodercut_Y,1+bodercut_X:end-bodercut_X,:,i,j);
            [curPSNR(i,j),curSSIM(i,j)] = compute_psnr(uint8(gt*255),uint8(im*255),0);
            if ismember(i,idx_in) && ismember(j,idx_in)
            else
                K=K+1;
                PSNR=PSNR+curPSNR(i,j);
                SSIM=SSIM+curSSIM(i,j);
            end
            if flag_write
                imwrite(im,[resultFile,sceneName,'/images/',sprintf('%02d',i),'_',sprintf('%02d',j),'.png']);
            end
        end
    end
    fprintf('\nMean PSNR and SSIM on synthetic views: %2.2f, %0.4f\n\n', PSNR/K, SSIM/K);
    if flag_write
        fprintf(fidn, 'Mean PSNR and SSIM on synthetic views: %2.2f, %0.4f\n', PSNR/K, SSIM/K);
    end
    fclose(fidn);
    PSNR_Batch=PSNR_Batch+PSNR/K;
    SSIM_Batch=SSIM_Batch+SSIM/K;
end
fprintf('\nMean PSNR and SSIM on the dataset: %2.2f, %0.4f\n', PSNR_Batch/n, SSIM_Batch/n);
fprintf(fid, 'Mean PSNR and SSIM on the dataset: %2.2f, %0.4f\n', PSNR_Batch/n, SSIM_Batch/n);
fclose(fid);