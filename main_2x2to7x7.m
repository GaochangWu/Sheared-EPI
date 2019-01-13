clear;clc;close all;
% Setting parameters
sceneName = 'flowers_plants_7_eslf';
sceneFolder = './Scenes/';  % You can download Lytro light fields at     http://lightfields.stanford.edu/LF2016.html     or     http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/
%% Initialize parameters
shear_range = 4;
num_pym = 3;

s_res = 14;
t_res = 14;
ang_start = 5;
ang_res_out = 7;
ang_res_in = 2;
tone_coef = 2.25;

para.windowFilter = 5;
para.sigmaFilter = 15.5;
up_scale = fix( (ang_res_out-1)/(ang_res_in-1) );
idx_in = [1:up_scale:ang_res_out];
model_path = './Model/model.mat';

% prepare functions
addpath(genpath('Functions'));
addpath(genpath('./matconvnet/matlab'));
%% Load input light field
fprintf('Loading input light field ...\n');
num_pool = 3;                                                     % A fixed parameter for encoder-decoder network
crop_rate = max(num_pym,num_pool);
[lf_gt, lf_input] = fun_load4DLF([sceneFolder,sceneName], s_res, t_res, ang_start, ang_res_out, ang_res_in, 1.0, crop_rate, shear_range);
W = size(lf_gt, 2);
figure(1); imshow(mean(mean(lf_input,4),5));
%% Sheared EPI reconstruction
ParallelComputing = parpool;
lf_recon = fun_shearedEPIRecon(lf_input, model_path, num_pym, para, shear_range, ang_res_out);
delete(ParallelComputing);
%% Evaluation
MakeDir(['./Result/',sceneName]);
bodercut_Y=22;
bodercut_X=fix((W-498)/2);

PSNR=0;
SSIM=0;
K=0;
for i = 1:ang_res_out
    for j = 1:ang_res_out
        GT = lf_gt(bodercut_Y+1:end-bodercut_Y, 1+bodercut_X:end-bodercut_X, :, i, j);
        im = lf_recon(bodercut_Y+1:end-bodercut_Y, 1+bodercut_X:end-bodercut_X, :, i, j);
        [psnr(i, j), ssim(i, j)] = compute_psnr(uint8(GT*255), uint8(im*255), 0);
        if ismember(i, idx_in) && ismember(j, idx_in)
        else
            K=K+1;
            SSIM=SSIM+ssim(i,j);
            PSNR=PSNR+psnr(i,j);
        end
        if tone_coef>1
            im = fun_adjustTone(im, tone_coef);
        end
        for k=1:1
        	figure(1);imshow(im);
        end
%         imwrite(im,['./Result/',sceneName,'/out_0',num2str(i),'0',num2str(j),'.png']);
    end
end
PSNR=PSNR/K;
SSIM=SSIM/K;
fprintf('\nMean PSNR and SSIM on synthetic views: %2.2f, %0.4f\n', PSNR, SSIM);