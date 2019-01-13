close all;
imGuide = imread('im0.png');
disp = double(imread('dispCCed.png'))*2;
[H,W]=size(disp);
imGuide=imresize(imGuide,[H,W]);

% disp: disparity map
% imGuide: guidence map
% winSize: window size
%sigma: standard deviation of the Gaussian kernel.
%iter: iteration times


winSize=10;
sigma=25.5;
iter=2;
tic;
res = jointWMF(disp,imGuide,winSize,sigma,256,256,iter,'exp');
toc;

figure, imagesc(res);