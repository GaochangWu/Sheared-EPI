
close all;

I = imread('imgs/image1.png');

if ~exist('.\mexJointWMF.mexw32', 'file')
    compileMex;
end

tic;
res = jointWMF(I,I,10,25.5,256,256,1,'exp');
toc;

figure, imshow(res);