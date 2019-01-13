
close all;

I = imread('imgs/image1.png');

if ~exist('.\mexJointWMF.mexw32', 'file')
    cd('..\..\complete_mex\win32');
    compileMex;
    copyfile('.\mexJointWMF.mexw32', '..\..\matlab_interface\win32\mexJointWMF.mexw32');
    cd('..\..\matlab_interface\win32\')
end

tic;
res = jointWMF(I,I,10,25.5,256,256,1,'exp');
toc;

figure, imshow(res);