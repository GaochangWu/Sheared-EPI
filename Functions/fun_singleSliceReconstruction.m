function imMultiView=fun_singleSliceReconstruction(imL,imR,disp,qualityDisp,pymLevel)
[H,W,C]=size(imL);
angularResOut=size(disp,3);
% shear value segmentation
shearAcc=1;
shearMap=round(disp/shearAcc)*shearAcc;

for k=1:angularResOut
    shearMap(:,:,k) = medfilt2(shearMap(:,:,k),[3,3]);
end
% shearMap=fun_consistencyCheck(shearMap);
% reconstruct multi-view

imMultiView=zeros(H,angularResOut,W,C);
parfor iRow=1:H
    epi=zeros(2,W,C);
    epi(1,:,:)=imL(iRow,:,:);
    epi(2,:,:)=imR(iRow,:,:);
    epiDisp=permute(shearMap(iRow,:,:),[3,2,1]);
    [epiSRed,epiSRedMask]=fun_shearReconstructionKernel(epi,epiDisp,qualityDisp);
    filter=[.0625, .25, .375, .25, .0625];
    filter=filter/sum(filter);
    curReconEPI=fun_pyramidEPIBlending(epiSRed,epiSRedMask,filter,pymLevel);
    imMultiView(iRow,:,:,:)=curReconEPI;
end
imMultiView=permute(imMultiView,[1,3,4,2]);
