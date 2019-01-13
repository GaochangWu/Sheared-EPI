function lf_dense3D=fun_3DSliceReconstruction(lf_volume, cur_fusionVolume,num_pym)
[H,W,C,~]=size(lf_volume);
ang_res_out=size(cur_fusionVolume,3);
% shear value segmentation
shear_acc=1;
cur_fusionVolume=round(cur_fusionVolume/shear_acc)*shear_acc;

for k=1:ang_res_out
    cur_fusionVolume(:,:,k) = medfilt2(cur_fusionVolume(:,:,k),[3,3]);
end

lf_dense3D=zeros(H,ang_res_out,W,C);
filter = [.0625, .25, .375, .25, .0625];
filter=filter/sum(filter);
parfor i_row=1:H
    EPI = lf_volume(i_row,:,:,:);
    EPI = permute(EPI,[4,2,3,1]);
    EPI_fusionMap = permute(cur_fusionVolume(i_row,:,:),[3,2,1]);
    [EPI_SRed, EPI_blendMap] = fun_shearReconstructionKernel(EPI, EPI_fusionMap);
    EPI_recon=fun_pyramidEPIBlending(EPI_SRed, EPI_blendMap, filter, num_pym);
    lf_dense3D(i_row,:,:,:)=EPI_recon;
end
lf_dense3D=permute(lf_dense3D,[1,3,4,2]);
