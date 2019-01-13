function curReconEPI=fun_pyramidEPIBlending(EPI_SRed,EPI_blendMap,filter,num_pym)
[ang_res_out,W,C,num_shear]=size(EPI_SRed);
EPI_pyramid=cell(size(EPI_SRed,4),num_pym);
weight_pyramid=cell(size(EPI_SRed,4),num_pym);
blended_pyramid=fun_constructPyrmid(zeros(ang_res_out,W,C),filter,num_pym);

for iShear=1:num_shear
    EPI_pyramid(iShear,:)=fun_constructPyrmid(EPI_SRed(:,:,:,iShear),filter,num_pym);
    weight_pyramid(iShear,:)=fun_constructGaussianPyrmid(EPI_blendMap(:,:,iShear),filter,num_pym);
    for i_pym=1:num_pym
        weight = repmat(weight_pyramid{iShear,i_pym},[1 1 3]);
        blended_pyramid{i_pym}=blended_pyramid{i_pym}+weight.*EPI_pyramid{iShear,i_pym};
    end
end
curReconEPI=fun_reconstructPyrmid(blended_pyramid,filter,num_pym);