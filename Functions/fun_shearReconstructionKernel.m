function [EPI_SRed, EPI_blendMap]=fun_shearReconstructionKernel(EPI, EPI_fusionMap)
[angularResIn,W,C]=size(EPI);
if angularResIn>=4
    method_interp='bicubic';
else
    method_interp='linear';
end
ang_res_out=size(EPI_fusionMap,1);

kShear=0;
shearValue=[];
for i_shear=min(EPI_fusionMap(:)) : max(EPI_fusionMap(:))
    cur_EPI_blend_mask=(EPI_fusionMap==i_shear);
    if sum(cur_EPI_blend_mask(:))~=0
        kShear = kShear+1;
        shearValue(kShear) = i_shear;
    end
end

EPI_SRed=zeros(ang_res_out,W,C,kShear);
EPI_blendMap=zeros(ang_res_out,W,kShear);
kShear=0;
epiSRedRef=imresize(EPI,[ang_res_out,W]);
[xIn,yIn]=meshgrid(1:W,1:angularResIn);
[xOut,yOut]=meshgrid(1:W,1:ang_res_out);
for i_shear=shearValue
    kShear=kShear+1;
    cur_EPI_blend_mask=(EPI_fusionMap==i_shear);
    
    % Construct shear field
    unshearField=zeros(ang_res_out,W);
    for k=1:ang_res_out
        unshearField(k,:)=-(k-1)*i_shear/(ang_res_out-1) +ceil(i_shear/2);
    end
    shearField=unshearField(1: round((ang_res_out-1)/(angularResIn-1)) :end,:);
    
    % Shear EPI & sparse fusion map
    EPI_sheared=EPI;
    for i_color=1:C
        tempIm=interp2(xIn,yIn,EPI_sheared(:,:,i_color),xIn+shearField,yIn,method_interp);
        I=find(isnan(tempIm));
        tempIm2=EPI(:,:,i_color);
        tempIm(I)=tempIm2(I);
        EPI_sheared(:,:,i_color)=tempIm;
    end

    % EPI super-resolution
    epiShearedSR=imresize(EPI_sheared,[ang_res_out,W],'bicubic');
    
    % Unshear EPI
    epiUnshearedSR=epiShearedSR;
    for i_color=1:C
        tempIm=interp2(xOut,yOut,epiShearedSR(:,:,i_color),xOut-unshearField,yOut,'bicubic');
        epiUnshearedSR(:,:,i_color)=tempIm;
    end
    I=find(isnan(epiUnshearedSR));
	epiUnshearedSR(I)=epiSRedRef(I);
    EPI_SRed(:,:,:,kShear)=epiUnshearedSR;
    EPI_blendMap(:,:,kShear)=cur_EPI_blend_mask;
end