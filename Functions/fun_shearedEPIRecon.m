function outLF = fun_shearedEPIRecon(lf_input, model_path, num_pym, para, shear_range, ang_res_out)
%%
[H,W,C,~,~]=size(lf_input);
ang_res_in=size(lf_input,4);
idx_slice=[1,1,1,ang_res_in,0;
    1,ang_res_in,1,1,1;
    ang_res_in,1,ang_res_in,ang_res_in,0;
    ang_res_in,ang_res_in,ang_res_in,1,1];

lf_input_gray = squeeze(0.2989 * lf_input(:,:,1,:,:) + 0.5870 * lf_input(:,:,2,:,:) + 0.1140 * lf_input(:,:,3,:,:));
%% Shear Evaluation
input_chn=9;
fusionTensor_sparse=zeros(H,W-2*shear_range,size(idx_slice,1));

fprintf('Shear value evaluation...');
t=tic;
for i_slice=1:size(idx_slice,1)
    % Prepare data
    data=[];
    for k=1:2*shear_range+1
        lf_volume=[];
        d=k-shear_range-1;
        if idx_slice(i_slice,5)==0
            lf_volume(:,:,1)=lf_input_gray(:,1+shear_range:end-shear_range,idx_slice(i_slice,1),idx_slice(i_slice,2));
            lf_volume(:,:,2)=lf_input_gray(:,1+shear_range-d:end-shear_range-d,idx_slice(i_slice,3),idx_slice(i_slice,4));
        else
            lf_volume(:,:,1)=flip(lf_input_gray(:,1+shear_range:end-shear_range,idx_slice(i_slice,1),idx_slice(i_slice,2)),2);
            temp=lf_input_gray(:,:,idx_slice(i_slice,3),idx_slice(i_slice,4));
            temp=flip(temp,2);
            lf_volume(:,:,2)=temp(:,1+shear_range-d:end-shear_range-d);
        end
        lf_volume=permute(lf_volume,[3,2,1]);
        lf_volume=imresize(lf_volume,[input_chn,size(lf_volume,2)]);
        lf_volume=permute(lf_volume,[3,2,1]);
        data(:,:,:,k)=lf_volume;
    end
    
    % Reconstruct score volume
    scoreVolume = fun_shearEvaluation(model_path, data);
    
    % WMF for fusion map
    if idx_slice(i_slice,5)==0
        guided_SAI=lf_input(:,1+shear_range:end-shear_range,:,idx_slice(i_slice,1),idx_slice(i_slice,2));
    else
        guided_SAI=flip(lf_input(:,1+shear_range:end-shear_range,:,idx_slice(i_slice,1),idx_slice(i_slice,2)));
    end
    [~,fusionMap]=max(scoreVolume,[],3);
    fusionMap = jointWMF(fusionMap, uint8(guided_SAI*255), para.windowFilter,para.sigmaFilter,256,256, 1,'exp');
    fusionMap = medfilt2(fusionMap,[3,3]);

    if idx_slice(i_slice,5)==0
        fusionTensor_sparse(:,:,i_slice)=fusionMap-shear_range-1;
    else
        fusionTensor_sparse(:,:,i_slice)=flip(fusionMap-shear_range-1,2);
    end
end
t=toc(t);
fprintf(' consumes %2.2f seconds.\n',t);
%% Fusion map computation propagation
fprintf('Computing fusion maps...');
lf_input=lf_input(:,shear_range+1:W-shear_range,:,:,:);
[H,W,~,~,~]=size(lf_input);

t=tic;
fusionTensor_dense=fun_propagateFusionMap(fusionTensor_sparse, ang_res_in, ang_res_out);
t=toc(t);
fprintf(' consumes %2.2f seconds.\n',t);
%% Colume reconstruction
outLF=zeros(H,W,C,ang_res_out,ang_res_out);
fprintf('Reconstructing light field...');
t=tic;
for i=1:ang_res_in
    curColume=(i-1)*(ang_res_out-1)/(ang_res_in-1)+1;
    lf_volume=squeeze(lf_input(:,:,:,:,i));
    lf_volume=permute(lf_volume,[2,1,3,4]);
    
    cur_fusionVolume=squeeze(fusionTensor_dense(:,:,:,curColume));
    cur_fusionVolume=permute(cur_fusionVolume,[2,1,3]);
    lf_volume_dense=fun_3DSliceReconstruction(lf_volume, cur_fusionVolume, num_pym);

    lf_volume_dense=permute(lf_volume_dense,[2,1,3,4]);
    outLF(:,:,:,:,curColume)=lf_volume_dense;
end
%% Row reconstruction
for i=1:ang_res_out
    lf_volume=squeeze(outLF(:,:,:,i,1:(ang_res_out-1)/(ang_res_in-1):end));
    
    cur_fusionVolume=squeeze(fusionTensor_dense(:,:,i,:));
    lf_volume_dense=fun_3DSliceReconstruction(lf_volume,cur_fusionVolume,num_pym);
    
    lf_volume_dense=permute(lf_volume_dense,[1,2,3,5,4]);
    outLF(:,:,:,i,2:end-1)=lf_volume_dense(:,:,:,:,2:end-1);
end
t=toc(t);
fprintf(' consumes %2.2f seconds.\n',t);