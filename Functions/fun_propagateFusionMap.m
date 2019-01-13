function fusionVolume_dense=fun_propagateFusionMap(fusionTensor_sparse, ang_res_in, ang_res_out)
%%
% prepare parameters
[H,W,~]=size(fusionTensor_sparse);

sIdxIn=[0:1/(ang_res_in-1):1];
tIdxIn=[0:1/(ang_res_in-1):1];
countIm=0;
for i=1:ang_res_in
    for j=1:ang_res_in
        countIm=countIm+1;
        idx_fusionMap(:,countIm)=[sIdxIn(i),tIdxIn(j)];
    end
end


% propagate fusion map
fusionVolume_dense=zeros(H,W,ang_res_out,ang_res_out);
count_S=0;
for s = 0:1/(ang_res_out-1):1
    count_T=0;
    count_S=count_S+1;
    for t = 0:1/(ang_res_out-1):1
        count_T=count_T+1;
        for k=1:countIm
            alpha(k)=s-idx_fusionMap(1,k);
            beta(k)=t-idx_fusionMap(2,k);
            weight(k)=sqrt(sum((idx_fusionMap(:,k)'-[1-s,1-t]).^2));
        end
        sumWeight=sum(weight);
        weight=weight/sumWeight;

        for k=1:countIm
            map_warped{k}=ones(H,W)*-255;
            for d = min(fusionTensor_sparse(:)):max(fusionTensor_sparse(:))
                curY = round(-d*alpha(k));
                curX = round(-d*beta(k));
                mask = double(fusionTensor_sparse(:,:,k)==d);
                y = [max(1,1-curY):min(H,H-curY)];
                x = [max(1,1-curX):min(W,W-curX)];
                y1 = [max(1,1+curY):min(H,H+curY)];
                x1 = [max(1,1+curX):min(W,W+curX)];
                
                cur_map = ones(H,W)*-255;
                cur_map(y1,x1) = cur_map(y,x).*(1-mask(y,x)) + fusionTensor_sparse(y,x,k).*mask(y,x);
                mask = double(cur_map==-255);
                
                map_warped{k}=map_warped{k}.*mask+cur_map.*(1-mask);
            end
        end
        
        mask = [];
        for k=1:countIm
            mask{k}=map_warped{k}~=-255;
        end

        map_novel=zeros(H,W);
        mask_weight=zeros(H,W);
        for k=1:countIm
            map_novel=map_novel+map_warped{k}.*mask{k}.*weight(k);
            mask_weight=mask_weight+mask{k}.*weight(k);
        end
        mask_weight(mask_weight==0)=1;
        map_novel=map_novel./mask_weight;
        fusionVolume_dense(:,:,count_S,count_T)=map_novel;
    end
end