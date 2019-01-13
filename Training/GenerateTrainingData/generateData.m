clear;clc;close all;
save_mode=true; %true false
noise_mode=true;
trainDataFolder='../TrainDataLF';  
folders=dir(trainDataFolder);
folders(1:2)=[];
folderNum=length(folders);
trainDataSavePath = './training_data.h5';
%% Parameters setting and kernel defination
angResOut=9;
angResOut2=2;
searchRange=20;
halfWinSizeV=2;
stride=16;
thrVarError=0.01;
size_input=31;
%%
chunksz = 20;
created_flag = false;
totalct = 0;
for iFolder=1:folderNum
    if iFolder>=12&&iFolder<=19
        downRatio=1;
    else
        downRatio=2:3;
    end
    for curRatio=downRatio
        count=0;
        data = zeros(size_input, size_input, angResOut2, 1);
        label = zeros(size_input, size_input, 1, 1);

        imLF=[];
        current_folder=folders(iFolder).name;
        current_images=dir([trainDataFolder,'/',current_folder]);
        for i=1:angResOut
            im=imread([trainDataFolder,'/',current_folder,'/',current_images(i+2).name]);
            im=imresize(im,1/curRatio);
            imLF(:,:,:,i)=im2double(rgb2gray(im));
        end
        [H,W,~]=size(imLF);
        C=halfWinSizeV*2+1;
        
        [X,Y]=meshgrid(1:W,1:angResOut);
        curData=[];
        curLabel=[];
        for k=1:2*searchRange+1
            imLFLow=[];
            d=k-searchRange-1;
            imLFLow(:,:,:,1)=imLF(:,1+searchRange:end-searchRange,:,1);
            imLFLow(:,:,:,2)=imLF(:,1+searchRange+d:end-searchRange+d,:,end);
            imLFHigh=imresize(permute(imLFLow,[4,2,3,1]),[angResOut,size(imLFLow,2)]);
            imLFHigh=permute(imLFHigh,[4,2,3,1]);
            
            shearField=([0:angResOut-1]*d/(angResOut-1))';
            shearField=repmat(shearField,[1,W]);
            imLFSheared=imLFHigh;
            for h=1:H
                curEPI=permute(imLF(h,:,:,:),[4,2,3,1]);
                for c=1:size(curEPI,3)
                    curEPI=interp2(X,Y,curEPI(:,:,c),X+shearField,Y,'linear');
                end
                imLFSheared(h,:,:,:)=permute(curEPI(:,1+searchRange:end-searchRange,:),[4,2,3,1]);
            end
            LFError=abs(imLFSheared-imLFHigh);
            LFError=sum(LFError(:,:,:,2:end-1),4);
            LFError(isnan(LFError))=0;
            LFError=jointWMF(exp(-LFError/0.05), uint8(imLFHigh(:,:,:,1)*255), 5, 15.5, 256, 256, 1,'exp');
%             LFError=exp(-LFError/0.05);
            imLFHigh=imresize(permute(imLFLow,[4,2,3,1]),[angResOut2,size(imLFLow,2)]);
            imLFHigh=permute(imLFHigh,[4,2,3,1]);
            imLFHigh=min(max(imLFHigh,0),1);
            imLFHigh=permute(imLFHigh,[1,2,4,3]);
            curData(k,:,:,:)=imLFHigh;
            curLabel(k,:,:)=LFError;
        end
        if (noise_mode)
            curDataN=imnoise(curData,'gaussian',0,0.0008);
            curData=cat(1,curData,curDataN);
            curLabel=cat(1,curLabel,curLabel);
        end
        
        for y = 1 : stride : size(curLabel,2)-size_input+1
            for x = 1 :stride : size(curLabel,3)-size_input+1
                for k = 1:size(curLabel,1)
                    subim_input = squeeze(curData(k, y : y+size_input-1, x : x+size_input-1, :));
                    subim_label = squeeze(curLabel(k, y : y+size_input-1, x : x+size_input-1));
                    temp = subim_input(:,:,1);
                    if nanvar(temp(:))>=thrVarError
                        count=count+1;
                        data(:, :, :, count) = subim_input;
                        label(:, :, :, count) = subim_label;
                    end
                end
            end
        end
        numPatch = size(data,4);
        order = randperm(numPatch);
        data = data(:, :, :, order);
        label = label(:,:,:, order);
        fprintf('Current number of training data is %d.\n', numPatch);

        % writing to HDF5
        if (save_mode)
            for batchno = 1:floor(numPatch/chunksz)
                last_read=(batchno-1)*chunksz;
                batchdata = data(:,:,:, last_read+1:last_read+chunksz); 
                batchlabs = label(:,:,:, last_read+1:last_read+chunksz);

                startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
                curr_dat_sz = store2hdf5(trainDataSavePath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
                created_flag = true;
                totalct = curr_dat_sz(end);
            end
            h5disp(trainDataSavePath);
        end
    end
end