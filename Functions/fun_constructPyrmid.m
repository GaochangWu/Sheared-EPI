function imPym=fun_constructPyrmid(im,filter,PYMLEVEL)
[H,W,C]=size(im);
W=floor(W*0.5^(PYMLEVEL-1))*2^(PYMLEVEL-1);
im=imresize(im,[H,W]);

imPym{1}=imresize(im,[H,W*0.5^(PYMLEVEL-1)]);
if filter==0
    curReconIm=zeros(size(imPym{1},1),size(imPym{1},2),C);
    for iPymLevel=2:PYMLEVEL
        curReconIm=imresize(curReconIm+imPym{iPymLevel-1},[H,W*0.5^(PYMLEVEL-iPymLevel)]);
        imPym{iPymLevel}=imresize(im,[H,W*0.5^(PYMLEVEL-iPymLevel)])-curReconIm;
    end
else
    imPym=cell(1,PYMLEVEL);
    % filter=[.0625, .25, .375, .25, .0625];
    preImLow=im;
    for iPymLevel=1:PYMLEVEL-1
        curImLow = preImLow;
        % low pass, convolve with filter
        curImLow = imfilter(curImLow,filter,'symmetric');
        curImLow = curImLow(:, 1:2:end, :);
        % upsampling
        curImHigh=zeros(size(curImLow,1),2*size(curImLow,2),C);
        curImHigh(:,1:2:end,:)=2*curImLow;
        curImHigh = imfilter(curImHigh,filter,'symmetric');
        imPym{PYMLEVEL-iPymLevel+1}=preImLow-curImHigh;
        preImLow=curImLow;
    end
    imPym{1}=preImLow;
end