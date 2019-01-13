function imPym=fun_constructGaussianPyrmid(im,filter,PYMLEVEL)
[H,W,C]=size(im);
W=floor(W*0.5^(PYMLEVEL-1))*2^(PYMLEVEL-1);
im=imresize(im,[H,W]);
if filter==0
    curReconIm=im;
    for iPymLevel=1:PYMLEVEL
        curReconIm=imresize(curReconIm,[H,W*0.5^(iPymLevel-1)]);
        imPym{PYMLEVEL-iPymLevel+1}=curReconIm;
    end
else
    % filter=[.0625, .25, .375, .25, .0625];
    imPym=cell(1,PYMLEVEL);
    preImLow=im;
    imPym{PYMLEVEL}=preImLow;
    for iPymLevel=2:PYMLEVEL
        preImLow=imfilter(preImLow,filter,'symmetric');
        preImLow = preImLow(:, 1:2:end, :);
        imPym{PYMLEVEL-iPymLevel+1}=preImLow;
    end
end
