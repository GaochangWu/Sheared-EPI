function curReconIm=fun_reconstructPyrmid(imPym,filter,RECONPYMLEVEL)
[H,W,C]=size(imPym{length(imPym)});
curReconIm=imPym{1};
FULLPYMLEVEL=length(imPym);
RECONPYMLEVEL=min([RECONPYMLEVEL,FULLPYMLEVEL]);

if filter==0
    if RECONPYMLEVEL>=2
        for iPymLevel=2:RECONPYMLEVEL
            curReconIm=imresize(curReconIm,[H,W*0.5^(FULLPYMLEVEL-iPymLevel)])+imPym{iPymLevel};
        end
    end
else
    % filter=[.0625, .25, .375, .25, .0625];
    if RECONPYMLEVEL>=2
        for iPymLevel=2:RECONPYMLEVEL
            curImHigh=zeros(size(curReconIm,1),2*size(curReconIm,2),C);
            curImHigh(:,1:2:end,:)=2*curReconIm;
            curImHigh = imfilter(curImHigh,filter,'symmetric');
            curReconIm=curImHigh+imPym{iPymLevel};
        end
    end
end