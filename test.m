function test( gpuIDs )

if isempty(gcp('nocreate'))
    parpool(numel(gpuIDs));
end
spmd
    gpuDevice(gpuIDs(labindex));
    a = {};
    a{1} = gpuArray.randn(512,4096,7,7,'single');
    a{end+1} = gpuArray.randn(4096,4096,1,1,'single');
    a{end+1} = gpuArray.randn(1000,4096,1,1,'single');
    a{end+1} = gpuArray.randn(256,512,3,3,'single');
    a{end+1} = gpuArray.randn(256,256,3,3,'single');
    a{end+1} = gpuArray.randn(256,256,3,3,'single');
    a{end+1} = gpuArray.randn(128,256,3,3,'single');
    a{end+1} = gpuArray.randn(128,128,3,3,'single');
    a{end+1} = gpuArray.randn(96,128,3,3,'single');
    a{end+1} = gpuArray.randn(64,96,3,3,'single');
    a{end+1} = gpuArray.randn(3,64,11,11,'single');
    b = cell(size(a));
    gpuop('init', 'GPUP2PMMapFile', numel(gpuIDs));

    tic;
    for i=1:numel(a)
        b{i} = gpuop('add', a{i});
    end
    toc;
    gpuop('close');
end



end

