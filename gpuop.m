function varargout = gpuop( op, varargin )
% GPUOP GPU Operation
% USAGE
%   P2PAva = GPUOP('check', GPUID1, GPUID2)
%   Check if p2p access available on two gpus
%
%   P2PAvaStruct = GPUOP('check', GPUIDs)
%   Check if p2p access available on these gpus
%
%   GPUOP('init', MMAPFileName, NGPU)
%   MMAPFileName: File name for mmap function
%   NGPU: Number of GPUs to be used
%
%   Result = GPUOP('add', Array)
%   Array: The variable to be sumed across specified gpus(labs).
%
%   GPUOP('close')
%   Clear and free allocated memory.

switch op
    case 'check'
        if nargin <1
            error('Plase enter device IDs.');
        elseif nargin ==3
            varargout{1} = gpup2p(int32(-1), int32(varargin{1}), int32(varargin{2}));
            return;
        elseif nargin ==2
            ids = varargin{1};
        end
        if numel(ids) <1
            error('Plase enter device IDs.');
        end
        ids = int32(ids);
        out = struct('ID1', [], 'ID2', [], 'P2P', []);
        for i=1:numel(ids)
            for j=(i+1):numel(ids)
                out(end+1).ID1 = ids(i);
                out(end).ID2 = ids(j);
                out(end).P2P = gpup2p(int32(-1), ids(i), ids(j));
            end
        end
        out(1) = [];
        varargout{1} = out;

    % ===============================================================
    case 'init'
        if ~ischar(varargin{1})
            error('file name must be a string.');
        end
        if ~isreal(varargin{2})
            error('nProc must be a number.');
        end
        p = fullfile(pwd, varargin{1});
        gpup2p(int32(0), p, int32(varargin{2}));
        labBarrier();

    % ===============================================================
    case 'add'
        % add all data in gpu, and clear stored data pointer
        gpup2p(int32(1), int32(labindex), varargin{1});
        labBarrier;
        gpup2p(int32(2), int32(labindex));
        labBarrier;
        varargout{1} = gpup2p(int32(3), int32(labindex));
        labBarrier;

     % ===============================================================
    case 'close'
        gpup2p(int32(4));

     % ===============================================================
    otherwise
      error(['Unknown operation: ', op]);
end

end