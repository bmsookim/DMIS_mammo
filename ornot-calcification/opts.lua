-- *******************************************************************
--  Copyright (c) 2016, DMIS, Digital Mammography DREAM Challenge Team.
--  All rights reserved.
--
--  (Author) Bumsoo Kim, 2016
--  Github : https://github.com/meliketoy/DreamChallenge
--
--  Korea University, Data-Mining Lab
--  Digital Mammography DREAM Challenge Torch Implementation
-- *******************************************************************

local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    'dreamChallenge', 'Options: dreamChallenge')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   cmd:option('-precision',  'single',   'Options: single | double | half') -- precision(?)
   ------------- Data options ------------------------
   cmd:option('-nThreads',        16, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'modelState',  'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              1e-2,  'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-3,  'weight decay')
   cmd:option('-dropout',         0,   'dropout ratio')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'preresnet', 'Options: resnet | preresnet | wide-resnet')
   cmd:option('-depth',        50,          'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-widen_factor', 10,          'ResNet width: 10 | 12 | 20 | ...', 'number')
   cmd:option('-shortcutType', '',          'Options: A | B | C')
   cmd:option('-retrain',      'none',      'Path to model to retrain with')
   cmd:option('-optimState',   'none',      'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false',  'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'
   opt.nGPU = cutorch.getDeviceCount()

   opt.save = paths.concat(opt.save, opt.dataset, opt.netType, opt.depth)
   if opt.resume ~= 'none' then
       opt.resume = paths.concat(opt.resume, opt.dataset, opt.netType, opt.depth)
   end

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create modelState directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'dreamChallenge' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing Dream Challenge data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: Dream Challenge missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 60 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end
   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
