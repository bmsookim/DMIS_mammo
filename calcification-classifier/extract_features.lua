require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'image'

local DataLoader = require 'dataloader'
local models = require 'networks/init'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local t = require 'datasets/transforms'
local datasets = require 'datasets/init'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

local checkpoint, optimState = checkpoints.best(opt)

local model, criterion = models.setup(opt, checkpoint)
local model2, _ = models.setup(opt, checkpoint)

-- Remove the fully connected layer
assert(torch.type(model.modules[1]:get(#model.modules[1].modules)) == 'nn.Linear')
model.modules[1]:remove(#model.modules[1].modules)

-- Evaluate mode
model:cuda()
model2:cuda()
criterion:cuda()
model:evaluate()
model2:evaluate()

local function findImages(dir)
	local imagePaths = torch.CharTensor()
	local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG'}
	local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
	for i = 2, #extensionList do
		findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
	end

	local f = io.popen('find -L ' .. dir .. findOptions)

	local maxLength = -1
	local imagePaths = {}

	while true do
		local line = f:read('*line')
		if not line then break end
		local dirname = paths.dirname(line)
		local filename = paths.basename(line)
		local path = dirname .. '/' .. filename

		table.insert(imagePaths, path)
		maxLength = math.max(maxLength, #path + 1)
	end

	f:close()

	local nImages = #imagePaths
	--print(nImages)
	return imagePaths, nImages
end

testImagePath, nImages = findImages(opt.data .. '/val/')

function loadImage(path)
	local ok, input = pcall(function()
		return image.load(path, 3, 'float')
	end)

	if not ok then
		local f = io.open(path, 'r')
		assert(f, 'Error reading: ' .. tostring(path))
		local data = f:read('*a')
		f:close()

		local b = torch.ByteTensor(string.len(data))
		ffi.copy(b:data(), data, b:size(1))

		input = image.decompress(b, 3, 'float')
	end
	return input
end

local meanstd = {
	mean = {0.358, 0.358, 0.358},
	std = {0.166, 0.166, 0.166},
}

file_num = 0

-- local out = assert(io.open("result.csv", "w")) -- open a file to write

for i = 1, nImages do
	test_path = testImagePath[i]
	test_image = loadImage(test_path)

	input_img = test_image
	for i=1,3 do
		input_img[i]:add(-meanstd.mean[i])
		input_img[i]:div(meanstd.std[i])
	end

	local size = 224
	local w1 = math.ceil((input_img:size(3)-size)/2)
	local h1 = math.ceil((input_img:size(2)-size)/2)
	local tmp = image.crop(input_img, w1, h1, w1+size, h1+size)
	input_img = tmp
	input_img:resize(1, 3, 224, 224)

	result = model:forward(input_img):float()
	sc = model2:forward(input_img):float()

	exp = torch.exp(sc)
	exp_sum = exp:sum()
	exp = torch.div(exp, exp_sum)

	print(test_path)
	print(result[1]:size())
	print(exp[1][2])

	file_num = file_num + 1
	torch.save('./vectors/features'..file_num..'.t7',
		{path = test_path,
		 features = result[1],
		 score = exp[1][2]
		})

end
-- out:close()
