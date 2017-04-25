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
model:cuda()
criterion:cuda()
model:evaluate()

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

local out = assert(io.open("result.csv", "w")) -- open a file to write

for i = 1, nImages do
	test_path = testImagePath[i]
	test_image = loadImage(test_path)
	input_img = test_image

	local size = 32
	local w1 = math.ceil((input_img:size(3)-size)/2)
	local h1 = math.ceil((input_img:size(2)-size)/2)
	local tmp = image.crop(input_img, w1, h1, w1+size, h1+size)
	input_img = tmp
	print(input_img:size())
	input_img:resize(1, 3, size, size)

	result = model:forward(input_img):cuda()
	exp = torch.exp(result)
	exp_sum = exp:sum()
	exp = torch.div(exp, exp_sum)

	out:write(test_path)
	out:write(",")
	out:write(exp[1][2])
	out:write("\n")
end
out:close()
