require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,64):add(nn.Dropout(0.3))   --200/2 = 100
ConvBNReLU(64,64)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)
vgg:add(MaxPooling(2,2,2,2):ceil())     -- 100/2 = 50

ConvBNReLU(128,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256)
vgg:add(MaxPooling(2,2,2,2):ceil())   -- 50/2 = 25

ConvBNReLU(256,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
vgg:add(MaxPooling(2,2,2,2):ceil())    -- 25/2 = 13

ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
vgg:add(MaxPooling(5,5,5,5):ceil())   --- 13/5 = 3
vgg:add(nn.View(512*3*3))    

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512*3*3,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,7))
vgg:add(classifier)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
