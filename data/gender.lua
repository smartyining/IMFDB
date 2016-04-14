require 'nn'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

-- load data file
allData = torch.load('data_good_resized.t7')
allLabel = torch.load('gender_label_good.t7')
N_sample = #allLabel

function splitDataset()
   local ratio = 0.7
   local shuffle = torch.randperm(N_sample)

   local numTrain = math.floor(N_sample * ratio)
   local numTest = N_sample- numTrain

   local train = torch.Tensor(numTrain, 3, 200, 200)
   local test = torch.Tensor(numTest, 3, 200, 200)
   local trainLabels = torch.Tensor(numTrain)
   local testLabels = torch.Tensor(numTest)

   for i=1, numTrain do
      train[i] = allData[shuffle[i]]:clone()
     if allLabel[shuffle[i]]=='MALE' then
	trainLabels[i]=2
     else
	trainLabels[i]=1
     end
   end

   for i=numTrain+1,numTrain+numTest do
      test[i-numTrain] = allData[shuffle[i]]:clone()
      if allLabel[shuffle[i]]=='MALE' then
         testLabels[i-numTrain]=2
      else
         testLabels[i-numTrain]=1
      end   
end

   local trainData = {
      data = train,
      labels = trainLabels,
      size = function() return numTrain end
   }

   local testData = {
      data = test,
      labels = testLabels,
      size = function() return numTest end
   }
   return trainData,testData
end


function normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  -- preprocess valSet
  for i = 1,testData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end

  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)

end



trainData,testData = splitDataset()
normalize()

print(trainData.labels:mean())
print(testData.labels:mean())














