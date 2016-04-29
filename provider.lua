-- read an index file and 
-- image all resized to 200*200

require 'image'


local file = io.open("fh_expression_ix")
if file then
    ims = {}
    labels ={}
    for line in file:lines() do
        local im,label = unpack(line:split(","))
        local f=io.open(im,"r")
        if f~=nil then
            local img = image.load(im,3,'float')            
            table.insert(ims, image.scale(img, 200, 200)   )
            table.insert(labels, label)
            f:close()
            
        end
    end
  
end


data = torch.Tensor(#ims,3,200,200)
for id = 1, #ims do
    data[id]= ims[id]    
end

data:size()
torch.save('fh_expression_data.t7',data)
torch.save('fh_expression_label.t7',labels) 