local utils = require 'misc.utils'
local net_utils = {}

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

return net_utils
