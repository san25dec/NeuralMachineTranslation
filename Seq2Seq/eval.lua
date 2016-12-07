require 'torch'
require 'nn'
require 'nngraph'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.SimpleMT'
require 'misc.optim_updates'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Machine Translation model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','','path to model to evaluate')
-- Basic options
cmd:option('-batchSize', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-test_sentences_use', 100, 'how many sentences to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-dump_json', 1, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_id', '', 'Dump vis id')
-- Sampling options
cmd:option('-sample_max', 0, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

cmd:option('-input_h5','','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json','','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-split', 'test', 'which split to use: val|test|train')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batchSize == 0 then opt.batchSize = checkpoint.opt.batchSize end
local fetch = {'rnnLayerSize', 'wordEmbeddingSize', 'drop_prob_ae', 'numLayers'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocabLang1 = checkpoint.vocabLang1 -- ix -> word mapping
local vocabLang2 = checkpoint.vocabLang2

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.crit = nn.LanguageModelCriterion()
protos.mt:createClones() -- reconstruct clones inside the language model
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

-------------------------------------------------------------------------------
-- Test set evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local test_sentences_use = utils.getopt(evalopt, 'test_sentences_use', true)
  
  protos.mt:evaluate()
  loader:resetIterator(split) -- rewind iterator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocabLang1 = loader:getVocabLang1()
  local vocabLang2 = loader:getVocabLang2()
  local count_sents = 0
  
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batchSize = opt.batchSize, split = split, wordEmbeddingSize = opt.wordEmbeddingSize}
	
	if opt.gpuid >= 0 then
      data.lang1Vector = data.lang1Vector:cuda()
	  data.lang2Vector = data.lang2Vector:cuda()
	end

    local logprobs = protos.mt:forward{data.lang1Vector, data.lang2Vector}
    local loss = protos.crit:forward(logprobs, data.lang2Vector)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each sentence
    local seq = protos.mt:sample_beam_karpathy(data.lang1Vector)
    local sents = net_utils.decode_sequence(vocabLang2, seq)
	local sents_actual = net_utils.decode_sequence(vocabLang2, data.lang2Vector)
	local sents_input = net_utils.decode_sequence(vocabLang1, data.lang1Vector)

	for k=1,#sents do
		count_sents = count_sents + 1
      local entry = {seqNo = count_sents, prediction = sents[k], actual = sents_actual[k], input = sents_input[k]}
	  --print('Prediction: ' .. sents[k] .. ' ||| Actual: ' .. sents_actual[k])
      table.insert(predictions, entry)
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, test_sentences_use)
    if verbose then
      print(string.format('evaluating performance... %d/%d (%f)', ix0-1, ix1, loss))
    end
	n = n + opt.batchSize
    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= test_sentences_use then break end -- we've used enough images
  end

  return loss_sum/loss_evals, predictions
end

local loss, split_predictions, lang_stats = eval_split(opt.split, {test_sentences_use = opt.test_sentences_use})
print('loss: ', loss)
if lang_stats then
  print(lang_stats)
end

if opt.dump_json == 1 then
  -- dump the json
  utils.write_json('results/results_' .. opt.dump_id .. '.json', split_predictions)
end
