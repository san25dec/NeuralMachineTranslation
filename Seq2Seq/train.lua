require 'torch'
require 'nn'
require 'nngraph'
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
cmd:text('Train an AutoEncoder model for sentences')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','data/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/data.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnnLayerSize',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-wordEmbeddingSize',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-numLayers', 2, 'number of layers in RNN')
-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batchSize',16,'what is the batch size in number of sentences per batch?')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_ae', 0.5, 'strength of dropout in the AutoEncoder RNN')

-- Optimization: for the AutoEncoder Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay', 1e-6, 'L2 regularization parameter')
-- Evaluation/Checkpointing
cmd:option('-val_sentences_use', 30000, 'how many sentences to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
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
-- Create the Data Loader instance
-------------------------------------------------------------------------------
loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  local mt_modules = protos.mt:getModulesList()
  for k,v in pairs(mt_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
  loaded_checkpoint = nil

else
  -- create protos from scratch
  -- intialize machine translator
  local mtOpt = {}
  mtOpt.lang1VocabSize = loader:getVocabSizeLang1()
  mtOpt.lang2VocabSize = loader:getVocabSizeLang2()
  mtOpt.wordEmbeddingSize = opt.wordEmbeddingSize
  mtOpt.rnnLayerSize = opt.rnnLayerSize
  mtOpt.numLayers = opt.numLayers
  mtOpt.dropout = opt.drop_prob_ae
  mtOpt.maxLength1 = loader:getSeqLengthLang1()
  mtOpt.maxLength2 = loader:getSeqLengthLang2()
  protos.mt = nn.SimpleMT(mtOpt)
  protos.crit = nn.LanguageModelCriterion()
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do
	  v:cuda()
	  print('Converted to GPU')
  end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.mt:getParameters()
--params:uniform(-0.08, 0.08)
print('total number of parameters in MachineTranslator: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_mt = protos.mt:clone()
thin_mt.Encoder:share(protos.mt.Encoder, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_mt.Decoder:share(protos.mt.Decoder, 'weight', 'bias')
thin_mt.lang1Embedding:share(protos.mt.lang1Embedding, 'weight', 'bias')
thin_mt.lang2Embedding:share(protos.mt.lang2Embedding, 'weight', 'bias')

-- sanitize all modules of gradient storage so that we dont save big checkpoints

local mt_modules = thin_mt:getModulesList()
for k,v in pairs(mt_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.mt:createClones()

collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_sentences_use = utils.getopt(evalopt, 'val_sentences_use', true)

  protos.mt:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
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
    local seq = protos.mt:sample(data.lang1Vector)
    local lang2Pred = net_utils.decode_sequence(vocabLang2, seq)
	local lang2Actual = net_utils.decode_sequence(vocabLang2, data.lang2Vector)
	local lang1Actual = net_utils.decode_sequence(vocabLang1, data.lang1Vector)

    for k=1,#lang2Pred do
		count_sents = count_sents + 1
      local entry = {seqNo = count_sents, prediction = lang2Pred[k], actual = lang2Actual[k], input = lang1Actual[k]}
      table.insert(predictions, entry)
    end

    -- if we wrapped around the split or used up val sentences budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_sentences_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end
	
	n = n + opt.batchSize
    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_sentences_use then break end -- we've used enough sentences
  end

  return loss_sum/loss_evals, predictions
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  protos.mt:training()
  grad_params:zero()

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batchSize = opt.batchSize, split = 'train', wordEmbeddingSize = opt.wordEmbeddingSize}
  -- data.seq: LxM where L is sequence length upper bound, and M = # of sentences 

  -- forward the matchine translation model
  if opt.gpuid >= 0 then
	  data.lang1Vector = data.lang1Vector:cuda()
	  data.lang2Vector = data.lang2Vector:cuda()
  end

  local logprobs = protos.mt:forward{data.lang1Vector, data.lang2Vector}
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.lang2Vector)
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.lang2Vector)
  -- backprop machine translation model
  local ddummy = protos.mt:backward(data.lang1Vector, dlogprobs)

  -- clip gradients
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.weight_decay > 0 then
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    grad_params:add(opt.weight_decay, params)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
while true do  

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  
  local params_norm = torch.norm(params)
  local num_updates = torch.sum(torch.ge(torch.abs(torch.mul(grad_params, opt.learning_rate)), torch.mul(torch.abs(params), 0.01)))
  print(string.format('iter %d: loss: %.3f | # updates: %4d | paramsNorm: %.4f', iter, losses.total_loss, num_updates, params_norm))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_sentences_use = opt.val_sentences_use})
    print('validation loss: ', val_loss)
    val_loss_history[iter] = val_loss

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history
	utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score = -val_loss

    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.mt = thin_mt -- these are shared clones, and point to correct param storage
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocabLang1 = loader:getVocabLang1()
		checkpoint.vocabLang2 = loader:getVocabLang2()

        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote BEST checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
--]]
