require 'nn'
local LSTM_decoder = require 'misc.LSTM_decoder'
local LSTM_encoder = require 'misc.LSTM_encoder'
local utils = require 'misc.utils'
local heap = require 'binary_heap'
local SimpleMT, Parent = torch.class('nn.SimpleMT', 'nn.Module')
local net_utils = require 'misc.net_utils'

function SimpleMT:__init(opt)
	
	Parent.__init(self)
	self.lang1VocabSize = utils.getopt(opt, 'lang1VocabSize')
	self.lang2VocabSize = utils.getopt(opt, 'lang2VocabSize')
	self.wordEmbeddingSize = utils.getopt(opt, 'wordEmbeddingSize', 512)
	self.rnnLayerSize = utils.getopt(opt, 'rnnLayerSize', 512)
	self.numLayers = utils.getopt(opt, 'numLayers', 2)
	self.dropout = utils.getopt(opt, 'dropout', 0.5)
	self.maxLength1 = utils.getopt(opt, 'maxLength1') -- maximum length of lang1 sentences
	self.maxLength2 = utils.getopt(opt, 'maxLength2') -- maximum length of lang2 sentences
	-- Word embedding layers for the 2 languages
	self.lang1Embedding = nn.LookupTable(self.lang1VocabSize+1, self.wordEmbeddingSize)
	self.lang2Embedding = nn.LookupTable(self.lang2VocabSize+1, self.wordEmbeddingSize)
	-- Lazy initialization of init states, helpful in conversion to CUDA tensor
	self:initializeStates(1)
	-- To encode the 1st langugage sentence
	self.Encoder = LSTM_encoder.lstm(self.wordEmbeddingSize, self.rnnLayerSize, self.numLayers, self.dropout)
	-- To decode to the 2nd language sentence
	self.Decoder = LSTM_decoder.lstm(self.wordEmbeddingSize, self.lang2VocabSize+1, self.rnnLayerSize, self.numLayers, self.dropout)

end

function SimpleMT:initializeStates(batchSize)
	assert(batchSize ~= nil, 'No batchSize provided!')
	if not self.EncoderInitStates then self.EncoderInitStates = {} end
	if not self.DecoderInitStates then self.DecoderInitStates = {} end

	for i = 1, 2*self.numLayers do
		if self.EncoderInitStates[i] then
			if self.EncoderInitStates[i]:size(1) ~= batchSize then
				self.EncoderInitStates[i]:resize(batchSize, self.rnnLayerSize):zero()
			end
		else
			self.EncoderInitStates[i] = torch.zeros(batchSize, self.rnnLayerSize)
		end
		if self.DecoderInitStates[i] then
			if self.DecoderInitStates[i]:size(1) ~= batchSize then
				self.DecoderInitStates[i]:resize(batchSize, self.rnnLayerSize):zero()
			end
		else
			self.DecoderInitStates[i] = torch.zeros(batchSize, self.rnnLayerSize)
		end
	end
	self.numStates = 2*self.numLayers
end

function SimpleMT:createClones()

	self.EncoderClones = {self.Encoder}
	self.DecoderClones = {self.Decoder}
	self.lang1EmbeddingClones = {self.lang1Embedding}
	self.lang2EmbeddingClones = {self.lang2Embedding}

	for t = 2, self.maxLength1 do
		print('Encoder clone: t = ' .. t)
		self.EncoderClones[t] = self.Encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lang1EmbeddingClones[t] = self.lang1EmbeddingClones[1]:clone('weight', 'gradWeight')
	end

	for t = 2, self.maxLength2+1 do
		print('Decoder clone: t = ' .. t)
		self.DecoderClones[t] = self.Decoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lang2EmbeddingClones[t] = self.lang2EmbeddingClones[1]:clone('weight', 'gradWeight')
	end

end

function SimpleMT:parameters()

	local p1,g1 = self.Encoder:parameters()
	local p2,g2 = self.Decoder:parameters()
	local p3,g3 = self.lang1Embedding:parameters()
	local p4,g4 = self.lang2Embedding:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
	for k,v in pairs(p3) do table.insert(params, v) end
	for k,v in pairs(p4) do table.insert(params, v) end

	local gradParams = {}
	for k,v in pairs(g1) do table.insert(gradParams, v) end
	for k,v in pairs(g2) do table.insert(gradParams, v) end
	for k,v in pairs(g3) do table.insert(gradParams, v) end
	for k,v in pairs(g4) do table.insert(gradParams, v) end

	return params, gradParams

end

function SimpleMT:training()
	
	if self.EncoderClones == nil or self.DecoderClones == nil then self:createClones() end
	for k,v in pairs(self.EncoderClones) do v:training() end
	for k,v in pairs(self.DecoderClones) do v:training() end
	for k,v in pairs(self.lang1EmbeddingClones) do v:training() end
	for k,v in pairs(self.lang2EmbeddingClones) do v:training() end

end

function SimpleMT:getModulesList()
	return {self.Encoder, self.Decoder, self.lang1Embedding, self.lang2Embedding}
end

function SimpleMT:evaluate()
	
	if self.EncoderClones == nil or self.DecoderClones == nil then self:createClones() end
	for k,v in pairs(self.EncoderClones) do v:evaluate() end
	for k,v in pairs(self.DecoderClones) do v:evaluate() end
	for k,v in pairs(self.lang1EmbeddingClones) do v:evaluate() end
	for k,v in pairs(self.lang2EmbeddingClones) do v:evaluate() end

end

-- Input is a table consisting of 1 element
-- [1] lang1Vector : maxLength1 x batchSize
function SimpleMT:sample(input)

	lang1Vector = input
	local batchSize = lang1Vector:size(2)
	self:initializeStates(batchSize)

	local lang2Pred = torch.Tensor(self.maxLength2+1, batchSize):zero()
	local lang2LogProbs = torch.Tensor(self.maxLength2+1, batchSize):zero()

	-- Encoder forward propagation
	local EncoderState = self.EncoderInitStates
	for t = 1, self.maxLength1 do
		local it = lang1Vector[t]
		
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then
			it[it:eq(0)] = self.lang1VocabSize+1
			xt = self.lang1EmbeddingClones[t]:forward(it)
			EncoderState = self.EncoderClones[t]:forward({xt, unpack(EncoderState)})
		end
	end

	-- Initialize DecoderState as the final EncoderState
	local DecoderState = EncoderState
	-- Decoder forward propagation
	for t = 1, self.maxLength2+1 do
		
		local it
		if t == 1 then
			it = torch.Tensor(batchSize):fill(self.lang2VocabSize+1)
		else
			it = lang2Pred[t-1]
		end
	
		xt = self.lang2EmbeddingClones[t]:forward(it)
		local DecoderOut = self.DecoderClones[t]:forward({xt, unpack(DecoderState)})
		DecoderState = {}
		for i = 1, self.numStates do
			table.insert(DecoderState, DecoderOut[i])
		end
		local logsoft = DecoderOut[self.numStates+1]
		-- argmax sampling
		local logprob, pred = torch.max(logsoft, 2)
		lang2LogProbs[t] = logprob:float()
		lang2Pred[t] = pred:float()
	end
		
	return lang2Pred--, lang2LogProbs

end

-- BeamSearch Implementation with the help of lua-heaps
-- obtained from: https://github.com/geoffleyland/lua-heaps
-- Input is a table consisting of 1 element
-- [1] lang1Vector : maxLength1 x batchSize
function SimpleMT:sample_beam_notworking(input, opt)

	local L = self.maxLength2
	local k = utils.getopt(opt, 'beam_size', 5)
	local batchSize = lang1Vector:size(2)
	
	lang1Vector = input
	self:initializeStates(batchSize)

	local isCudaFlag = 0
	if lang1Vector:type() == 'torch.CudaTensor' then
		isCudaFlag = 1
	end
	local lang2Pred = torch.Tensor(self.maxLength2+1, batchSize):zero()
	--local lang2LogProbs = torch.Tensor(self.maxLength2+1, batchSize):zero()

	-- Encoder forward propagation
	local EncoderState = self.EncoderInitStates
	for t = 1, self.maxLength1 do
		local it = lang1Vector[t]
		
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then
			it[it:eq(0)] = self.lang1VocabSize+1
			xt = self.lang1EmbeddingClones[t]:forward(it)
			EncoderState = self.EncoderClones[t]:forward({xt, unpack(EncoderState)})
		end
	end

	-- Initialize DecoderState as the final EncoderState
	local DecoderStateInit = EncoderState

	-- comparison function to create a max heap
	local function comparison(k1, k2)
		return k1 > k2
	end

	-- not batch-wise prediction
	for b = 1, batchSize do

		local Heaps = {}
		local hinit = {}
		local FinalHeap = heap:new(comparison)
		for i = 1, self.numStates do
			hinit[i] = DecoderStateInit[i][{{b}, {}}]
		end
		for t = 0, L+1 do
			Heaps[t] = heap:new(comparison)
		end
		-- insert logprob, predicted words, hidden state @ that point
		Heaps[0]:insert(0, {pred={}, hidden=hinit})
		for t = 1, L+1 do
			for i = 1, math.min(k, Heaps[t-1].length) do
				local lp, popped = Heaps[t-1]:pop()
				local it
				if t == 1 then
					it =  torch.Tensor(1):fill(self.lang2VocabSize+1)
					if isCudaFlag == 1 then
						it = it:cuda()
					end
				else
					it = popped['pred'][t-1]
				end
				
				-- If the prediction for this beam stopped early
				-- or it stopped just in the last time instance
				-- then don't predict further, just push to heap
				if not it or (t > 1 and it[1] == self.lang2VocabSize+1) then
					FinalHeap:insert(lp, popped)
				else
					xt = self.lang2EmbeddingClones[t]:forward(it)
					local DecoderOut = self.DecoderClones[t]:forward({xt, unpack(popped['hidden'])})
					local DecoderState = {}
					for j = 1, self.numStates do
						table.insert(DecoderState, DecoderOut[j])
					end
					local logsoft = DecoderOut[self.numStates+1]
					-- topk sampling
					local logprob, pred = logsoft:topk(k, 2, true)
					for j = 1, k do
						local all_preds = {unpack(popped['pred'])}
						table.insert(all_preds, pred[1][{{j}}]:clone())
						Heaps[t]:insert(lp+logprob[1][j], {pred=all_preds, hidden=DecoderState})
						if t == L+1 then
							FinalHeap:insert(lp+logprob[1][j], {pred=all_preds, hidden=DecoderState})
						end
					end
				end
			end
		end

		local lp, best_pred = FinalHeap:pop()
		for i = 1, #best_pred['pred'] do
			lang2Pred[i][b] = best_pred['pred'][i]:float()
		end
	end
	
	return lang2Pred
end

-- Borrowing Beam Search from NeuralTalk2 base
function SimpleMT:sample_beam_karpathy(input, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 5)
  local batchSize = input:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.lang2VocabSize+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

	lang1Vector = input
	self:initializeStates(batchSize)

	local lang2Pred = torch.Tensor(self.maxLength2+1, batchSize):zero()
	--local lang2LogProbs = torch.Tensor(self.maxLength2+1, batchSize):zero()

	-- Encoder forward propagation
	local EncoderState = self.EncoderInitStates
	for t = 1, self.maxLength1 do
		local it = lang1Vector[t]
		
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then
			it[it:eq(0)] = self.lang1VocabSize+1
			xt = self.lang1EmbeddingClones[t]:forward(it)
			EncoderState = self.EncoderClones[t]:forward({xt, unpack(EncoderState)})
		end
	end

	-- Initialize DecoderState as the final EncoderState
	local DecoderStateInit = EncoderState

    local seq = torch.LongTensor(self.maxLength2, batchSize):zero()
    local seqLogprobs = torch.FloatTensor(self.maxLength2, batchSize)
    -- lets process every image independently for now, for simplicity
    for k=1,batchSize do

    	-- create initial states for all beams
		self:initializeStates(beam_size)
		local state = self.DecoderInitStates
		for i = 1, self.numStates do
			state[i]:copy(torch.repeatTensor(DecoderStateInit[i][{{k}, {}}], beam_size, 1))
		end

		-- we will write output predictions into tensor seq
		local beam_seq = torch.LongTensor(self.maxLength2, beam_size):zero()
		local beam_seq_logprobs = torch.FloatTensor(self.maxLength2, beam_size):zero()
		local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
		local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
		local done_beams = {}
		for t=1,self.maxLength2+1 do

		    local xt, it, sampleLogprobs
      		local new_state
      		if t == 1 then
        		-- feed in the start tokens
        		it = torch.LongTensor(beam_size):fill(self.lang2VocabSize+1)
        		xt = self.lang2Embedding:forward(it)
      		else
        	--[[
        	  	perform a beam merge. that is,
          		for every previous beam we now many new possibilities to branch out
          		we need to resort our beams to maintain the loop invariant of keeping
          		the top beam_size most likely sequences.
        	]]--
				local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
				ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
				local candidates = {}
				local cols = math.min(beam_size,ys:size(2))
				local rows = beam_size
				if t == 2 then rows = 1 end -- at first time step only the first beam is active
				for c=1,cols do -- for each column (word, essentially)
					for q=1,rows do -- for each beam expansion
						-- compute logprob of expanding beam q with word in (sorted) position c
						local local_logprob = ys[{ q,c }]
						local candidate_logprob = beam_logprobs_sum[q] + local_logprob
						table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
					end
				end
				table.sort(candidates, compare) -- find the best c,q pairs

				-- construct new beams
				new_state = net_utils.clone_list(state)
				local beam_seq_prev, beam_seq_logprobs_prev
				if t > 2 then
				    -- well need these as reference when we fork beams around
          			beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
          			beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        		end
				for vix=1,beam_size do
					local v = candidates[vix]
			  		-- fork beam index q into index vix
			  		if t > 2 then
						beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
						beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
			  		end
			  		-- rearrange recurrent states
			  		for state_ix = 1,#new_state do
						-- copy over state in previous beam q to new beam at vix
						new_state[state_ix][vix] = state[state_ix][v.q]
			  		end
			  		-- append new end terminal at the end of this beam
			  		beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
			  		beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
			  		beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

			  		if v.c == self.lang2VocabSize+1 or t == self.maxLength2+1 then
						-- END token special case here, or we reached the end.
						-- add the beam to a set of done beams
						table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
										  logps = beam_seq_logprobs[{ {}, vix }]:clone(),
										  p = beam_logprobs_sum[vix]
										 })
			 		end
				end
			
				-- encode as vectors
				it = beam_seq[t-1]
				xt = self.lang2Embedding:forward(it)
		  	end

      		if new_state then state = new_state end -- swap rnn state, if we reassinged beams

			local inputs = {xt,unpack(state)}
			local out = self.Decoder:forward(inputs)
			logprobs = out[self.numStates+1] -- last element is the output vector
			state = {}
			for i=1,self.numStates do table.insert(state, out[i]) end
		end

		table.sort(done_beams, compare)
		seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
		seqLogprobs[{ {}, k }] = done_beams[1].logps
	end

  	-- return the samples and their log likelihoods
  	return seq, seqLogprobs
end


-- Input is a table consisting of 2 elements
-- [1] lang1Vector : maxLength1 x batchSize
-- [2] leng2Vector : maxLength2 x batchSize
function SimpleMT:updateOutput(input)

	lang1Vector = input[1]
	lang2Vector = input[2]
	self.EncoderInputs = {}
	self.DecoderInputs = {}
	self.EncoderStates = {}
	self.DecoderStates = {}
	self.lang1EmbeddingInputs = {}
	self.lang2EmbeddingInputs = {}
	self.tmax1 = 0 
	self.tmax2 = 0

	local batchSize = lang1Vector:size(2)
	-- Initialize the states if not initialized already
	self:initializeStates(batchSize)
	self.EncoderStates[0] = self.EncoderInitStates

	-- Encoder forward propagation
	for t = 1, self.maxLength1 do
		local it = lang1Vector[t]
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then -- for optimization
			it[it:eq(0)] = self.lang1VocabSize+1  -- To ensure that the LookupTable
			-- doesn't throw errors. 
			self.lang1EmbeddingInputs[t] = it
			xt = self.lang1EmbeddingClones[t]:forward(it)
			self.EncoderInputs[t] = {xt, unpack(self.EncoderStates[t-1])}
			self.EncoderStates[t] = self.EncoderClones[t]:forward(self.EncoderInputs[t])
			self.tmax1 = t
		end
	end
	
	if not self.outputDec then
		self.outputDec = torch.Tensor(self.maxLength2+1, batchSize, self.lang2VocabSize+1)
		if lang2Vector:type() == 'torch.CudaTensor' then
			self.outputDec = self.outputDec:cuda()
		end
	else
		self.outputDec:resize(self.maxLength2+1, batchSize, self.lang2VocabSize+1)
	end

	-- Decoder forward propagation
	-- Initialize decoder states with the encoder final states
	self.DecoderStates[0] = self.EncoderStates[self.tmax1]
	for t = 1, self.maxLength2+1 do -- +1 because first token is the start token
		local it
		if t == 1 then
			-- start and end tokens are the vocab+1 index
			it = torch.Tensor(batchSize):fill(self.lang2VocabSize+1)
		else
			it = lang2Vector[t-1]
		end
		
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then
			it[it:eq(0)] = self.lang2VocabSize+1

			self.lang2EmbeddingInputs[t] = it
			xt = self.lang2EmbeddingClones[t]:forward(it)
			self.DecoderInputs[t] = {xt, unpack(self.DecoderStates[t-1])}
			local tempOut = self.DecoderClones[t]:forward(self.DecoderInputs[t])
			self.outputDec[t] = tempOut[self.numStates+1]
			self.DecoderStates[t] = {}
			for i = 1, self.numStates do
				self.DecoderStates[t][i] = tempOut[i]
			end
			self.tmax2 = t
		end
	end

	return self.outputDec

end

-- input: 2 tensors
-- [1]: maxLength1 x batchSize
-- [2]: maxLength2 x batchSize
-- gradOutput: maxLength2 x batchSize
function SimpleMT:updateGradInput(input, gradOutput)
	
	self.dDecoderStates = {[self.tmax2]=self.DecoderInitStates} -- just initialize as zeros 
	self.dEncoderStates = {}
	-- Decoder backward propagation
	for t = self.tmax2, 1, -1 do
		
		local dDecoder = {}
		for i = 1, self.numStates do
			table.insert(dDecoder, self.dDecoderStates[t][i])
		end
		table.insert(dDecoder, gradOutput[t])
		
		local dInputs = self.DecoderClones[t]:backward(self.DecoderInputs[t], dDecoder)
		self.dDecoderStates[t-1] = {}
		-- gradients wrt prev_c, prev_h
		for i = 2, self.numStates+1 do
			table.insert(self.dDecoderStates[t-1], dInputs[i])
		end
		-- gradients wrt wordEmbedding
		local it = self.lang2EmbeddingInputs[t]
		dw = self.lang2EmbeddingClones[t]:backward(it, dInputs[1])
	end

	-- Initialize encoder gradients from the first decoder gradient
	self.dEncoderStates[self.tmax1] = self.dDecoderStates[0]
	-- Encoder backward propagation
	for t = self.tmax1, 1, -1 do
		local dEncoder = {}
		for i = 1, self.numStates do
			table.insert(dEncoder, self.dEncoderStates[t][i])
		end

		local dInputs = self.EncoderClones[t]:backward(self.EncoderInputs[t], dEncoder)
		
		self.dEncoderStates[t-1] = {}
		-- gradients wrt prev_C, prev_h
		for i = 2, self.numStates+1 do
			table.insert(self.dEncoderStates[t-1], dInputs[i])
		end
		-- gradients wrt wordEmbedding
		local it = self.lang1EmbeddingInputs[t]
		dw = self.lang1EmbeddingClones[t]:backward(it, dInputs[1])
	end

	self.gradInput = {torch.Tensor()}
	return self.gradInput

end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+1)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target
in this criterion is as follows:
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 1 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do -- iterate over sequence time, t = 1 is where the start sequence begins 
      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}] -- t is correct, since at t=1 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end

