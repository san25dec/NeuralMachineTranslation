require 'hdf5'
local utils = require 'misc.utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

	-- open the json file to read auxilliary data
	print('Loading DataLoader json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.ix_to_word_lang1 = self.info.ix_to_word_lang1
	self.ix_to_word_lang2 = self.info.ix_to_word_lang2
	self.split_count = {}
	self.split_count['train'] = self.info.num_train
	self.split_count['val'] = self.info.num_val
	self.split_count['test'] = self.info.num_test
	self.vocab_size_lang1 = utils.count_keys(self.ix_to_word_lang1)
	self.vocab_size_lang2 = utils.count_keys(self.ix_to_word_lang2)
	print('Language 1 vocab size: ' .. self.vocab_size_lang1)
	print('Language 2 vocab size: ' .. self.vocab_size_lang2)

	-- open the hdf5 file to read the text data
	print('Loading DataLoader h5 file: ', opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')

	-- initialize the iterators for the text data
	local seq_size_lang1 = self.h5_file:read('lang1/train'):dataspaceSize()
	local seq_size_lang2 = self.h5_file:read('lang2/train'):dataspaceSize()
	self.seq_length_lang1 = seq_size_lang1[2]
	self.seq_length_lang2 = seq_size_lang2[2]
	self.iterators = {}
	self.iterators['train'] = 1
	self.iterators['val'] = 1
	self.iterators['test'] = 1

	print('max sequence length in data for language 1 is ' .. self.seq_length_lang1)
	print('max sequence length in data for language 2 is ' .. self.seq_length_lang2)

end

function DataLoader:resetIterator(split)
	self.iterators[split] = 1
end

function DataLoader:getVocabSizeLang1()
	return self.vocab_size_lang1
end

function DataLoader:getVocabSizeLang2()
	return self.vocab_size_lang2
end

function DataLoader:getVocabLang1()
	return self.ix_to_word_lang1
end

function DataLoader:getVocabLang2()
	return self.ix_to_word_lang2
end

function DataLoader:getSeqLengthLang1()
	return self.seq_length_lang1
end

function DataLoader:getSeqLengthLang2()
	return self.seq_length_lang2
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - y (L,M) containing the sentences as columns 
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
	local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
	local batchSize = utils.getopt(opt, 'batchSize', 5) -- how many sentences  get returned at one time
	local wordEmbeddingSize = utils.getopt(opt, 'wordEmbeddingSize')
	-- pick an index of the datapoint to load next
	local lang1Vector = torch.LongTensor(batchSize, self.seq_length_lang1)
	local lang2Vector = torch.LongTensor(batchSize, self.seq_length_lang2)
	local max_index = self.split_count[split]
	local wrapped = false

	if self.iterators[split] + batchSize - 1 > max_index then
	    wrapped = true
	    if self.iterators[split] < max_index then
	      	local num_left = max_index-self.iterators[split]+1 
		    lang1Vector[{{1, num_left}, {}}] = self.h5_file:read('lang1/'..split):partial({self.iterators[split], max_index}, {1, self.seq_length_lang1})
		  	lang1Vector[{{max_index-self.iterators[split]+2, batchSize}, {}}] = self.h5_file:read('lang1/'..split):partial({1, batchSize-num_left}, {1, self.seq_length_lang1})
		  	lang2Vector[{{1, num_left}, {}}] = self.h5_file:read('lang2/'..split):partial({self.iterators[split], max_index}, {1, self.seq_length_lang2})
		  	lang2Vector[{{max_index-self.iterators[split]+2, batchSize}, {}}] = self.h5_file:read('lang2/'..split):partial({1, batchSize-num_left}, {1, self.seq_length_lang2})

		  	self.iterators[split] = batchSize-num_left+1
	  	else
			lang1Vector[{{1, batchSize}, {}}] = self.h5_file:read('lang1/'..split):partial({1, batchSize}, {1, self.seq_length_lang1})
			lang2Vector[{{1, batchSize}, {}}] = self.h5_file:read('lang2/'..split):partial({1, batchSize}, {1, self.seq_length_lang2})
			self.iterators[split] = batchSize+1 
	  	end
	else
	  	lang1Vector[{{1, batchSize}, {}}] = self.h5_file:read('lang1/'..split):partial({self.iterators[split], self.iterators[split]+batchSize-1}, {1, self.seq_length_lang1})
	  	lang2Vector[{{1, batchSize}, {}}] = self.h5_file:read('lang2/'..split):partial({self.iterators[split], self.iterators[split]+batchSize-1}, {1, self.seq_length_lang2})

	  	self.iterators[split] = self.iterators[split]+batchSize
	end

	local data = {}
	data.lang1Vector = lang1Vector:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.lang2Vector = lang2Vector:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.iterators[split], it_max = max_index, wrapped = wrapped}
	return data
end

