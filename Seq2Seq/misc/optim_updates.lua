
-- optim, simple as it should be, written from scratch. That's how I roll

function sgd(x, dx, lr, lrs)
  if lrs == nil then
    x:add(-lr, dx)
  else
	x:addcmul(-lr, dx, lrs)
  end
end

function sgdm(x, dx, lr, alpha, state, lrs)
  -- sgd with momentum, standard update
  if not state.v then
    state.v = x.new(#x):zero()
  end
  state.v:mul(alpha)
  if lrs == nil then
    state.v:add(lr, dx)
  else
	state.v:addcmul(lr, dx, lrs)
  end
  x:add(-1, state.v)
end

function sgdmom(x, dx, lr, alpha, state, lrs)
  -- sgd momentum, uses nesterov update (reference: http://cs231n.github.io/neural-networks-3/#sgd)
  if not state.m then
    state.m = x.new(#x):zero()
    state.tmp = x.new(#x)
  end
  state.tmp:copy(state.m)
  if lrs == nil then
    state.m:mul(alpha):add(-lr, dx)
  else
	state.m:mul(alpha):addcmul(-lr, dx, lrs)
  end
  x:add(-alpha, state.tmp)
  x:add(1+alpha, state.m)
end

function adagrad(x, dx, lr, epsilon, state, lrs)
  if not state.m then
    state.m = x.new(#x):zero()
    state.tmp = x.new(#x)
  end
  -- calculate new mean squared values
  state.m:addcmul(1.0, dx, dx)
  -- perform update
  state.tmp:sqrt(state.m):add(epsilon)
  local updateVal = torch.cdiv(dx, state.tmp)
  if lrs == nil then
    x:add(-lr, updateVal)
  else
    x:addcmul(-lr,updateVal, lrs)
  end
end

-- rmsprop implementation, simple as it should be
function rmsprop(x, dx, lr, alpha, epsilon, state, lrs)
  if not state.m then
    state.m = x.new(#x):zero()
    state.tmp = x.new(#x)
  end
  -- calculate new (leaky) mean squared values
  state.m:mul(alpha)
  state.m:addcmul(1.0-alpha, dx, dx)
  -- perform update
  state.tmp:sqrt(state.m):add(epsilon)
  local updateVal = torch.cdiv(dx, state.tmp)
  if lrs == nil then
    x:add(-lr, updateVal)
  else
	x:addcmul(-lr, updateVal, lrs)
  end
end

function adam(x, dx, lr, beta1, beta2, epsilon, state, lrs)
  local beta1 = beta1 or 0.9
  local beta2 = beta2 or 0.999
  local epsilon = epsilon or 1e-8

  if not state.m then
    -- Initialization
    state.t = 0
    -- Exponential moving average of gradient values
    state.m = x.new(#dx):zero()
    -- Exponential moving average of squared gradient values
    state.v = x.new(#dx):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.tmp = x.new(#dx):zero()
  end

  -- Decay the first and second moment running average coefficient
  state.m:mul(beta1):add(1-beta1, dx)
  state.v:mul(beta2):addcmul(1-beta2, dx, dx)
  state.tmp:copy(state.v):sqrt():add(epsilon)

  state.t = state.t + 1
  local biasCorrection1 = 1 - beta1^state.t
  local biasCorrection2 = 1 - beta2^state.t
  local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
  
  -- perform update
  local updateVal = torch.cdiv(state.m, state.tmp)
  if lrs == nil then
    x:add(-stepSize, updateVal)
  else
	x:addcmul(-stepSize, updateVal, lrs)
  end
end
