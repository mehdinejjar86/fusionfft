import math

def make_inference_recursive(I0, I1, n, model, scale=1.0):    
  if model.version >= 3.9:
    res = []
    flows = []
    masks = []
    original_n = n

    for i in range(n):

        if i == 0 or i == n - 1:
            res.append(model.inference_recursive(I0, I1, (i+1) * 1. / (n+1), scale))
        else:
            flow, mask = model.flow_extractor(I0, I1, (i+1) * 1. / (n+1), scale)
            flows.append([flow])
            masks.append([mask])
    n -= 2
    I0 = res[0]
    I1 = res[-1]

    while len(res) != original_n:
      for i in range(n):
    
        if i == 0 or i == n - 1:
            prev_flows = flows[0] if i == 0 else flows[i-1]
            prev_masks = masks[0] if i == 0 else masks[i-1]
            res_out = model.inference(I0, I1, (i+1) * 1. / (n+1), scale, prev_flows=prev_flows, prev_masks=prev_masks)
            middle_index = math.ceil(len(res) / 2) if len(res) != 0 else 0
            res.insert(middle_index, res_out)
        else:
            flow, mask = model.flow_extractor(I0, I1, (i+1) * 1. / (n+1), scale)
            flows[i].append(flow)
            masks[i].append(mask)
              
      I0 = res[middle_index-1]
      I1 = res[middle_index]

      n -= 2

      if len(res) == original_n:
          break

      if flows != []:
          flows.pop(0)
          masks.pop(0)
          flows.pop(-1)
          masks.pop(-1)

  else:
      raise NotImplementedError("Recursive inference is not implemented for versions below 3.9")
  return res