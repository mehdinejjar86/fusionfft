def make_inference(I0, I1, n, model, scale=1.0):    
  res = []
  for i in range(n):
      res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
  return res