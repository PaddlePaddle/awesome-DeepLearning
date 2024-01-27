import numpy as np

# L1 Loss
def l1_loss(pred, label):
    loss = np.abs(pred - label)
    loss = np.mean(loss)
    return loss

# Mean Bias Error
def mean_bias_error(pred, label):
    loss = pred - label
    loss = np.mean(loss)
    return loss

# Huber Loss
def huber_loss(pred, true, delta=1):
    loss = np.where(np.abs(true-pred) <= delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    loss = np.mean(loss)
    return loss

# Log-Cosh
def logcosh(pred,true):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)

# Weighed Cross Entropy
def wce_loss(logits,label,weight):

    logits = logits * (1-1e-05)  # 缩放预测值，以防出现log(0)

    loss =  -1 * (np.log2(logits)*label * weight + \
           np.log2(1 - logits) * (1-label))
    loss = np.mean(loss)
    return loss

# Balanced Cross Entropy
def balanced_ce_loss(logits,label,beta):

    logits = logits * (1-1e-05)  # 缩放预测值，以防出现log(0)

    loss =  -1 * (np.log2(logits)*label * beta + \
           np.log2(1 - logits) * (1-label)) * (1 - beta)
    loss = np.mean(loss)
    return loss

# Focal Loss
def focal_loss(logits, label, alpha, gamma):
    '''
        alpha 越大越关注y=1的情况
        gamma 越大越关注不确定的情况
    '''
    logits = logits * (1-1e-05)  # 缩放预测值，以防出现log(0)

    p_1 = - alpha*np.power(1-logits,gamma)*np.log2(logits)*label
    p_0 = - (1-alpha)*np.power(logits,gamma)*np.log2(1-logits)*(1-label)
    loss = p_0 + p_1
    loss = np.mean(loss)
    return loss

# Hinge Loss
def hinge_loss(pred, label):
    zeros = np.zeros_like(pred)
    loss = np.maximum(zeros,1-(pred*label))
    loss = np.mean(loss)
    return loss

pred_val = np.array([[0.9], [0.1], [1] ],dtype="float32")
gt_val = np.array([[1], [-1], [1] ],dtype="float32")

loss_value=hinge_loss(pred_val, gt_val)
print("Hinge Loss:",loss_value)