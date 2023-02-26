from torch.optim.sgd import SGD
from torch.optim import Adam
import torch
from torch import functional as F
import math

class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads, if_update = True):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']
        params = []
        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            # parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            if if_update:
                self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))
            else:
                # print("what is lr", lr)
                params.append(parameter - lr*grad_n)
        if not if_update:
            return params

    def meta_step_adam(self, grads, lr):
        lr = lr
        params = []
        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            # parameter.detach_()
            params.append(parameter - lr*grad)
        return params


class MetaAdam(Adam):
    """
    implements ADAM Algorithm, as a preceding step.
    """
    def __init__(self, net, *args, **kwargs):
        super(MetaAdam, self).__init__(*args, **kwargs)
        self.net = net


    def meta_step(self, grads, if_update = True):
        """
        Performs a single optimization step.
        """
        params = []
        for (n, p), grad in zip(self.net.named_parameters(), grads):
            state = self.state[p]
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Momentum (Exponential MA of gradients)
                state['exp_avg'] = torch.zeros_like(p.data)
                #print(p.data.size())
                # RMS Prop componenet. (Exponential MA of squared gradients). Denominator.
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            b1, b2 = self.param_groups[0]['betas']
            state['step'] += 1
            # L2 penalty. Gotta add to Gradient as well.
            if self.param_groups[0]['weight_decay'] != 0:
                grad = grad.add(self.param_groups[0]['weight_decay'], p.data)
            # Momentum
            exp_avg = torch.mul(exp_avg, b1) + (1 - b1)*grad
            # RMS
            exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1-b2)*(grad*grad)

            denom = exp_avg_sq.sqrt() + self.param_groups[0]['eps']

            bias_correction1 = 1 / (1 - b1 ** state['step'])
            bias_correction2 = 1 / (1 - b2 ** state['step'])

            adapted_learning_rate = self.param_groups[0]['lr'] * bias_correction1 / math.sqrt(bias_correction2)
            if if_update:
                p.data = p.data - adapted_learning_rate * exp_avg / denom
            else:
                params.append(p - adapted_learning_rate * exp_avg / denom)
        if not if_update:
            return params