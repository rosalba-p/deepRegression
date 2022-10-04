from typing import Type
import utils, net_models, teachers 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.sgd import SGD

from torchvision import datasets, transforms
from torchvision.utils import make_grid 
import torch, torch.nn as nn
from theory_multiple_layers_all import compute_theory, compute_theory_synthetic, compute_theory_synthetic_sign
from theory_erf import compute_theory_synthetic_erf, compute_theory_erf
start_time = utils.time.time()
args = utils.parseArguments()
#args.resume_data = False
#print(args.resume_data)

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


class Langevin_SGD(Optimizer):

    def __init__(self, params, lr, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(Langevin_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if len(p.shape) == 1 and p.shape[0] == 1:
                    p.data.add_(-group['lr'], d_p)
                    
                else:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    unit_noise = Variable(p.data.new(p.size()).normal_())

                    p.data.add_(-group['lr'], 0.5*d_p + unit_noise/group['lr']**0.5)

        return loss


def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)
    
    return - (log_coeff + exponent).sum()


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)
    
    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()
    
    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
        return (exponent + log_coeff).sum()


class Langevin_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Langevin_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        
    def forward(self, x):
        
        return torch.mm(x, self.weights) + self.biases

class Langevin_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_log_noise):
        super(Langevin_Model, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # network with two hidden and one output layer
        self.layer1 = Langevin_Layer(input_dim, no_units)
        self.layer2 = Langevin_Layer(no_units, no_units)
        self.layer3 = Langevin_Layer(no_units, output_dim)
        
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace = True)
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

    
    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
        
        x = self.layer1(x)
        x = self.activation(x)
        
        x = self.layer3(x)
        
        return x

try:
	if args.device == 'cpu':
		raise TypeError
	torch.cuda.is_available() 
	device = 'cuda'	
except:
	device ='cpu'
print('\nWorking on', device)

class Langevin_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, init_log_noise, weight_decay):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches
        
        self.network = Langevin_Model(input_dim = input_dim, output_dim = output_dim,
                                      no_units = no_units, init_log_noise = init_log_noise)
        self.network.cuda()
        
        self.optimizer = Langevin_SGD(self.network.parameters(), lr=self.learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss
    
    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)
        
        # reset gradient and total loss
        self.optimizer.zero_grad()
        
        output = self.network(x)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)
        
        loss.backward()
        self.optimizer.step()

        return loss

home = utils.os.environ['HOME']
#P_list = [int(69*(args.step**i)) for i in range(9,args.nPoints)]
#P_list = [int(args.N1*i) for i in range(11,args.nPoints)]
P_list = [69, 110, 176, 282, 452, 723, 1157, 1852, 2963]
#P_list = [282, 452, 723, 1157, 1852, 2963]
#P_list = [723]
#P_list = [69, 110, 176]
mother_dir = './runs_erf/'
utils.make_directory(mother_dir)
first_subdir = mother_dir + f'teacher_{args.teacher_type}_net_{args.net_type}_opt_{args.opt}_bias_{args.bias}/'
utils.make_directory(first_subdir)

if args.net_type == 'resnet':
	net_class = net_models.make_resnet18()
	conv = True
elif args.net_type == 'densenet':
	net_class = net_models.make_densenet121()
	conv = True
elif args.net_type == 'vgg':
	net_class = net_models.make_vgg11()
	conv = True
elif args.net_type == '2hl':
	net_class = net_models.make_2hl(args.N, args.N1)
	conv = False
elif args.net_type == '1hl':
	net_class = net_models.make_1hl(args.N)
	conv = False
elif args.net_type == 'rfm':
	net_class = net_models.make_1hl(args.N)	
	conv = False

teacher_dir = f'{first_subdir}teachers/'
utils.make_directory(teacher_dir)
trainsets_dir = f'{first_subdir}trainsets/'
utils.make_directory(trainsets_dir)

if args.teacher_type == '1hl':
	teacherFilename = f'{teacher_dir}teacher_N_{args.N}_N1_{args.N1T}.pt'
	teacher_class = teachers.one_hl_dataset()
	teacher_class.resume = args.resume_data
	teacher_class.make_teacher_parameters(args.N, args.N1T, teacherFilename)
	
elif args.teacher_type == 'quadratic': 
	teacherFilename = f'{teacher_dir}/teacher_N_{args.N}.pt'
	teacher_class = teachers.quadratic_dataset()
	teacher_class.resume = args.resume_data
	teacher_class.make_teacher_parameters(args.N, teacherFilename)
elif args.teacher_type == 'linear': 
	teacherFilename = f'{teacher_dir}/teacher_N_{args.N}.pt'
	teacher_class = teachers.linear_dataset()

	teacher_class.resume = args.resume_data
	teacher_class.make_teacher_parameters(args.N, teacherFilename)
elif args.teacher_type == 'mnist': 
	teacher_class = teachers.mnist_dataset(args.N)
elif args.teacher_type == 'random': 
	teacher_class = teachers.random_dataset(args.N)



start_time = utils.time.time()

#teacher_class.N = args.N

teacher_class.save_data = args.save_data
train_type, test_type = teacher_class.training_type()
#R0 = teacher_class.trivial_predictor(args.Pnorm)
#print("\ntrivial predictor is", R0)
R0 = 1.


print(f'\npreambole took --- {utils.time.time() - start_time} seconds ---')
start_time = utils.time.time()
#CREATION OF FOLDERS
net_class.last_layer_size = args.N1
attributes_string = f'lr_{args.lr}_w_decay_{args.wd}_noise_{args.noise}_bs_{args.bs}_lambda0_{args.lambda0}_lambda1_{args.lambda1}'
run_folder = first_subdir + attributes_string + net_class.attributes_string()
if args.teacher_type == "1hl":
	run_folder = f'{run_folder}_N1T_{args.N1T}' 
run_folder = f'{run_folder}/'	
utils.make_directory(run_folder)
	
for P in P_list:

	print('\nNumber of examples in the train set:', P)
	trainsetFilename = f'{trainsets_dir}trainset_N_{args.N}_P_{P}_Ptest_{args.Ptest}.pt'
	solutionFilename = f'{run_folder}solution_P_{P}_replica_{args.R}.pt'

#BATCH SIZE

	if args.bs == 0: 
		batch_size_train = P
		batch_size_test  = min(args.Ptest, P)
	else: 
		batch_size_train = args.bs
		batch_size_test = args.bs
	teacher_class.batch_size = batch_size_train
	
	lambda1 = args.lambda1
	lambda0 = args.lambda0
#CRREATION OF DADTASET AND 
#THEORY CALCULATION
	if train_type == 'train_synthetic':
		inputs, targets, test_inputs, test_targets, resumed = teacher_class.make_data(P, args.Ptest, conv, trainsetFilename, device)

		print("\ntest size",len(test_targets))
		if args.noise and not resumed > 0: 
			print('\nadding noise...')
			targets = targets.to(device)
			targets += (args.noise*torch.randn(targets.size())).to(device) 
			test_targets += args.noise*torch.randn(test_targets.size())
		if args.compute_theory:
			start_time = utils.time.time()
			gen_error_pred, Qbar = compute_theory_synthetic_erf(inputs, targets, test_inputs, test_targets, args.N1, args.lambda1, first_subdir,P, args.Ptest, lambda0)
			print(f'\ntheory computation took - {utils.time.time() - start_time} seconds -')
			start_time = utils.time.time()
		else: 
			gen_error_pred, Qbar = 1.,torch.tensor(1.).cpu()
	elif train_type == 'train': 
		trainloader, testloader, testset_small, data, labels = teacher_class.make_data(P, args.Ptest, conv)
		if args.compute_theory:
			start_time = utils.time.time()
			gen_error_pred, Qbar = compute_theory_erf(data, labels, testset_small, args.N1, args.lambda1, first_subdir, P,args.Ptest, lambda0)
			print(f'\ntheory computation took - {utils.time.time() - start_time} seconds -')
			start_time = utils.time.time()
		else: 
			gen_error_pred, qbar = 1.,1.

	if args.only_theo: 
		continue

	for r in range(args.minR, args.R):
#FROM HERE YOU ARE WORKING WITH THE SAME TRAINSET
		runFilename = f'{run_folder}run_P_{P}_replica_{r}.txt'
		if not utils.os.path.exists(runFilename):
			sourceFile = open(runFilename, 'a')
			print('#1. epoch', '2. train error', '3. test error', '4. theory prediction', '5. Qbar', file = sourceFile)
			sourceFile.close()
		sourceFile = open(runFilename, 'a')
#NET INITIALISATION
		
		net = net_class.sequential(args.bias)
		start_epoch = int(0)
		if args.resume_net: 
			net, start_epoch = utils.load_net_state(solutionFilename, net, device)
			utils.cuda_init(net,device)
		else: 
			utils.cuda_init(net, device)	
			utils.init_network(net, args.bias)
			lastlayer = len(net)-1

			len_lastlayer = len(net[lastlayer].weight[0])

			utils.normalise(net, lastlayer, lambda1)
			utils.normalise(net, lastlayer-2,lambda0)
			#print(lastlayer, len_lastlayer,sigma2)		
		if args.net_type == 'rfm':
			(net[0].weight.requires_grad) = False
			(net[0].bias.requires_grad) = False
#TR	INING DYNAMICS SPECIFICATION
		lr = args.lr
		if args.opt == 'adam': 
			optimizer = utils.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd)
		elif args.opt == 'sgd':
			optimizer = utils.optim.SGD(net.parameters(), lr=lr, weight_decay=args.wd)
		elif args.opt == 'langevin':
			optimizer = Langevin_SGD(net.parameters(), lr=lr, weight_decay=args.wd)
		criterion = nn.MSELoss()
		if train_type == "train_synthetic":
			train_function_args = [net,inputs,targets,criterion,optimizer,device,batch_size_train]
			#print("\ninputs",inputs[0])
			test_function_args = [net,test_inputs,test_targets,criterion,optimizer,device,batch_size_test]
			#print("\ntest input",test_inputs[0])
		elif train_type == "train":
			train_function_args = [net,trainloader,criterion,optimizer, device,conv]
			test_function_args = [net,testloader,criterion,optimizer, device,conv]
#ZE	OTH EPOCH
		train_loss = utils.wrapper(eval(f'utils.{test_type}'), train_function_args)/R0
		#### TRY THIS ONE 
		#train_loss = eval(f'{test_type}(*{train_function_args}))
		test_loss = utils.wrapper(eval(f'utils.{test_type}'), test_function_args)/R0
		if args.compute_theory:
			print(f'{start_epoch}', train_loss, test_loss, gen_error_pred, Qbar.item(), file = sourceFile)
		else: 
			print(f'{start_epoch}', train_loss, test_loss, file = sourceFile)
		print(f'\nEpoch: 0 \nTrain Loss: {train_loss} \nTest Loss: {test_loss}')	
		for epoch in range(start_epoch+1, args.epochs):
			train_loss = utils.wrapper(eval(f'utils.{train_type}'), train_function_args)/R0
			
			test_loss = utils.wrapper(eval(f'utils.{test_type}'), test_function_args)/R0
			print(epoch, train_loss, test_loss, file = sourceFile)
			if epoch % args.checkpoint == 0 or epoch == args.epochs-1 :
				sourceFile.close()
				sourceFile = open(runFilename, 'a')
				print(f'\nEpoch: {epoch} \nTrain Loss: {train_loss} \nTest Loss: {test_loss} \n training took --- {utils.time.time() - start_time} seconds ---')				
				start_time = utils.time.time()	
				
				#utils.save_state(net, epoch, solutionFilename)
		sourceFile.close()