from typing import Type
import utils, net_models, teachers  
import torch, torch.nn as nn
from theory_multiple_layers_all import compute_theory, compute_theory_synthetic, compute_theory_synthetic_sign
from theory_erf import compute_theory_synthetic_erf, compute_theory_erf
start_time = utils.time.time()
args = utils.parseArguments()
#args.resume_data = False
#print(args.resume_data)



try:
	if args.device == 'cpu':
		raise TypeError
	torch.cuda.is_available() 
	device = 'cuda'	
except:
	device ='cpu'
print('\nWorking on', device)


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