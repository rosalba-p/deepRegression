from typing import Type
from utils import * 
from net_models import * 
from teachers import * 


"""# **UTILS**"""


def parseArguments():

	parser = argparse.ArgumentParser()

	# Positional mandatory arguments
	parser.add_argument("teacher_type", help="choose a teacher type: linear, quadratic, 1hl, mnist", type=str)
	parser.add_argument("net_type", help="choose a net type: rfm, 1hl, 2hl, resnet18, vgg11, densenet 121 ", type=str)


	parser.add_argument("-lr", "--lr", help="learning rate", type=float, default=1e-04)
	parser.add_argument("-wd", "--wd", help="weight decay", type=float, default=1e-05)
	parser.add_argument("-resume", "--resume", help="resume previous training", type=bool, default=False)
	parser.add_argument("-device", "--device",  type=str, default="cpu")

	parser.add_argument("-epochs", "--epochs", help="number of train epochs", type = int, default = 10000)
	parser.add_argument("-bs", "--bs", help="batch size train", type=int, default=0)

	#you will trigger a series of experiment with increasing dataset size. at each step Pstart -> Pstart + step. the number of steps is: nPoints 
	parser.add_argument("-Pstart", "--Pstart", help="size of training set", type=int, default=200)
	parser.add_argument("-step", "--step", help="step to increase size", type=float, default=10)
	parser.add_argument("-nPoints", "--nPoints", help="number of iterations", type=int, default=50)

	#specify the networks you want to use 
	parser.add_argument("-N", "--N", help="size of input data", type=int, default=300)
	parser.add_argument("-N1", "--N1", help="size of first hidden layer", type=int, default=400)
	parser.add_argument("-N2", "--N2", help="size of second hidden layer", type=int, default=400)
	parser.add_argument("-N1T", "--N1T", help="size of teacher's hidden layer", type=int, default=200)
	parser.add_argument("-Ptest", "--Ptest", help="# examples in test set", type=int, default=10000)


	# you can specify this index if you want to do more than one run of experiments. by default it is set to 0. 
	parser.add_argument("-R", "--R", help="replica index", type=int, default=0)
	parser.add_argument("-checkpoint", "--checkpoint", help="# epochs checkpoint", type=int, default=1000)
	parser.add_argument("-noise", "--noise", help="signal to noise ratio", type=float, default=0.)

	args = parser.parse_args()
	return args



#@profile
def train(net, trainloader,criterion, optimizer,  device, teacher_type):
	net.train()
	train_loss = 0

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		
		if teacher_type == "mnist":				
			targets = targets.type(torch.FloatTensor).unsqueeze(1)
		else:
			targets = targets.unsqueeze(1)
		inputs, targets = inputs.to(device), targets.to(device)

		optimizer.zero_grad()
		outputs = net(inputs)

		loss = criterion(outputs, targets)

		loss.backward()
		optimizer.step()

		train_loss += loss.item()
	#print_stats()
	return(train_loss/(batch_idx+1))


#@profile
def test(net, testloader,criterion, device,teacher_type):
	net.eval()
	test_loss = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):

				if teacher_type == "mnist":				
					targets = targets.type(torch.FloatTensor).unsqueeze(1)
				else:
					targets = targets.unsqueeze(1)
				
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = net(inputs)
				loss = criterion(outputs,targets)


				test_loss += loss.item()
	#print_stats()
	return(test_loss/(batch_idx+1))

def train_synthetic(net,data, labels, criterion, optimizer, device, batch_size):
		net.train()
		train_loss = 0
		P = len(data)
		batch_num = max(int(P/batch_size),1)

		s = np.arange(P)
		np.random.shuffle(s)
		data = data[s]
		labels = labels[s]
		for i in range(batch_num):
			start = i*(batch_size)
			inputs, targets = data[start:start+batch_size].to(device), (labels[start:start+batch_size]).unsqueeze(1).to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			train_loss += loss.item() 
		return train_loss/batch_num

def test_synthetic(net, test_data, test_labels, criterion, optimizer, device, batch_size):
		net.eval()
		test_loss = 0
		P_test = len(test_data)
		batch_num = max(int(P_test/batch_size),1)

		for i in range(batch_num):
			start = i*(batch_size)
			with torch.no_grad():
					inputs, targets = test_data[start:start+batch_size].to(device), (test_labels[start:start+batch_size]).unsqueeze(1).to(device)
					outputs = net(inputs)
					loss = criterion(outputs, targets)
					test_loss += loss.item()
		return test_loss/batch_num



args = parseArguments()
"""# **MAIN**"""


try:
	if args.device == "cpu":
		raise TypeError
	torch.cuda.is_available() 
	device = "cuda"	
except:
	device ='cpu'
#device = 'cpu'
print("\nyou are working on", device)

save_data = True #TODO
attributes_string = f"lr_{args.lr}_w_decay_{str(args.wd)}_noise_{args.noise}"  

home = os.environ['HOME']
P_list = [int(args.Pstart*(args.step**i)) for i in range(args.nPoints)]
mother_dir = "./runs/"
if not os.path.isdir(mother_dir): 
	os.mkdir(mother_dir)


first_subdir = mother_dir + f"teacher_{args.teacher_type}_net_{args.net_type}/"
if not os.path.isdir(first_subdir): 
	os.mkdir(first_subdir)

RGB = True

if args.net_type == "resnet":
	net_class = make_resnet18()
elif args.net_type == "densenet":
	net_class = make_densenet121()
elif args.net_type == "vgg":
	net_class = make_vgg11()
elif args.net_type == "2hl":
	net_class = make_2hl(args.N, args.N1, args.N2)
	RGB = False
elif args.net_type == "1hl":
	net_class = make_1hl(args.N, args.N1)
	RGB = False
elif args.net_type == "rfm":
	net_class = make_1hl(args.N, args.N1)	
	RGB = False


dir_name = first_subdir + attributes_string + net_class.attributes_string() 	


if args.teacher_type == "1hl":
	teacher = torch.randn(args.N1T,args.N)
	teacher_vec_2 = torch.randn(args.N1T)
	teacher_class = one_hl_dataset(teacher, teacher_vec_2)
	dir_name = f"{dir_name}_N1T_{args.N1T}"
elif args.teacher_type == "quadratic": 
	teacher = torch.randn(args.N)
	teacher_class = quadratic_dataset(teacher)
elif args.teacher_type == "linear": 
	teacher = torch.randn(args.N)
	teacher_class = linear_dataset(teacher)
elif args.teacher_type == "mnist": 
	teacher_class = mnist_dataset()
elif args.teacher_type == "random": 
	teacher_class = random_dataset(args.N)

dir_name = f"{dir_name}/"	
if not os.path.isdir(dir_name): 
	os.mkdir(dir_name)

start_time = time.time()


teacher_class.RGB = RGB 
teacher_class.P_test = args.Ptest 

teacher_class.resume = args.resume

for P in P_list:
	print("\nnumber of examples in the train set:", P)
	trainsetFilename = f"{dir_name}trainset_P_{P}_replica_{args.R}.pt"
	solutionFilename = f"{dir_name}solution_P_{P}_replica_{args.R}.pt"

	runFilename = dir_name + f"run_P_{P}_replica_{args.R}.txt"

	if not os.path.exists(runFilename):
		sourceFile = open(runFilename, "a")
		print("#epoch", "train error", "test error", file = sourceFile)
		sourceFile.close()


	if args.bs == 0: 
		batch_size_train = P
		batch_size_test  = min(args.Ptest, P)

	
	net = net_class.sequential()
	start_epoch = int(0)

	if args.resume: 
		net, start_epoch = load_net_state(solutionFilename, net, device)
		cuda_init(net,device)
	else: 
		cuda_init(net, device)	
		init_network(net)
		normalise(net)

	if args.net_type == "rfm":
		(net[0].weight.requires_grad) = False
		(net[0].bias.requires_grad) = False

	optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
	criterion = nn.MSELoss()


	teacher_class.P = P	
	teacher_class.batch_size = args.bs
	train_type, test_type = teacher_class.training_type()


	if train_type == "train_synthetic":
		inputs, targets, test_inputs, test_targets, resumed = teacher_class.make_data(trainsetFilename, device)

		if not resumed: 
			targets = targets.to(device)
			targets += (args.noise*torch.randn(targets.size())).to(device) 
			test_targets += args.noise*torch.randn(test_targets.size())

		if save_data:
			print('Saving data...')
			state = {
			'inputs': inputs,
			'targets': targets,
			#'teacher' : teacher,
		}
			if args.teacher_type == "1hl":
				state = {
				'inputs': inputs,
				'targets': targets,
				'teacher' : teacher,
				'teacher2' : teacher_vec_2
			}
		torch.save(state, trainsetFilename)

		train_function_args = [net,inputs,targets,criterion,optimizer,device,batch_size_train]
		test_function_args = [net,test_inputs,test_targets,criterion,optimizer,device,batch_size_test]

	elif train_type == "train": 
		train_function_args = [net,trainloader,criterion,device,RGB]
		test_function_args = [net,testloader,criterion,device,RGB]
		trainloader, testloader, data, labels = teacher_class.make_data()


	print("starting training")
	sourceFile = open(runFilename, 'a')

	#zeroth epoch
	train_loss = wrapper(eval(test_type), train_function_args)

	#### TRY THIS ONE 
	#train_loss = eval(f"{test_type}(*{train_function_args}))

	test_loss = wrapper(eval(test_type), test_function_args)
	print(f"{start_epoch}", train_loss, test_loss, test_loss-train_loss, file = sourceFile)
	print(f'\nEpoch: 0 \nTrain Loss: {train_loss} \nTest Loss: {test_loss}')	

	for epoch in range(start_epoch+1, args.epochs):		

		train_loss = wrapper(eval(train_type), train_function_args)
		test_loss = wrapper(eval(test_type), test_function_args)
		print(epoch, train_loss, test_loss, file = sourceFile)

		if epoch % args.checkpoint == 0 or epoch == args.epochs-1 :
			sourceFile.close()
			sourceFile = open(runFilename, 'a')
			print(f'\nEpoch: {epoch} \nTrain Loss: {train_loss} \nTest Loss: {test_loss}')				
			print("training took --- %s seconds ---" % (time.time() - start_time))
			start_time = time.time()	
			save_state(net, epoch, solutionFilename)
			if train_loss < 1e-05:
				break



	#print_stats()
	sourceFile.close()
