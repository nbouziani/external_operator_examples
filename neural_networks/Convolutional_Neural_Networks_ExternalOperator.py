from firedrake import *
import matplotlib.pyplot as plt
#%matplotlib inline

l = 2
nn = 256
mesh = SquareMesh(nn, nn, l, quadrilateral=True)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "DG", 0)

u = TrialFunction(V)
v = TestFunction(V)
f = Function(P)

a = 0.3
b = 0.7
c1 = -2
c2 = 2

s1 = (x<=b)*(x>=a)*(y>=a)*(y<=b)
s2 = (x<=b)*(x>=a)*(y>=1+a)*(y<=1+b)
s3 = (x<=1+b)*(x>=1+a)*(y>=a)*(y<=b)
s4 = (x<=1+b)*(x>=1+a)*(y>=1+a)*(y<=1+b)
square = Function(P).interpolate(s1 + s2 + s3 + s4)

f.interpolate(square*c1 + (1-square)*c2);
fig = tripcolor(f)
plt.colorbar(fig)
plt.title('f')

a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f,v) * dx

bc = DirichletBC(V, Constant(0.0), "on_boundary")

u = Function(V)
solve(a == L, u, bcs=bc,solver_parameters={'ksp_type': 'preonly', 'pc_dtype': 'lu'})

fig = tripcolor(u)
plt.title('u')
plt.colorbar(fig)

# Import pytorch
import torch
import torch.nn as nn

# Build the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=3,
                                      padding=1),
                            nn.ReLU(True),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(32, 64, kernel_size=3,
                                      padding=1),
                            nn.ReLU(True))

        self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(64, 32, kernel_size=3,
                                               stride=1, padding=1),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(32, 16, kernel_size=3,
                                               stride=2, padding=1),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(16, 1, kernel_size=4,
                                               padding=1),
                            nn.Tanh())

    def forward(self, x):
        e = self.encoder(x)
        out = self.decoder(e)
        return out

net = Net()

from torchsummary import summary
summary(net, input_size=(1, 256, 256))

# Import and load the pre-trained model
net = net.double()
PATH = './f_conv_ae.pth'
net.load_state_dict(torch.load(PATH))
net.eval()

# Convert a Function f to a matrix containing its values
# and vice versa
from utils.preprocess_fct import convert_func_, revert_func_

# Subclass the PytorchOperator to define a new operator that
# will have its own way to define the input
class Conv_Auto_Encoder(PytorchOperator):
    def __init__(self, *args, **kwargs):
        PytorchOperator.__init__(self, *args, **kwargs)

    def _compute_derivatives(self):
        """Compute the gradient of the network wrt inputs"""
        op = self.interpolate(self.ufl_operands[0])
        torch_op = torch.from_numpy(op.dat.data).type(torch.FloatTensor)
    
        model_output = self.evaluate().dat.data
        res = []
        for i, e in enumerate(torch_op):
            xi = torch.unsqueeze(e, o)
            yi = model_output[i]
            res.append(torch.autograd.grad(yi, xi)[0])
        return res

    def _evaluate(self):
        """Evaluate the neural network, i.e. forward pass"""
        space = self.ufl_function_space()
        model = self.model.eval()
        op = self.interpolate(self.ufl_operands[0])

        # Turns the Function into a numpy matrix containing the
        # function evaluations
        f_np = convert_func_(op, space)
        f_torch = torch.from_numpy(f_np).reshape((1, 1, 256, 256))

        # Now let's apply the forward pass and scale the values
        # which were normalised for the training
        val = model(f_torch).detach().numpy()
        val = val*(f_np.max() - f_np.min()) + f_np.min()

        # Turn the function evaluations into a Function
        val = revert_func_(val.reshape((256, 256)), space)
        return self.assign(val)

# We have defined a new type of PytorchOperator!
nP = partial(Conv_Auto_Encoder, function_space=P, operator_data={'model': net})


f_cae = nP(f)  # since we want to learn identity 

# Let's check how good was the learning
print('\n Error f: ', assemble( (f-f_cae)**2*dx ) / assemble( f**2*dx ))

# Let's now solve the problem with f_cae in the rhs
# We don't need to redefine the lhs but just the rhs
L = inner(f_cae, v)*dx

u2 = Function(V)

solve(a == L, u2, bcs=bc,solver_parameters={'ksp_type': 'preonly', 'pc_dtype': 'lu'})

fig = tripcolor(f_cae.get_coefficient())
plt.title('f obtained from the Convolutional Auto Encoder Neural Network')
plt.colorbar(fig)

fig = tripcolor(u2)
plt.title('u2')
plt.colorbar(fig)

# Let's check the error between u and u2

print('\n Error u: ', assemble( (u-u2)**2*dx )/assemble(u**2*dx))

plt.show()
