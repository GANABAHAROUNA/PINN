import torch
import torch.nn as nn
import numpy as np
import scipy.io
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")



# ============================================================================
# CLASS 1: NEURAL NETWORK
# ============================================================================
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        
        # Better initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)

    

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x





# ============================================================================
# STEP 2: LOAD AND PREPARE THE DATA
# ============================================================================
print("="*60)
print("LOADING DATA")
print("="*60)

try:
    data = scipy.io.loadmat('burgers_shock.mat')
    x = data['x'].flatten()[:, None]
    t = data['t'].flatten()[:, None]
    usol = np.real(data['usol']).T
    print("✅ Data loaded successfully from 'burgers_shock.mat'")
except FileNotFoundError:
    print("❌ File not found. Generating synthetic data...")
    nx, nt = 256, 100
    x = np.linspace(-1, 1, nx)[:, None]
    t = np.linspace(0, 1, nt)[:, None]
    X, T = np.meshgrid(x.flatten(), t.flatten())
    usol = -np.sin(np.pi * X) * np.exp(-0.5 * T)
    usol = usol.T.reshape(nt, nx)
    print("✅ Synthetic data generated")

# Create meshgrid
X, T = np.meshgrid(x, t)

# Combine x and t into a single tensor
train = torch.concat([
    torch.Tensor(X.flatten()[:, None]),
    torch.Tensor(T.flatten()[:, None])
], 1)

# Get min and max values
X_min = train.min(0)
X_max = train.max(0)

def getData():
    return train, usol, X_min, X_max

X_star, u_star, lb, ub = getData()

print(f"X_star shape: {X_star.shape}")
print(f"u_star shape: {u_star.shape}")





# ============================================================================
# STEP 3: DEFINE THE PINN CLASS WITH BOUNDARY CONDITIONS
# ============================================================================
class PINN():
    def __init__(self, X, u, lb, ub, physics, lambda_physics=0.001, 
                 X_boundary=None, u_boundary=None, lambda_boundary=1.0, lambda_data = 1.0):
        """
        Args:
            X: Training coordinates (interior points)
            u: Training data (solution values)
            lb: Lower bounds
            ub: Upper bounds
            physics: Boolean - include physics loss or not
            lambda_physics: Weight for physics loss
            X_boundary: Boundary coordinates (x=-1 and x=1)
            u_boundary: Boundary values (should be 0)
            lambda_boundary: Weight for boundary loss
        """
        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()
        self.physics = physics
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary
        self.lambda_data = lambda_data
        
        # Interior points (training data)
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float()
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float()
        self.u = torch.tensor(u).float()
        
        # Boundary points (NEW: Enforce u(t,-1) = u(t,1) = 0)
        if X_boundary is not None and u_boundary is not None:
            self.x_b = torch.tensor(X_boundary[:, 0:1], requires_grad=True).float()
            self.t_b = torch.tensor(X_boundary[:, 1:2], requires_grad=True).float()
            self.u_b = torch.tensor(u_boundary).float()
        else:
            self.x_b = None
            self.t_b = None
            self.u_b = None
        
        # Create the neural network
        self.network = Network()
        
        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100
        )
        
    def makeNetwork(self, x, t):
        X = torch.cat([x, t], 1)
        return self.network(X)
    
    def residual(self, x, t):
        """
        PDE residual: u_t + u*u_x - (0.01/pi)*u_xx = 0
        """
        u = self.makeNetwork(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u), 
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u), 
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x), 
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Burgers' equation residual
        return u_t + u * u_x - (0.01 / np.pi) * u_xx
    
    def loss(self):
        """
        Total loss = Data loss + Physics loss + Boundary loss
        """
        # 1. DATA LOSS: Match training data (includes initial condition)
        u_pred = self.makeNetwork(self.x, self.t)
        loss_data = torch.mean((self.u - u_pred)**2)
        
        # 2. PHYSICS LOSS: Satisfy PDE
        loss_physics = torch.tensor(0.0)
        if self.physics:
            residual_pred = self.residual(self.x, self.t)
            loss_physics = torch.mean(residual_pred**2)
        
        # 3. BOUNDARY LOSS: Enforce u(t,-1) = u(t,1) = 0
        loss_boundary = torch.tensor(0.0)
        if self.u_b is not None:
            u_b_pred = self.makeNetwork(self.x_b, self.t_b)
            loss_boundary = torch.mean((self.u_b - u_b_pred)**2)
        
        # Total weighted loss
        loss = self.lambda_data*loss_data + self.lambda_physics * loss_physics + self.lambda_boundary * loss_boundary
        
        return loss, loss_data, loss_physics, loss_boundary
    
    def train(self, epochs):
        lossTracker = []
        data_losses = []
        physics_losses = []
        boundary_losses = []
        self.network.train()
        
        for idx in range(epochs):
            # Compute loss
            loss, loss_data, loss_physics, loss_boundary = self.loss()
            
            lossTracker.append(loss.item())
            data_losses.append(loss_data.item())
            physics_losses.append(loss_physics.item() if self.physics else 0)
            boundary_losses.append(loss_boundary.item() if self.u_b is not None else 0)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step(loss)
            
            if (idx + 1) % 200 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {idx+1:4d}: Loss={loss.item():.6f}, Data={loss_data.item():.6f}, "
                      f"Physics={loss_physics.item():.6f}, Boundary={loss_boundary.item():.6f}, "
                      f"LR={current_lr:.6f}")
            
        return lossTracker, data_losses, physics_losses, boundary_losses
        
    def predict(self, X_test=None):
        """
        Make predictions on test points
        """
        self.network.eval()
        
        if X_test is not None:
            x_test = torch.tensor(X_test[:, 0:1], requires_grad=True).float()
            t_test = torch.tensor(X_test[:, 1:2], requires_grad=True).float()
            u_pred = self.makeNetwork(x_test, t_test)
            return u_pred.detach().numpy()
        else:
            u_pred = self.makeNetwork(self.x, self.t)
            return u_pred.detach().numpy()




# ============================================================================
# STEP 4: PREPARE TRAINING DATA WITH BOUNDARY CONDITIONS
# ============================================================================
print("\n" + "="*60)
print("PREPARING TRAINING DATA WITH BOUNDARY CONDITIONS")
print("="*60)

# 1. Interior points (training data)
num_train = 1000
idx = np.random.choice(X_star.shape[0], num_train, replace=False)
X_train = X_star[idx, :]
u_train = u_star.flatten()[:, None][idx, :]

print(f"Interior points: {X_train.shape}")

# 2. Boundary points: u(t, -1) = u(t, 1) = 0
print("\nCreating boundary points...")
t_boundary = np.linspace(0, 1, 500)
x_boundary_left = -np.ones_like(t_boundary)
x_boundary_right = np.ones_like(t_boundary)

# Combine boundary points
X_boundary = np.vstack([
    np.column_stack([x_boundary_left, t_boundary]),
    np.column_stack([x_boundary_right, t_boundary])
])
u_boundary = np.zeros((X_boundary.shape[0], 1))  # u = 0 at boundaries

print(f"Boundary points: {X_boundary.shape}")
print(f"Boundary values: {u_boundary.shape}")

# Convert to numpy
if isinstance(X_train, torch.Tensor):
    X_train_np = X_train.numpy()
else:
    X_train_np = X_train

if isinstance(u_train, torch.Tensor):
    u_train_np = u_train.numpy()
else:
    u_train_np = u_train

print(f"X_train_np shape: {X_train_np.shape}")
print(f"u_train_np shape: {u_train_np.shape}")





# ============================================================================
# STEP 5: TRAIN PINN WITH BOUNDARY CONDITIONS
# ============================================================================
print("\n" + "="*60)
print("TRAINING PINN WITH ALL COMPONENTS")
print("="*60)
print("  ✅ PDE: u_t + u*u_x - (0.01/π)*u_xx = 0")
print("  ✅ Initial Condition: u(0,x) = -sin(πx)")
print("  ✅ Boundary Conditions: u(t,-1) = u(t,1) = 0")
print("="*60)
model_pinn = PINN(
    X_train_np, u_train_np, lb[0], ub[0], 
    physics=True, 
    lambda_physics=1.0,
    X_boundary=X_boundary,
    u_boundary=u_boundary,
    lambda_boundary=1.0,
    lambda_data=1.0
)
pinn_loss, pinn_data, pinn_phys, pinn_boundary = model_pinn.train(300000)
torch.save(model_pinn.network.state_dict(), 'pinn_model.pth')
print("✅ Model saved as pinn_model.pth")





# ============================================================================
# STEP 6: TRAIN VANILLA NN (NO PHYSICS, NO BOUNDARY)
# ============================================================================
print("\n" + "="*60)
print("TRAINING VANILLA NN (Data Only)")
print("="*60)

model_nn = PINN(
    X_train_np, u_train_np, lb[0], ub[0], 
    physics=False, 
    lambda_physics=0,
    X_boundary=None,
    u_boundary=None,
    lambda_boundary=0,
    lambda_data=1.0
)
nn_loss, nn_data, nn_phys, nn_boundary = model_nn.train(300000)
torch.save(model_nn.network.state_dict(), 'nn_model.pth')
print("✅ Model saved as nn_model.pth")





# ============================================================================
# STEP 7: VISUALIZE LOSS CURVES
# ============================================================================
print("\n" + "="*60)
print("VISUALIZING LOSS CURVES")
print("="*60)

epochs = list(range(len(pinn_loss)))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=epochs, 
    y=pinn_loss, 
    mode='lines', 
    name='PINN (PDE + BC + IC)',
    line=dict(color='blue', width=2)
))
fig.add_trace(go.Scatter(
    x=epochs, 
    y=nn_loss, 
    mode='lines', 
    name='Vanilla NN (Data Only)',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title='Loss vs. Epochs',
    xaxis=dict(title='Epochs'),
    yaxis=dict(title='Loss', type='log'),
    width=800,
    height=500
)
fig.show()





# ============================================================================
# STEP 8: VISUALIZE PREDICTIONS
# ============================================================================
print("\n" + "="*60)
print("VISUALIZING PREDICTIONS")
print("="*60)

# Get predictions
if isinstance(X_star, torch.Tensor):
    X_test_full = X_star.numpy()
else:
    X_test_full = X_star

u_pinn = model_pinn.predict(X_test_full)
u_nn = model_nn.predict(X_test_full)

# Reshape
u_pinn_grid = u_pinn.reshape(usol.shape)
u_nn_grid = u_nn.reshape(usol.shape)



# True Solution
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=usol,
    x=x.flatten(),
    y=t.flatten(),
    colorscale='Viridis'
))
fig.update_layout(
    title='True Solution',
    xaxis=dict(title='x'),
    yaxis=dict(title='t'),
    width=600,
    height=500
)
fig.show()



# PINN Prediction
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=u_pinn_grid,
    x=x.flatten(),
    y=t.flatten(),
    colorscale='Viridis'
))
fig.update_layout(
    title='PINN Prediction (PDE + BC + IC)',
    xaxis=dict(title='x'),
    yaxis=dict(title='t'),
    width=600,
    height=500
)
fig.show()



# Vanilla NN Prediction
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=u_nn_grid,
    x=x.flatten(),
    y=t.flatten(),
    colorscale='Viridis'
))
fig.update_layout(
    title='Vanilla NN Prediction',
    xaxis=dict(title='x'),
    yaxis=dict(title='t'),
    width=600,
    height=500
)
fig.show()



# Error Comparison
pinn_mse = np.mean((usol - u_pinn_grid)**2)
nn_mse = np.mean((usol - u_nn_grid)**2)

fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=np.abs(usol - u_pinn_grid),
    x=x.flatten(),
    y=t.flatten(),
    colorscale='Reds',
    showscale=True
))
fig.update_layout(
    title=f'PINN Error (MSE: {pinn_mse:.6f})',
    width=600,
    height=500
)
fig.show()



fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=np.abs(usol - u_nn_grid),
    x=x.flatten(),
    y=t.flatten(),
    colorscale='Reds',
    showscale=True
))
fig.update_layout(
    title=f'Vanilla NN Error (MSE: {nn_mse:.6f})',
    width=600,
    height=500
)
fig.show()




# ============================================================================
# STEP 9: QUANTITATIVE COMPARISON
# ============================================================================
print("\n" + "="*60)
print("QUANTITATIVE COMPARISON")
print("="*60)

print(f"PINN - MSE: {pinn_mse:.6f}")
print(f"Vanilla NN - MSE: {nn_mse:.6f}")

if pinn_mse < nn_mse:
    improvement = ((nn_mse - pinn_mse) / nn_mse) * 100
    print(f"\n✅ PINN is {improvement:.2f}% better!")
    print("   (Physics + Boundary conditions help!)")
else:
    improvement = ((pinn_mse - nn_mse) / nn_mse) * 100
    print(f"\n⚠️ Vanilla NN is {improvement:.2f}% better")

print("\n" + "="*60)
print("✅ COMPLETE! All components included:")
print("   - PDE: u_t + u*u_x - (0.01/π)*u_xx = 0")
print("   - Initial: u(0,x) = -sin(πx)")
print("   - Boundary: u(t,-1) = u(t,1) = 0")
print("="*60)

