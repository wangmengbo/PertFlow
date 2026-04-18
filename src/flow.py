import os
import sys
import math
import torch
import logging
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import unwrap_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RectifiedFlow:
    """Implements rectified flow dynamics for generative modeling."""
    
    def __init__(self, sigma_min=0.002, sigma_max=80.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def noise_schedule(self, t):
        """Cosine noise schedule that determines noise level at time t."""
        return self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
    
    def drift_coefficient(self, t):
        """Calculate drift coefficient for rectified flow."""
        return 1.0
    
    def sample_path(self, x_1, t, noise=None):
        """Sample from the path at time t using linear rectified flow interpolation."""
        t_expanded = t.view(-1, *([1] * (len(x_1.shape) - 1)))
        
        if noise is None:
            noise = torch.randn_like(x_1)
        
        x_t = (1 - t_expanded) * noise + t_expanded * x_1
        velocity = x_1 - noise
        
        return {
            "x_t": x_t,
            "velocity": velocity,
            "noise": noise,
            "sigma_t": self.noise_schedule(t_expanded)
        }
    
    def loss_fn(self, model_output, target_velocity, target_images=None, noise=None, 
                labels=None, use_triplet=True, temperature=1.0):
        """Compute loss between predicted and target velocities."""
        if use_triplet:
            if target_images is None or noise is None:
                raise ValueError("target_images and noise required for triplet loss")
            return self.triplet_contrastive_loss_fn(model_output, target_images, noise, labels, temperature)
        return torch.mean((model_output - target_velocity) ** 2)

    def triplet_contrastive_loss_fn(self, model_output, target_images, noise, labels=None, temperature=0.05):
        """Compute triplet contrastive loss for rectified flow."""
        target_velocity = target_images - noise
        
        pred_flat = model_output.flatten(1)
        target_flat = target_velocity.flatten(1)
        
        pos_error = torch.mean((pred_flat - target_flat) ** 2, dim=1)
        
        batch_size = pred_flat.shape[0]
        negative_indices = self._sample_negatives(labels, batch_size, pred_flat.device)
        
        target_neg = target_flat[negative_indices]
        neg_error = torch.mean((pred_flat - target_neg) ** 2, dim=1)
        
        margin = 1.0
        contrastive_term = torch.clamp(margin - (neg_error - pos_error), min=0.0)
        triplet_loss = pos_error + temperature * contrastive_term
        
        return {
            "loss": triplet_loss.mean(),
            "flow_loss": pos_error.mean(), 
            "contrastive_loss": neg_error.mean(),
            "margin_violation": (contrastive_term > 0).float().mean()
        }

    def _sample_negatives(self, labels, batch_size, device):
        """Sample negative indices based on labels or randomly."""
        if labels is not None:
            return self._class_conditioned_sampling(labels)
        
        negative_indices = torch.randint(0, batch_size, (batch_size,), device=device)
        self_mask = negative_indices == torch.arange(batch_size, device=device)
        negative_indices[self_mask] = (negative_indices[self_mask] + 1) % batch_size
        return negative_indices

    def _class_conditioned_sampling(self, labels):
        """Sample negatives based on class labels (different class)."""
        batch_size = labels.shape[0]
        device = labels.device
        
        mask = ~(labels[None] == labels[:, None])
        weights = mask.float()
        weights_sum = weights.sum(dim=1, keepdim=True)
        
        if (weights_sum == 0).any():
            choices = torch.randint(0, batch_size, (batch_size,), device=device)
            self_mask = choices == torch.arange(batch_size, device=device)
            choices[self_mask] = (choices[self_mask] + 1) % batch_size
        else:
            weights = weights / weights_sum.clamp(min=1)
            choices = torch.multinomial(weights, 1).squeeze(1)
        
        return choices

class MultiModalDOPRI5Solver:
    """DOPRI5 solver for multimodal drug-conditioned models."""
    
    def __init__(self, model, rectified_flow, control_rna, control_images, conditioning_info, 
                 rtol=1e-3, atol=1e-4, safety=0.9):
        self.model = unwrap_model(model)
        self.rf = rectified_flow
        self.control_rna = control_rna
        self.control_images = control_images
        self.conditioning_info = conditioning_info
        self.rtol = rtol
        self.atol = atol
        self.safety = safety
        
        # Dormand-Prince Butcher tableau
        self.a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        ]
        
        self.b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        self.b_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    def _get_velocity(self, x, t, batch_size, device):
        """Get velocity from model at given state and time."""
        t_batch = torch.ones(batch_size, device=device) * t
        return self.model(x, t_batch, self.control_rna, self.control_images, 
                         self.conditioning_info, mode='image')
    
    def _dormand_prince_step(self, x, t, dt, batch_size, device):
        """Perform one DOPRI5 step."""
        with torch.no_grad():
            k = []
            
            # Stage 1
            k.append(self._get_velocity(x, t, batch_size, device))
            
            # Stages 2-6
            for i in range(1, 6):
                x_i = x.clone()
                for j in range(i):
                    x_i = x_i + dt * self.a[i][j] * k[j]
                
                t_i = t + self.c[i] * dt
                k.append(self._get_velocity(x_i, t_i, batch_size, device))
            
            # Stage 7
            x_final = x.clone()
            for j in range(5):
                x_final = x_final + dt * self.a[5][j] * k[j]
            
            k.append(self._get_velocity(x_final, t + dt, batch_size, device))
            
            # 5th order solution
            x_next = x.clone()
            for i in range(6):
                x_next = x_next + dt * self.b[i] * k[i]
            
            # 4th order solution for error estimation
            x_next_star = x.clone()
            for i in range(7):
                x_next_star = x_next_star + dt * self.b_star[i] * k[i]
            
            return x_next, x_next - x_next_star
    
    def _compute_adaptive_step_size(self, error, x, dt):
        """Compute adaptive step size based on error estimate."""
        error_ratio = torch.norm(error) / (self.atol + self.rtol * torch.norm(x))
        
        if error_ratio == 0:
            return dt * 2.0
        
        dt_new = self.safety * dt * (1.0 / error_ratio) ** (1/5)
        return torch.clamp(dt_new, dt * 0.1, dt * 5.0)
    
    def generate_sample(self, num_steps=100, initial_dt=0.01, device="cuda"):
        """Generate treatment images using multimodal model."""
        # Handle case where control_rna might be None (for image-only models)
        if self.control_rna is not None:
            batch_size = self.control_rna.shape[0]
        else:
            batch_size = self.control_images.shape[0]
            
        img_channels = self.model.img_channels
        img_size = self.model.img_size
        
        x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
        
        t = 0.0
        dt = initial_dt
        step_count = 0
        dt_history = [dt]
        
        while t < 1.0 and step_count < num_steps:
            if t + dt > 1.0:
                dt = 1.0 - t
            
            x_next, error = self._dormand_prince_step(x, t, dt, batch_size, device)
            dt_next = self._compute_adaptive_step_size(error, x, dt)
            
            x = x_next
            t += dt
            step_count += 1
            dt = dt_next
            dt_history.append(dt)
            
            if step_count % 10 == 0:
                logger.info(f"MultiModal generation progress: t={t:.4f}, dt={dt:.6f}, steps={step_count}")
        
        logger.info(f"MultiModal DOPRI5 completed in {step_count} steps, final t={t:.4f}")
        
        if len(dt_history) > 1:
            min_dt, max_dt, avg_dt = min(dt_history), max(dt_history), sum(dt_history)/len(dt_history)
            logger.info(f"Step size stats: min={min_dt:.6f}, max={max_dt:.6f}, avg={avg_dt:.6f}")
        
        return torch.clamp(x, -1, 1)