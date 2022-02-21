"""Implements implicit differentiation."""
import torch

from .grad_helpers import my_jacobian, flatten_param, compute_grad
from .conjugate_gradient import conjugate_gradient
from .neumann_series import neumann_approximation


def vector_inverse_hessian_product(val_data, model, train_direct_grad, method='exact', cfg=None, optimizer=None):
    val_direct_grad = compute_grad(val_data, model, structure=False)
    if method == 'EXACT':
        hessian = my_jacobian(
            train_direct_grad, list(model.model_parameters()))
        eps = 1e-6*torch.eye(hessian.shape[0], device=hessian.device)
        hess_inv = torch.inverse(hessian+eps)
        vihp = val_direct_grad.T.mm(hess_inv)
        return vihp.T
    elif method == 'IDENTITY':
        return val_direct_grad
    elif method in ['CG', 'NEUMANN']:
        def A(xx): return my_jacobian(
            [train_direct_grad], list(model.model_parameters()), [xx])
        b = val_direct_grad
        if method == 'CG':
            if cfg is None:
                return conjugate_gradient(A, b, None, 1e-4, 30)
            return conjugate_gradient(A, b, None, cfg.SOLVER.CG.TOL, cfg.SOLVER.CG.MAXITER)
        elif method == 'NEUMANN':
            if cfg is not None:
                return neumann_approximation(A, b, cfg.SOLVER.NEUMANN.MAXITER, cfg.SOLVER.NEUMANN.ALPHA)
            else:
                if optimizer is not None:
                    alpha = optimizer.param_groups[0]['lr']
                else:
                    alpha = 0.1
                with torch.no_grad():
                    return neumann_approximation(A, b, 20, alpha)
    else:
        raise NotImplemented("Unsupported method %s!" % method)


def vector_jacobian_product(model, train_direct_grad, vihp):
    return my_jacobian([train_direct_grad], list(model.hyper_parameters()), [vihp])


def compute_hypergrad(train_data, val_data, model, optimizer=None, method='exact', cfg=None):
    # Get direct hypergradient on val.
    val_direct_hypergrad = compute_grad(
        val_data, model, structure=True, is_train=False)

    # Get indirect hypergradient.
    # Vector-inverse Hessian Product
    train_direct_grad = compute_grad(train_data, model, structure=False,
                                     create_graph=True, retain_graph=True)
    vihp = vector_inverse_hessian_product(
        val_data, model, train_direct_grad, method=method, cfg=cfg, optimizer=optimizer)
    # Vector-Jacobian product
    val_indirect_hypergrad = -1*vector_jacobian_product(
        model, train_direct_grad, vihp)

    # Return sum of direct and indirect gradient.
    return val_direct_hypergrad + val_indirect_hypergrad
