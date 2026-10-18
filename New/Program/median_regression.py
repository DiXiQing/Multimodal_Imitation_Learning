"""Weighted median (tau=0.5) cubic spline regression, solved by linear programming."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack


def fit_median_spline(z, y, weights, penalty, knot_strategy='weighted_quantile'):
    """Minimize weighted absolute residuals + L1 second-difference smoothness."""
    if knot_strategy=='uniform':
        interior=np.linspace(0,1,8)[1:-1]
    elif knot_strategy=='weighted_quantile':
        # Training-only weighted quantiles provide resolution near contact, where
        # most measurements and the sharp distance/aperture transition occur.
        order=np.argsort(z,kind='stable')
        sorted_z=np.asarray(z)[order]
        sorted_weights=np.asarray(weights)[order]
        cdf=(np.cumsum(sorted_weights)-.5*sorted_weights)/sorted_weights.sum()
        interior=np.unique(np.interp(np.linspace(0,1,14)[1:-1],cdf,sorted_z))
        interior=interior[(interior>1e-8)&(interior<1-1e-8)]
    else:
        raise ValueError(f'Unsupported knot strategy: {knot_strategy}')
    knots=np.r_[np.zeros(4),interior,np.ones(4)]
    design=csr_matrix(BSpline.design_matrix(z,knots,3))
    n,p=design.shape
    difference=csr_matrix(np.diff(np.eye(p),n=2,axis=0))
    q=difference.shape[0]
    residual_identity=eye(n,format='csr')
    smooth_identity=eye(q,format='csr')
    constraints=vstack([
        hstack([design,-residual_identity,csr_matrix((n,q))]),
        hstack([-design,-residual_identity,csr_matrix((n,q))]),
        hstack([difference,csr_matrix((q,n)),-smooth_identity]),
        hstack([-difference,csr_matrix((q,n)),-smooth_identity]),
    ],format='csr')
    objective=np.r_[np.zeros(p),weights/weights.sum(),np.full(q,float(penalty))]
    result=linprog(objective,A_ub=constraints,b_ub=np.r_[y,-y,np.zeros(2*q)],
                   bounds=[(None,None)]*p+[(0,None)]*(n+q),method='highs')
    if not result.success:
        raise RuntimeError(f'Median spline fitting failed: {result.message}')
    return dict(coefficients=result.x[:p].tolist(),knots=knots.tolist(),degree=3,
                quantile=.5,loss='weighted_absolute_error',knot_strategy=knot_strategy,
                smoothness='L1 second differences of spline coefficients',
                optimizer='scipy.optimize.linprog/highs')
