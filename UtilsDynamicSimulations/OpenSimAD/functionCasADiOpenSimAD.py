'''
    ---------------------------------------------------------------------------
    OpenCap processing: functionCasADiOpenSimAD.py
    ---------------------------------------------------------------------------
    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    This script defines several CasADi functions for use when setting up
    the optimal control problem.
'''

import casadi as ca
import numpy as np

# %% CasADi function to approximate muscle-tendon lenghts, velocities,
# and moment arms based on polynomial approximations of joint positions and
# velocities.
def polynomialApproximation(musclesPolynomials, polynomialData, NPolynomial):    
    
    from polynomialsOpenSimAD import polynomials
    
    # Function variables.
    qin = ca.SX.sym('qin', 1, NPolynomial)
    qdotin  = ca.SX.sym('qdotin', 1, NPolynomial)
    
    lMT = ca.SX(len(musclesPolynomials), 1)
    vMT = ca.SX(len(musclesPolynomials), 1)
    dM = ca.SX(len(musclesPolynomials), NPolynomial)    
    for count, musclePolynomials in enumerate(musclesPolynomials):        
        coefficients = polynomialData[musclePolynomials]['coefficients']
        dimension = polynomialData[musclePolynomials]['dimension']
        order = polynomialData[musclePolynomials]['order']        
        spanning = polynomialData[musclePolynomials]['spanning']
        polynomial = polynomials(coefficients, dimension, order)
        idxSpanning = [i for i, e in enumerate(spanning) if e == 1]        
        lMT[count] = polynomial.calcValue(qin[0, idxSpanning])
        dM[count, :] = 0
        vMT[count] = 0        
        for i in range(len(idxSpanning)):
            dM[count, idxSpanning[i]] = - polynomial.calcDerivative(
                    qin[0, idxSpanning], i)
            vMT[count] += (-dM[count, idxSpanning[i]] * 
               qdotin[0, idxSpanning[i]])
    f_polynomial = ca.Function('f_polynomial',[qin, qdotin],[lMT, vMT, dM])
    
    return f_polynomial
        
# %% CasADi function to describe the Hill equilibrium based on the
# DeGrooteFregly2016MuscleModel muscle model.
def hillEquilibrium(mtParameters, tendonCompliance, tendonShift,
                    specificTension, ignorePassiveFiberForce=False):
    
    NMuscles = mtParameters.shape[1]
    
    # Function variables
    activation = ca.SX.sym('activation', NMuscles)
    mtLength = ca.SX.sym('mtLength', NMuscles)
    mtVelocity = ca.SX.sym('mtVelocity', NMuscles)
    normTendonForce = ca.SX.sym('normTendonForce', NMuscles)
    normTendonForceDT = ca.SX.sym('normTendonForceDT', NMuscles)
     
    hillEquilibrium = ca.SX(NMuscles, 1)
    tendonForce = ca.SX(NMuscles, 1)
    activeFiberForce = ca.SX(NMuscles, 1)
    normActiveFiberLengthForce = ca.SX(NMuscles, 1)
    passiveFiberForce = ca.SX(NMuscles, 1)
    normFiberLength = ca.SX(NMuscles, 1)
    fiberVelocity = ca.SX(NMuscles, 1)
    activeFiberForcePen = ca.SX(NMuscles, 1)
    passiveFiberForcePen = ca.SX(NMuscles, 1)
    
    from muscleModelOpenSimAD import DeGrooteFregly2016MuscleModel
    for m in range(NMuscles):    
        muscle = DeGrooteFregly2016MuscleModel(
            mtParameters[:, m], activation[m], mtLength[m],
            mtVelocity[m], normTendonForce[m], 
            normTendonForceDT[m], tendonCompliance[:, m],
            tendonShift[:, m], specificTension[:, m],
            ignorePassiveFiberForce=ignorePassiveFiberForce)
        hillEquilibrium[m] = muscle.deriveHillEquilibrium()
        tendonForce[m] = muscle.getTendonForce()
        activeFiberForce[m] = muscle.getActiveFiberForce()[0]
        passiveFiberForce[m] = muscle.getPassiveFiberForce()[0]
        normActiveFiberLengthForce[m] = muscle.getActiveFiberLengthForce()
        normFiberLength[m] = muscle.getFiberLength()[1]
        fiberVelocity[m] = muscle.getFiberVelocity()[0]
        activeFiberForcePen[m] = muscle.getActiveFiberForce()[2]
        passiveFiberForcePen[m] = muscle.getPassiveFiberForce()[2]
    f_hillEquilibrium = ca.Function(
        'f_hillEquilibrium', [activation, mtLength, mtVelocity, 
                              normTendonForce, normTendonForceDT], 
        [hillEquilibrium, tendonForce, activeFiberForce, passiveFiberForce,
         normActiveFiberLengthForce, normFiberLength, fiberVelocity,
         activeFiberForcePen, passiveFiberForcePen]) 
    
    return f_hillEquilibrium

# %% CasADi function to describe the dynamics of the coordinate actuators.
def coordinateActuatorDynamics(nJoints):
    
    # Function variables
    eArm = ca.SX.sym('eArm',nJoints)
    aArm = ca.SX.sym('aArm',nJoints)
    
    t = 0.035 # time constant  
    aArmDt = (eArm - aArm) / t    
    f_armActivationDynamics = ca.Function('f_armActivationDynamics',
                                          [eArm, aArm], [aArmDt])
    
    return f_armActivationDynamics

# %% CasADi function to compute passive limit joint torques.
def limitPassiveTorque(k, theta, d):
    
    # Function variables
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = (k[0] * np.exp(k[1] * (Q - theta[1])) + k[2] * 
                           np.exp(k[3] * (Q - theta[0])) - d * Qdot)    
    f_limitPassiveTorque = ca.Function('f_limitPassiveTorque', [Q, Qdot], 
                                       [passiveJointTorque])
    
    return f_limitPassiveTorque

# %% CasADi function to compute linear passive joint torques given stiffness
# and damping.
def linarPassiveTorque(k, d):
    
    # Function variables
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = -k * Q - d * Qdot
    f_linarPassiveTorque = ca.Function('f_linarPassiveTorque', [Q, Qdot], 
                                       [passiveJointTorque])
    
    return f_linarPassiveTorque

# %% CasADi function to compute the normalized sum of the weighted elements in
# a vector elevated to a given power.
def normSumWeightedPow(N, exp):
    
    # Function variables
    x = ca.SX.sym('x', N,  1)
    w = ca.SX.sym('w', N,  1)
      
    nsp = ca.sum1(w * (x**exp))       
    nsp = nsp / N    
    f_normSumPow = ca.Function('f_normSumWeightedPow', [x, w], [nsp])
    
    return f_normSumPow

# %% CasADi function to compute the normalized sum of the squared elements in a
# vector.
def normSumSqr(N):
    
    # Function variables
    x = ca.SX.sym('x', N, 1)
    
    ss = ca.sumsqr(x) / N        
    f_normSumSqr = ca.Function('f_normSumSqr', [x], [ss])
    
    return f_normSumSqr

# %% CasADi function to compute difference in torques (inverse dynamics vs
# muscle and passive contributions).
def diffTorques():
    
    # Function variables
    jointTorque = ca.SX.sym('x', 1) 
    muscleTorque = ca.SX.sym('x', 1) 
    passiveTorque = ca.SX.sym('x', 1)
    
    diffTorque = jointTorque - (muscleTorque + passiveTorque)    
    f_diffTorques = ca.Function(
            'f_diffTorques', [jointTorque, muscleTorque, passiveTorque], 
            [diffTorque])
        
    return f_diffTorques

# %% CasADi function to compute the normalized sum of the weighted squared
# difference between two vectors.
def normSumWeightedSqrDiff(dim):
    
    # Function variables
    x = ca.SX.sym('x', dim, 1) 
    x_ref = ca.SX.sym('x_ref', dim, 1)  
    w = ca.SX.sym('w', dim, 1) 
    
    nSD = ca.sum1(w * (x-x_ref)**2)
    nSD = nSD / dim        
    f_normSumSqrDiff = ca.Function('f_normSumSqrDiff', [x, x_ref, w], [nSD])
    
    return f_normSumSqrDiff



def derivativeSumOfNegatives(dim1, dim2, small_derivative_threshold=0.001):
    """
    Computes a weighted squared sum of:
    1. Negative derivatives (strong penalty)
    2. Derivatives with magnitude < threshold (small penalty)
    
    Args:
        dim1 (int): Number of rows (e.g., 52 for time steps).
        dim2 (int): Number of columns (e.g., 3 for COP data).
        small_derivative_threshold (float): Threshold below which derivatives are penalized.
    
    Returns:
        casadi.Function: A function that computes the weighted penalized sum.
    """
    # Function variables
    x = ca.SX.sym('x', dim1, dim2)  # Input matrix (dim1 x dim2)
    w = ca.SX.sym('w', dim2, 1)  # Weight vector (dim2 x 1)

    # Compute derivatives (finite differences)
    derivative = x[1:, :] - x[:-1, :]  # Shape: (dim1-1, dim2)

    # **1. Heavy Penalty for Negative Derivatives (Quadratic)**
    neg_deriv = ca.fmin(derivative, 0)  # Only negative derivatives
    neg_penalty = neg_deriv**4  # Quadratic penalty (strong)

    # **2. Mild Penalty for Small Derivatives (Linear)**
    small_deriv_threshold = 0.01
    small_deriv_mask = ca.fabs(derivative) < small_deriv_threshold
    small_penalty = (small_deriv_threshold - ca.fabs(derivative)) * small_deriv_mask  # Linear penalty

    # **Total Penalty (Weighted Sum)**
    total_penalty = (
        ca.sum2(ca.mtimes(neg_penalty, ca.diag(w.T))) +  # Heavy penalty on negatives
        ca.sum2(ca.mtimes(small_penalty, ca.diag(w.T)))  # Mild penalty on small derivatives
    )

    f = ca.Function('f_derivativePenaltyHardcoded', [x, w], [total_penalty])
    return f

def thirdDerivativeInflectionPenalty(dim1, dim2):
    """
    Penalizes inflection points by applying a weighted sum of squared
    third-order finite differences (i.e., curvature change).

    This approximates zero-crossings in the second derivative, which signal
    changes in concavity (inflection points) in a smooth, differentiable way.

    Args:
        dim1 (int): Number of rows (must be ≥ 4).
        dim2 (int): Number of columns (e.g., 3 for COP data).

    Returns:
        casadi.Function: CasADi function that outputs a scalar penalty.
    """
    import casadi as ca

    # Inputs
    x = ca.SX.sym('x', dim1, dim2)  # Time-series matrix: (dim1 x dim2)
    w = ca.SX.sym('w', dim2, 1)     # Weights per dimension (dim2 x 1)

    # Compute third derivative: f'''(i) = x[i] - 3x[i-1] + 3x[i-2] - x[i-3]
    third_deriv = x[3:, :] - 3 * x[2:-1, :] + 3 * x[1:-2, :] - x[0:-3, :]
    squared = third_deriv**2

    # Weighted sum of squared third derivatives
    total_penalty = ca.sum2(ca.mtimes(squared, ca.diag(w.T)))

    # Return as CasADi function
    f = ca.Function('f_thirdDerivativeInflectionPenalty', [x, w], [total_penalty])
    return f

def totalVariationGradientPenalty(dim1, dim2):
    """
    Computes a penalty based on Total Variation Regularization (TVR) and Gradient Matching.

    Args:
        dim1 (int): Number of rows (e.g., 52 for time steps).
        dim2 (int): Number of columns (e.g., 3 for data dimensions).

    Returns:
        casadi.Function: A function that computes the combined penalty.
    """
    # Define symbolic variables
    x = ca.SX.sym('x', dim1, dim2)  # Input matrix (dim1 x dim2)
    w = ca.SX.sym('w', dim2, 1)  # Weight vector (dim2 x 1)
    
    # Compute finite differences (gradients) along columns
    derivative = x[1:, :] - x[:-1, :]  # Shape: (dim1-1, dim2)

    # **Total Variation Regularization (TVR)**
    tv_penalty = ca.sum1(ca.fabs(derivative))  # L1 norm of gradient differences

    # **Gradient Matching Penalty (encourages smooth progression)**
    median_step = ca.median(derivative)  # Compute median step size
    gradient_penalty = ca.sumsqr(derivative - median_step)  # Penalize deviations

    # **Final weighted penalty**
    total_penalty = ca.mtimes((tv_penalty + gradient_penalty), w)

    # Create CasADi function
    f_totalVariationGradientPenalty = ca.Function('f_totalVariationGradientPenalty', [x, w], [total_penalty])
    return f_totalVariationGradientPenalty


def doubleDerivativeSquare(dim1, dim2):
    """
    Computes a penalty based on Total Variation Regularization (TVR) and Gradient Matching.

    Args:
        dim1 (int): Number of rows (e.g., 52 for time steps).
        dim2 (int): Number of columns (e.g., 3 for data dimensions).

    Returns:
        casadi.Function: A function that computes the combined penalty.
    """
    # Define symbolic variables
    x = ca.SX.sym('x', dim1, dim2)  # Input matrix (dim1 x dim2)
    w = ca.SX.sym('w', dim2, 1)  # Weight vector (dim2 x 1)
    
    # Compute finite differences (gradients) along columns
    derivative = x[1:, :] - x[:-1, :]  # Shape: (dim1-1, dim2)
    
    # Compute finite differences (gradients) along columns
    derivative2 = derivative[1:, :] - derivative[:-1, :]  # Shape: (dim1-1, dim2)
    derivative_penalty = (ca.sum1(ca.fmax(ca.fabs(derivative2), 0.01)))

    # derivative_penalty = (ca.sum1(ca.fmax(ca.fabs(derivative), 0.01))**2)



    # **Final weighted penalty**
    total_penalty = ca.mtimes(derivative_penalty, w)

    # Create CasADi function
    f_totalVariationGradientPenalty = ca.Function('f_totalVariationGradientPenalty', [x, w], [total_penalty])
    return f_totalVariationGradientPenalty





def getCOP_casadi(N):
    """
    CasADi version of the getCOP function.
    Computes the Center of Pressure (COP) from Ground Reaction Forces (GRF) and Moments (GRM).
    
    Args:
        N (int): The number of time steps (columns of GRF and GRM).
    
    Returns:
        casadi.Function: A function that takes GRF and GRM as inputs and returns COP for all time steps.
    """
    # Define symbolic inputs (3xN matrices for GRF and GRM)
    GRF = ca.SX.sym('GRF', 3, N)  # 3xN matrix
    GRM = ca.SX.sym('GRM', 3, N)  # 3xN matrix

    # Initialize COP as symbolic variable (3xN matrix)
    COP = ca.SX.zeros(3, N)  # Create a 3xN matrix for COP

    # Loop over each time step and compute COP
    for k in range(N):  # Loop over N time steps # edit this to be just a mask
        COP[0, k] = ca.if_else(GRF[1, k] > 0.001, GRM[2, k] / GRF[1, k], 0)  # COP_x
        COP[2, k] = ca.if_else(GRF[1, k] > 0.001, -GRM[0, k] / GRF[1, k], 0)  # COP_z

    # Create CasADi function to evaluate COP for all time steps
    f_getCOP = ca.Function('f_getCOP', [GRF, GRM], [COP])

    return f_getCOP



def calculate_rmse_casadi(dim, threshold=0.0001):
    
    # # Function variables: both are (dim x 1), which may be just scalar (1x1)
    # Mocap_R = ca.SX.sym('Mocap_R', dim, 1)
    # X_COP = ca.SX.sym('X_COP', dim, 1)

    # # Compute absolute error
    # abs_error = ca.fabs(Mocap_R - X_COP)

    # # Penalize only if outside the threshold
    # excess_error = ca.fmax(0, abs_error - threshold)

    # # Apply RMSE formula
    # rmse_thresh = ca.sqrt(ca.sumsqr(excess_error) / dim)

    # # Return CasADi function
    # f_thresh_rmse = ca.Function('f_thresh_COP_rmse', [Mocap_R, X_COP], [rmse_thresh])

    # return f_thresh_rmse
    
        # Function variables
    Mocap_R = ca.SX.sym('Mocap_R', dim, 1)
    X_COP = ca.SX.sym('X_COP', dim, 1)
    
    # Calculate RMSE
    rmse_r = ca.sqrt(ca.sumsqr(Mocap_R - X_COP) / dim)
    
    # Create CasADi function
    f_rmse = ca.Function('f_COP_rmse', [Mocap_R, X_COP], [rmse_r])
    
    return f_rmse


def copXOutsideBoundsPenalty():
    import casadi as ca

    def _penalty(COP, foot_x_min, foot_x_max, vGRF, weight_vector):
        COPX = COP[0]
        x_center = 0.5 * (foot_x_min + foot_x_max)
        x_radius = 0.5 * (foot_x_max - foot_x_min)
        delta = ca.fabs(COPX - x_center) - x_radius

        # Smooth Huber-like penalty: quadratic for small violations, linear for large
        eps = 1e-3  # smoothing threshold
        quad_region = ca.fmax(0, ca.fmin(delta, eps))
        linear_region = ca.fmax(0, delta - eps)
        smooth_penalty = 0.5 * quad_region**2 / eps + linear_region

        return ca.if_else(vGRF > 2.0, weight_vector[0] * smooth_penalty, 0.0)

    return _penalty

