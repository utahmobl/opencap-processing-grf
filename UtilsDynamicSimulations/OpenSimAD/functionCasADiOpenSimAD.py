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



def derivativeSumOfNegatives(dim1, dim2):
    """
    Computes a weighted squared sum of negative derivatives along columns.
    
    Args:
        dim1 (int): Number of rows (e.g., 52 for time steps).
        dim2 (int): Number of columns (e.g., 3 for COP data).
    
    Returns:
        casadi.Function: A function that computes the weighted squared sum.
    """
    # Function variables
    x = ca.SX.sym('x', dim1, dim2)  # Input matrix (dim1 x dim2)
    w = ca.SX.sym('w', dim2, 1)  # Weight vector (dim2 x 1)

    # Compute finite differences along columns (derivative)
    derivative = x[1:, :] - x[:-1, :]  # (dim1-1 x dim2)

    # **Very large penalty for non-increasing values**
    non_increasing_penalty = (ca.sum1(ca.fabs(ca.fmin(derivative, 0)))**2)*100
   # **Extreme penalty for non-increasing values**
    #non_increasing_penalty = ca.if_else(derivative < 0, ca.exp(-10 * derivative), 0)  # Large multiplier

    # **Final weighted penalty**
    weighted_penalty = ca.mtimes(non_increasing_penalty, w)

    # Create CasADi function
    f_derivativeSumOfNegatives = ca.Function('f_derivativeSumOfNegatives', [x, w], [weighted_penalty])
    return f_derivativeSumOfNegatives



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
    derivative_penalty = (ca.sum1(ca.fmax(ca.fabs(derivative2), 0.01))**2)

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



def calculate_rmse_casadi(dim):
    
    # Function variables
    Mocap_R = ca.SX.sym('Mocap_R', dim, 1)
    X_COP = ca.SX.sym('X_COP', dim, 1)
    
    # Calculate RMSE
    rmse_r = ca.sqrt(ca.sumsqr(Mocap_R - X_COP) / dim)
    
    # Create CasADi function
    f_rmse = ca.Function('f_COP_rmse', [Mocap_R, X_COP], [rmse_r])
    
    return f_rmse


