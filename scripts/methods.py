'''
Author: Hamid Manouchehri
co authors: Mohammad Shahbazi, Nooshin Koohli
Year: 2022-2023
'''
import pylab as pl
from math import pi
import rbdl
import numpy as np
import csv
from sys import path
path.insert(1, '/home/rebel/bimanual_ws/src/single_arm_pkg/config/')  # TODO: Insert dir of config folder of the package.
import config

plotLegend = ''
writeHeaderOnceFlag = True

CSVFileName_plot_data = config.main_single_arm_dic['CSVFileName']
pathToCSVFile = config.main_single_arm_dic['CSVFileDirectory']

####### Computes inverse dynamics with the Newton-Euler Algorithm.
####### This function computes the generalized forces from given generalized states, velocities, and accelerations



def CalcM(model, q):
    """Compute join-space inertia matrices of arms."""

    M = np.zeros((model.q_size, model.q_size))  # (6*6)
    rbdl.CompositeRigidBodyAlgorithm(model, q, M, True)

    return M



def CalcH(model, dampingVec, q, qdot, qddot, M):
    """Compute centrifugal, coriolis and gravity force terms."""
    q = np.array(q, dtype=float)
    qdot = np.array(qdot, dtype=float)
    tau = np.zeros(model.q_size)

    rbdl.InverseDynamics(model, q, qdot, np.zeros(model.qdot_size), tau)  # (1*6)
    H = tau - M.dot(qddot)

    return H



def InverseDynamic(model, q, qdot, qddot):

    Tau = np.zeros(model.q_size)
    rbdl.InverseDynamics(model, q, qdot, qddot, Tau)

    return Tau



####### Computes position of the tip of right end-effector
def pose_end(model, q):

    tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])

    pose = rbdl.CalcBodyToBaseCoordinates(model, q,
                                          model.GetBodyId('hand'),
                                          tipOfEndEffectorInItsFrame)  # (1*3)

    return pose



#######
def RotationMatToEuler(rotationMat):
    """Calculate 'xyz' Euler angles in radians."""
    r11, r12, r13 = rotationMat[0]  # first row
    r21, r22, r23 = rotationMat[1]  # second row
    r31, r32, r33 = rotationMat[2]  # third row

    theta1 = np.arctan2(-r23, r33)
    theta2 = np.arctan2(r13 * np.cos(theta1), r33)
    theta3 = np.arctan2(-r12, r11)

    thetaVec = np.array([-theta1, -theta2, -theta3])

    return thetaVec  # in radians


def rotationToVtk(R):
    '''
    Concert a rotation matrix into the Mayavi/Vtk rotation paramaters (pitch, roll, yaw)
    '''
    def euler_from_matrix(matrix):
        """Return Euler angles (syxz) from rotation matrix for specified axis sequence.
        :Author:
          `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

        full library with coplete set of euler triplets (combinations of  s/r x-y-z) at
            http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

        Note that many Euler angle triplets can describe one matrix.
        """
        # epsilon for testing whether a number is close to zero
        _EPS = np.finfo(float).eps * 5.0

        # axis sequences for Euler angles
        _NEXT_AXIS = [1, 2, 0, 1]
        firstaxis, parity, repetition, frame = (1, 1, 0, 0) # ''

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        M = np.array(matrix, dtype='float', copy=False)[:3, :3]
        if repetition:
            sy = np.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
            if sy > _EPS:
                ax = np.arctan2( M[i, j],  M[i, k])
                ay = np.arctan2( sy,       M[i, i])
                az = np.arctan2( M[j, i], -M[k, i])
            else:
                ax = np.arctan2(-M[j, k],  M[j, j])
                ay = np.arctan2( sy,       M[i, i])
                az = 0.0
        else:
            cy = np.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
            if cy > _EPS:
                ax = np.arctan2( M[k, j],  M[k, k])
                ay = np.arctan2(-M[k, i],  cy)
                az = np.arctan2( M[j, i],  M[i, i])
            else:
                ax = np.arctan2(-M[j, k],  M[j, j])
                ay = np.arctan2(-M[k, i],  cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az
    # r_yxz = pl.array(euler_from_matrix(R))*180/pi
    r_yxz = pl.array(euler_from_matrix(R))
    r_xyz = r_yxz[[1, 0, 2]]
    # r_xyz = -r_yxz[[1, 0, 2]]
    return r_xyz





####### Compute position of COM of the object:
def GeneralizedPoseOfObj(model, q):

    lengthOfBox = .25
    poseOfObjInHandFrame = np.asarray([lengthOfBox/2, 0.25, 0.0])
    # poseOfObjInHandFrame = np.asarray([0., 0., 0.])
    print(model.GetBodyId('hand'))

    poseOfObj = rbdl.CalcBodyToBaseCoordinates(model, q,
                                               model.GetBodyId('hand'),
                                               poseOfObjInHandFrame)

    rotationMatOfBox = rbdl.CalcBodyWorldOrientation(model, q,
                                                     model.GetBodyId('hand'))

    orientationOfBox = RotationMatToEuler(rotationMatOfBox)
    # orientationOfBox = rotationToVtk(rotationMatOfBox)
    # print(orientationOfBox)
    print(rotationMatOfBox.dot(poseOfObjInHandFrame))

    generalizedPoseOfEndEffector = np.concatenate((poseOfObj, orientationOfBox))

    return generalizedPoseOfEndEffector



def Jacobian(model, q):

    workspaceDof = 6
    jc = np.zeros((workspaceDof, model.q_size))  # (6*6): due to whole 'model' and 'q' are imported.

    lengthOfBox = .25
    poseOfObjInHandFrame = np.asarray([lengthOfBox/2, 0.25, 0.0])
    # poseOfObjInHandFrame = np.asarray([0., 0., 0.0])

    rbdl.CalcPointJacobian6D(model, q, model.GetBodyId('hand'),
                             poseOfObjInHandFrame, jc)  # (6*6)

    jc_r = jc[3:, :]
    jc_l = jc[:3, :]
    jc = np.vstack((jc_r, jc_l))

    return jc



def CalcGeneralizedVelOfObject(model, q, qdot):
    """
    Calculate generalized velocity of the object via the right-hand ...
    kinematics in base frame.
    """
    lengthOfBox = .25
    poseOfObjInHandFrame = np.asarray([lengthOfBox/2, 0.25, 0.0])
    # poseOfObjInHandFrame = np.asarray([0., 0., 0.0])

    generalizedVelOfObj = rbdl.CalcPointVelocity6D(model, q, qdot,
                                                   model.GetBodyId('hand'),
                                                   poseOfObjInHandFrame)

    angularVelOfObj = generalizedVelOfObj[:3]
    translationalVelOfObj = generalizedVelOfObj[3:6]

    generalizedVelOfObj = np.hstack((translationalVelOfObj, angularVelOfObj))

    return generalizedVelOfObj



def CalcdJdq(model, q, qdot, qddot):
    """Compute linear acceleration of a point on body."""

    lengthOfBox = .25
    poseOfObjInHandFrame = np.asarray([lengthOfBox/2, 0.25, 0.0])
    # poseOfObjInHandFrame = np.asarray([0., 0., 0.0])

    bodyAccel = rbdl.CalcPointAcceleration6D(model, q, qdot, qddot,
                                             model.GetBodyId('hand'),
                                             poseOfObjInHandFrame)  # (1*3)

    return bodyAccel



def WriteToCSV(data, t, legendList=None):
    """
    Write data to the CSV file to have a live plot by reading the file ...
    simulataneously.

    Pass 'data' as a list of  data and 'legendList' as a list of string type,
    legend for each value of the plot.
    Note: 'legendList' and 't' are arbitrary arguments.
    """
    global plotLegend, writeHeaderOnceFlag

    plotLegend = legendList

    with open(pathToCSVFile + CSVFileName_plot_data, 'a', newline='') as file:

        writer = csv.writer(file)

        if writeHeaderOnceFlag is True and legendList is not None:
            ## Add header to the CSV file.
            writer.writerow(np.hstack(['time', legendList]))
            writeHeaderOnceFlag = False

        writer.writerow(np.hstack([t, data]))  # the first element is time var.
