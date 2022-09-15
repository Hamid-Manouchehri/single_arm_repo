'''
Author: Hamid Manouchehri
co authors: Mohammad Shahbazi, Nooshin Koohli
Year: 2022-2023
'''

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

    # M_r = M[6:12, 6:12]
    # M_l = M[:6, :6]
    #
    # M[:6, :6] = M_r
    # M[6:12, 6:12] = M_l

    return M



def CalcH(model, q, qdot):
    """Compute centrifugal, coriolis and gravity force terms."""
    damping = -2
    H = np.zeros(model.q_size)

    rbdl.InverseDynamics(model, q, qdot, np.zeros(model.qdot_size), H)  # (1*6)

    # D = np.eye(len(q))*damping
    # H = H - np.dot(D, qdot)
    #
    # H_l = H[:6]
    # H_r = H[6:12]
    #
    # H = np.concatenate((H_r, H_l))

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

    orientationOfRightHand = \
        rbdl.CalcBodyWorldOrientation(model, q, model.GetBodyId('hand'))

    eulerAngles = RotationMatToEuler(orientationOfRightHand)
    pose = np.array([pose[0], pose[1], eulerAngles[2] - np.pi/2])

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



####### Compute position of COM of the object:
def GeneralizedPoseOfObj(model, q):

    tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])

    poseOfObj = rbdl.CalcBodyToBaseCoordinates(model, q,
                                               model.GetBodyId('hand'),
                                               tipOfEndEffectorInItsFrame)

    rotationMatOfBox = \
        rbdl.CalcBodyWorldOrientation(model, q,
                                      model.GetBodyId('hand'))

    orientationOfBox = RotationMatToEuler(rotationMatOfBox)

    generalizedPoseOfEndEffector = np.concatenate((poseOfObj, orientationOfBox))

    return generalizedPoseOfEndEffector




def jc(model, q):

    workspaceDof = 6
    jc = np.zeros((workspaceDof, model.q_size))  # (3*12): due to whole 'model' and 'q' are imported.

    tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])

    rbdl.CalcPointJacobian(model, q, model.GetBodyId('hand'),
                           tipOfEndEffectorInItsFrame, jc)  # (3*12)

    return jc



def CalcGeneralizedVelOfObject(model, q, qdot):
    """
    Calculate generalized velocity of the object via the right-hand ...
    kinematics in base frame.
    """
    tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])

    generalizedVelOfObj = rbdl.CalcPointVelocity6D(model, q, qdot,
                                                   model.GetBodyId('hand'),
                                                   tipOfEndEffectorInItsFrame)

    angularVelOfObj = generalizedVelOfObj[:3]
    translationalVelOfObj = generalizedVelOfObj[3:6]

    generalizedVelOfObj = np.hstack((translationalVelOfObj, angularVelOfObj))

    return generalizedVelOfObj



def CalcdJdq(model, q, qdot, qddot):
    """Compute linear acceleration of a point on body."""

    tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])

    bodyAccel = rbdl.CalcPointAcceleration6D(model, q, qdot, qddot,
                                             model.GetBodyId('hand'),
                                             tipOfEndEffectorInItsFrame)  # (1*3)

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

    # if t is None:
    #     ## to set the time if it is necessary.
    #     t = time_

    with open(pathToCSVFile + CSVFileName_plot_data, 'a', newline='') as file:

        writer = csv.writer(file)

        if writeHeaderOnceFlag is True and legendList is not None:
            ## Add header to the CSV file.
            writer.writerow(np.hstack(['time', legendList]))
            writeHeaderOnceFlag = False

        writer.writerow(np.hstack([t, data]))  # the first element is time var.
