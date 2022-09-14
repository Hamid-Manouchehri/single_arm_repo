'''
Author: Hamid Manouchehri
co authors: Mohammad Shahbazi, Nooshin Koohli
Year: 2022-2023
'''

import rbdl
import numpy as np


####### Computes inverse dynamics with the Newton-Euler Algorithm.
####### This function computes the generalized forces from given generalized states, velocities, and accelerations



def CalcM(model, q):
    """Compute join-space inertia matrices of arms."""

    M = np.zeros((model.q_size, model.q_size))  # (12*12)

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

    rbdl.InverseDynamics(model, q, qdot, np.zeros(model.qdot_size), H)  # (1*12)

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
    print(model.q_size)
    rbdl.InverseDynamics(model, q, qdot, qddot, Tau)

    return Tau



####### Computes position of the tip of left end-effector
def pose_end_l(model, q):

    tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])

    pose = rbdl.CalcBodyToBaseCoordinates(model, q,
                                          model.GetBodyId('hand_left'),
                                          tipOfEndEffectorInItsFrame)  # (1*3)

    orientationOfLeftHand = \
        rbdl.CalcBodyWorldOrientation(model, q, model.GetBodyId('hand_left'))

    eulerAngles = RotationMatToEuler(orientationOfLeftHand)
    pose = np.array([pose[0], pose[1], eulerAngles[2] - np.pi/2])

    return pose



####### Computes position of the tip of right end-effector
def pose_end_r(model, q):

    tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])

    pose = rbdl.CalcBodyToBaseCoordinates(model, q,
                                          model.GetBodyId('hand_right'),
                                          tipOfEndEffectorInItsFrame)  # (1*3)

    orientationOfRightHand = \
        rbdl.CalcBodyWorldOrientation(model, q, model.GetBodyId('hand_right'))

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

    poseOfRightHandFrameInWorldFrame_y = .773
    initialPoseOfObjInWorldFrame_y = .9
    poseOfCOMOfObjInRightHandFrame_y = initialPoseOfObjInWorldFrame_y - \
                                       poseOfRightHandFrameInWorldFrame_y
    poseOfCOMOfObjInRightHandFrame = \
                    np.array([poseOfCOMOfObjInRightHandFrame_y, 0.2 + .05, 0.])

    poseOfObj = rbdl.CalcBodyToBaseCoordinates(model, q,
                                               model.GetBodyId('hand_right'),
                                               poseOfCOMOfObjInRightHandFrame)

    rotationMatOfBox = \
        rbdl.CalcBodyWorldOrientation(model, q,
                                      model.GetBodyId('hand_right'))

    orientationOfBox = RotationMatToEuler(rotationMatOfBox)

    poseOfObj[2] = orientationOfBox[2] - np.pi/2  # object frame rotates '-pi/2' in 'hand_right' frame.

    return poseOfObj



###### Compute jacobian
def jc_left(model, q):

    workspaceDof = 3
    jc = np.zeros((workspaceDof, model.q_size))  # (3*12): due to whole 'model' and 'q' are imported.

    # tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])
    poseOfLeftHandFrameInWorldFrame_y = .773
    initialPoseOfObjInWorldFrame_y = .9
    poseOfCOMOfObjInLefttHandFrame_y = initialPoseOfObjInWorldFrame_y - \
                                       poseOfLeftHandFrameInWorldFrame_y
    poseOfCOMOfObjInLeftHandFrame = \
                    np.array([poseOfCOMOfObjInLefttHandFrame_y, -0.2 - .05, 0.])

    rbdl.CalcPointJacobian(model, q, model.GetBodyId('hand_left'),
                           poseOfCOMOfObjInLeftHandFrame, jc)  # (3*12)

    return jc[:, :6]




def jc_right(model, q):

    workspaceDof = 3
    jc = np.zeros((workspaceDof, model.q_size))  # (3*12): due to whole 'model' and 'q' are imported.

    # tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])
    poseOfRightHandFrameInWorldFrame_y = .773
    initialPoseOfObjInWorldFrame_y = .9
    poseOfCOMOfObjInRightHandFrame_y = initialPoseOfObjInWorldFrame_y - \
                                       poseOfRightHandFrameInWorldFrame_y
    poseOfCOMOfObjInRightHandFrame = \
                    np.array([poseOfCOMOfObjInRightHandFrame_y, 0.2 + .05, 0.])

    rbdl.CalcPointJacobian(model, q, model.GetBodyId('hand_right'),
                           poseOfCOMOfObjInRightHandFrame, jc)  # (3*12)

    return jc[:, 6:]



def CalcGeneralizedVelOfObject(model, q, qdot):
    """
    Calculate generalized velocity of the object via the right-hand ...
    kinematics in base frame.
    """
    poseOfRightHandFrameInWorldFrame_y = .773
    initialPoseOfObjInWorldFrame_y = .9
    poseOfCOMOfObjInRightHandFrame_y = initialPoseOfObjInWorldFrame_y - \
                                       poseOfRightHandFrameInWorldFrame_y
    poseOfCOMOfObjInRightHandFrame = \
                    np.array([poseOfCOMOfObjInRightHandFrame_y, 0.2 + .05, 0.])

    generalizedVelOfObj = \
        rbdl.CalcPointVelocity6D(model, q, qdot,
                                 model.GetBodyId('hand_right'),
                                 poseOfCOMOfObjInRightHandFrame)  # pose of COM of obj in 'hand_right' frame.

    angularVelOfObj = generalizedVelOfObj[:3]
    translationalVelOfObj = generalizedVelOfObj[3:6]

    generalizedVelOfObj = np.hstack((translationalVelOfObj[:2],
                                     angularVelOfObj[2]))

    return generalizedVelOfObj



def CalcdJdq_r(model, q, qdot, qddot):
    """Compute linear acceleration of a point on body."""

    # tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])
    poseOfRightHandFrameInWorldFrame_y = .773
    initialPoseOfObjInWorldFrame_y = .9
    poseOfCOMOfObjInRightHandFrame_y = initialPoseOfObjInWorldFrame_y - \
                                       poseOfRightHandFrameInWorldFrame_y
    poseOfCOMOfObjInRightHandFrame = \
                    np.array([poseOfCOMOfObjInRightHandFrame_y, 0.2 + .05, 0.])

    bodyAccel = \
        rbdl.CalcPointAcceleration(model, q, qdot, qddot,
                                   model.GetBodyId('hand_right'),
                                   poseOfCOMOfObjInRightHandFrame)  # (1*3)

    return bodyAccel



def CalcdJdq_l(model, q, qdot, qddot):
    """Compute linear acceleration of a point on body."""

    # tipOfEndEffectorInItsFrame = np.asarray([.15, 0.0, 0.0])
    poseOfLeftHandFrameInWorldFrame_y = .773
    initialPoseOfObjInWorldFrame_y = .9
    poseOfCOMOfObjInLefttHandFrame_y = initialPoseOfObjInWorldFrame_y - \
                                       poseOfLeftHandFrameInWorldFrame_y
    poseOfCOMOfObjInLeftHandFrame = \
                    np.array([poseOfCOMOfObjInLefttHandFrame_y, -0.2 - .05, 0.])

    bodyAccel = \
        rbdl.CalcPointAcceleration(model, q, qdot, qddot,
                                   model.GetBodyId('hand_left'),
                                   poseOfCOMOfObjInLeftHandFrame)  # (1*3)

    return bodyAccel
