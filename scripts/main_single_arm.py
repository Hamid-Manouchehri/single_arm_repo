#!/usr/bin/env python3
"""
Created on Sat Aug 27 11:20:33 2022

@author: Hamid Manouchehri
@co authors: Mohammad Shahbazi, Nooshin Kohli
"""
from __future__ import division
from numpy.linalg import inv, pinv
from scipy.linalg import qr, sqrtm
import rbdl
import os
import csv
import time
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
from sys import path
path.insert(1, '/home/rebel/bimanual_ws/src/single_arm_pkg/config/')  # TODO: Insert dir of config folder of the package.
import config
os.chdir('/home/rebel/bimanual_ws/src/single_arm_pkg/scripts/')  # TODO: Insert dir of config folder of the package.
import methods  # TODO: change the file name if necessary.
####


time_ = 0
dt = 0.01
t_end = 6
g0 = 9.81
finiteTimeSimFlag = True  # TODO: True: simulate to 't_end', Flase: Infinite time

workspaceDof = 6  # TODO
singleArmDof = 6  # TODO
W_invDyn = np.eye(singleArmDof)

kp_a = 110
kd_a = kp_a/9

## TODO: damping coefficients must be set similar to xacro model:
dampingCoeff = np.array([0.]*singleArmDof)
dampingCoeff[0] = 1  # arm_zero
dampingCoeff[1] = 1  # arm_one
dampingCoeff[2] = 1  # arm_two
dampingCoeff[3] = 1  # arm_three
dampingCoeff[4] = .1  # arm_four
dampingCoeff[5] = .1  # hand


# Object parameters:
mass_box = .1  # TODO: Check the parameters of the box in the 'upper_body.xacro' file.
width_box = .4  # TODO
length_box = .25  # TODO
io_zz = 1/12.*mass_box*(width_box**2 + length_box**2)

M_o = np.eye(workspaceDof)*mass_box
M_o[5, 5] = io_zz
h_o = np.array([[0.], [mass_box*g0], [0.], [0.], [0.], [0.]])

"""
(x(m), y(m), z(m), roll(rad), pitch(rad), yaw(rad)):
Note: any modification to trajectory need modification of 'LABEL_1',
function of 'ChooseRef' and of course the following trajecory definitions:

Note: translational coordinations (x, y, z) are in world frame, so avoid values
which are close to 'child_neck' frame.
"""
poseOfObjInWorld_x = 0.
poseOfObjInWorld_y = .773 + length_box/2
poseOfObjInWorld_z = .195

desiredInitialStateOfObj_traj_1 = np.array([poseOfObjInWorld_x, poseOfObjInWorld_y, poseOfObjInWorld_z, 0., 0., np.pi/2])
desiredFinalStateOfObj_traj_1 = np.array([poseOfObjInWorld_x, .6, poseOfObjInWorld_z, 0., 0., np.pi/2])

desiredInitialStateOfObj_traj_2 = desiredFinalStateOfObj_traj_1
desiredFinalStateOfObj_traj_2 = np.array([poseOfObjInWorld_x, poseOfObjInWorld_y, poseOfObjInWorld_z, 0., 0., np.pi/2])

desiredInitialStateOfObj_traj_3 = desiredFinalStateOfObj_traj_2
desiredFinalStateOfObj_traj_3 = desiredInitialStateOfObj_traj_1


initPoseVelAccelOfObj_traj_1 = [desiredInitialStateOfObj_traj_1, np.zeros(workspaceDof), np.zeros(workspaceDof)]
finalPoseVelAccelOfObj_traj_1 = [desiredFinalStateOfObj_traj_1, np.zeros(workspaceDof), np.zeros(workspaceDof)]

initPoseVelAccelOfObj_traj_2 = [desiredInitialStateOfObj_traj_2, np.zeros(workspaceDof), np.zeros(workspaceDof)]
finalPoseVelAccelOfObj_traj_2 = [desiredFinalStateOfObj_traj_2, np.zeros(workspaceDof), np.zeros(workspaceDof)]

initPoseVelAccelOfObj_traj_3 = [desiredInitialStateOfObj_traj_3, np.zeros(workspaceDof), np.zeros(workspaceDof)]
finalPoseVelAccelOfObj_traj_3 = [desiredFinalStateOfObj_traj_3, np.zeros(workspaceDof), np.zeros(workspaceDof)]

q_obj = [desiredInitialStateOfObj_traj_1]
q = [[0.]*singleArmDof]
q = np.hstack((q, q_obj))

q_des = [q[-1][:singleArmDof]]

qctrl_des_prev = q_des[-1][:singleArmDof]
dqctrl_des_prev = np.zeros(singleArmDof)  # TODO: we usually start from rest


CSVFileName_plot_data = config.main_single_arm_dic['CSVFileName']
pathToCSVFile = config.main_single_arm_dic['CSVFileDirectory']

upperBodyModelFileName = config.main_single_arm_dic['urdfModelName']
pathToArmURDFModels = config.main_single_arm_dic['urdfDirectory']

loaded_model = rbdl.loadModel(pathToArmURDFModels + upperBodyModelFileName)

## create instances of publishers:
pub_0 = rospy.Publisher('/arm/arm_zero_joint_effort_controller/command',
                        Float64, queue_size=10)
pub_1 = rospy.Publisher('/arm/arm_one_joint_effort_controller/command',
                        Float64, queue_size=10)
pub_2 = rospy.Publisher('/arm/arm_two_joint_effort_controller/command',
                        Float64, queue_size=10)
pub_3 = rospy.Publisher('/arm/arm_three_joint_effort_controller/command',
                        Float64, queue_size=10)
pub_4 = rospy.Publisher('/arm/arm_four_joint_effort_controller/command',
                        Float64, queue_size=10)
pub_hand = rospy.Publisher('/arm/hand_joint_effort_controller/command',
                           Float64, queue_size=10)

rospy.init_node('main_single_arm_node')

def TrajEstimate(qdd):
    Dt = dt
    qdot_des_now = dqctrl_des_prev + Dt*qdd  # numerical derivation
    q_des_now = qctrl_des_prev + Dt*qdot_des_now

    q_des_now_filtered = []

    for i in q_des_now:
        i = np.mod(i, np.pi*2)
        if i > np.pi:
            i = i - np.pi*2
        q_des_now_filtered.append(i)

    return np.array(q_des_now_filtered), qdot_des_now


def TrajPlan(t_start, t_end, z_start_o, z_end_o, traj_type='Quantic'):
    """
    Compute desired generalize trajectory, velocity and acceleration ...
    of the Object.
    """
    initGeneralizedPose, initGeneralizedVel, initGeneralizedAccel = z_start_o
    finalGeneralizedPose, finalGeneralizedVel, finalGeneralizedAccel = z_end_o

    def f(x):
        return [x**5, x**4, x**3, x**2, x, 1]

    def fd(x):
        return [5*x**4, 4*x**3, 3*x**2, 2*x, 1, 0]

    def fdd(x):
        return [20*x**3, 12*x**2, 6*x, 2, 0, 0]

    if traj_type == 'Quantic':

        A = np.array([f(t_start), fd(t_start), fdd(t_start),
                      f(t_end), fd(t_end), fdd(t_end)])

        desiredGeneralizedTraj = []
        desiredGeneralizedVel = []
        desiredGeneralizedAccel = []

        """
        Iterate to calculate all the states of the object for position,
        velocity and acceleration then append to the list:
        """
        for i in range(len(initGeneralizedPose)):

            B = np.array([[initGeneralizedPose[i], initGeneralizedVel[i],
                           initGeneralizedAccel[i], finalGeneralizedPose[i],
                           finalGeneralizedVel[i],
                           finalGeneralizedAccel[i]]]).T

            p = np.dot(np.linalg.inv(A), B)

            desiredGeneralizedTraj += list([lambda x, p=p:
                                            sum([p[0]*x**5, p[1]*x**4,
                                                 p[2]*x**3, p[3]*x**2,
                                                 p[4]*x, p[5]])])

            desiredGeneralizedVel.append(lambda x, p=p:
                                         sum([p[0]*5*x**4, p[1]*4*x**3,
                                              p[2]*3*x**2, p[3]*2*x, p[4]]))

            desiredGeneralizedAccel.append(lambda x, p=p:
                                           sum([p[0]*20*x**3, p[1]*12*x**2,
                                                p[2]*6*x, p[3]*2]))

    return [desiredGeneralizedTraj, desiredGeneralizedVel,
            desiredGeneralizedAccel]


def PubTorqueToGazebo(torqueVec):
    """
    Publish torques to Gazebo (manipulate the object in a linear trajectory)
    """
    pub_0.publish(torqueVec[0])  # arm_zero
    pub_1.publish(torqueVec[1])  # arm_one
    pub_2.publish(torqueVec[2])  # arm_two
    pub_3.publish(torqueVec[3])  # arm_three
    pub_4.publish(torqueVec[4])  # arm_four
    pub_hand.publish(torqueVec[5])  # hand
    # pub_hand.publish(-1.5)  # hand


def GdotVDot_o(model, qCurrent, qDotCurrent, r_o_a):

    poseOfObj = methods.GeneralizedPoseOfObj(model, qCurrent)
    velOfObj = methods.CalcGeneralizedVelOfObject(model, qCurrent, qDotCurrent)
    velOfObj = CalcEulerGeneralizedVel(poseOfObj, velOfObj)

    ## Calculate Gdot_oi:
    zeroMat = np.zeros([3, 3])
    block1 = np.concatenate((zeroMat, zeroMat), 0)  # (6*3): vertical stack

    omegaCrossRoaCrossOmega = \
                        np.cross(velOfObj[3:], np.cross(r_o_a, velOfObj[3:]))

    block2_a = SkewSym(np.array(omegaCrossRoaCrossOmega))  # (3*3)

    block3_a = np.concatenate((block2_a, zeroMat), 0)  # (6*3): vertical stack
    Gdot_oa = np.concatenate((block1, block3_a), 1)  # (6*6): horizontal stack

    Gdot_oaV_o = Gdot_oa.dot(velOfObj)  # GDot_oa*qDot_o:(1*6)

    return Gdot_oaV_o


def SkewSym(inputVector):
    """Compute skew symetric matrix of 'inputList' vector."""
    inputVector = np.array([inputVector])
    zeroMat = np.zeros((3, 3))
    zeroMat[0, 1] = -inputVector[0, 2]
    zeroMat[0, 2] = inputVector[0, 1]
    zeroMat[1, 2] = -inputVector[0, 0]
    zeroMat = zeroMat + (-zeroMat.T)

    return zeroMat  # (3*3)


def CalcGoi(r_oi):

    P_oi_cross = SkewSym(r_oi[:3])
    G_oi_topRows = np.hstack((np.eye(3), P_oi_cross))
    G_oi_bottomRows = np.hstack((np.zeros((3, 3)), np.eye(3)))
    G_oi = np.vstack((G_oi_topRows, G_oi_bottomRows))  # (6*6)

    return G_oi


def CalcPoseErrorInQuaternion(desPose, currentPose, zDesQuater, zCurrentQuater):
    """Calculate position error in Quaternion based on the paper."""

    wDes = zDesQuater[0]
    vDes = zDesQuater[1:]

    wCurrent = zCurrentQuater[0]
    vCurrent = zCurrentQuater[1:]

    angularError = np.dot(wCurrent, vDes) - np.dot(wDes, vCurrent) - \
                                            np.cross(vDes, vCurrent)
    # print(np.round(angularError, 3))

    translationalError = desPose[:3] - currentPose[:3]

    generalizedError = np.concatenate((translationalError, angularError))

    return generalizedError



def EulerToQuaternion(z):
    """Calculate quaternion of input Euler angles (radians)."""
    roll = z[3]
    pitch = z[4]
    yaw = z[5]

    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
         np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
         np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
         np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)

    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
         np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

    return [qw, qx, qy, qz]



def CalcEulerGeneralizedAccel(angularPose, eulerVel, eulerAccel):
    """
    Calculate conversion of Euler acceleration (ddRoll, ddPitch, ddYaw)
    into angular acceleration (alpha_x, alpha_y, alpha_z):
    """
    ## roll = angularPose[3]
    pitch = angularPose[4]
    yaw = angularPose[5]

    dRoll = eulerVel[3]
    dPitch = eulerVel[4]
    dYaw = eulerVel[5]

    ddRoll = eulerAccel[3]
    ddPitch = eulerAccel[4]
    ddYaw = eulerAccel[5]

    ## Time derivative of 'transformMat.objEulerVel':
    alpha_x = ddRoll*np.sin(pitch)*np.sin(yaw) + \
              dRoll*dPitch*np.cos(pitch)*np.sin(yaw) + \
              dRoll*dYaw*np.sin(pitch)*np.cos(yaw) + ddPitch*np.cos(yaw) - \
              dPitch*dYaw*np.sin(yaw)

    alpha_y = ddRoll*np.sin(pitch)*np.cos(yaw) + \
              dRoll*dPitch*np.cos(pitch)*np.cos(yaw) - \
              dRoll*dYaw*np.sin(pitch)*np.sin(yaw) - ddPitch*np.sin(yaw) - \
              dPitch*dYaw*np.cos(yaw)

    alpha_z = ddRoll*np.cos(pitch) - dRoll*dPitch*np.sin(pitch) + ddYaw

    desiredGeneralizedAccelOfObj = np.concatenate((eulerAccel[:3],
                                                   [alpha_x, alpha_y, alpha_z]))

    return desiredGeneralizedAccelOfObj



def CalcEulerGeneralizedVel(xObj, xDotObj):
    """
    Calculate conversion of Euler angles rate of change (dRoll, dPitch, dYaw)
    into angular velocity (omega_x, omega_y, omega_z):
    """
    # roll = xObj[3]
    pitch = xObj[4]
    yaw = xObj[5]

    transformMat = np.array([[np.sin(pitch)*np.sin(yaw), np.cos(yaw), 0],
                             [np.sin(pitch)*np.cos(yaw), -np.sin(yaw), 0],
                             [np.cos(pitch), 0, 1]])

    omegaVec = transformMat.dot(xDotObj[3:])

    desiredGeneralizedVelOfObj = np.concatenate((xDotObj[:3], omegaVec))

    return desiredGeneralizedVelOfObj



def QRDecompose(J_T):
    m, n = J_T.shape

    if m == 0 or n == 0:
        raise TypeError(
            'Try to calculate QR decomposition, while there is no contact!')

    qr_Q, qr_R = qr(J_T)

    qr_R = qr_R[:n, :]

    return qr_Q, qr_R



def InverseDynamic(qCurrent, qDotCurrent, qDDotCurrent, qDes, qDotDes, qDDotDes):

    jac = methods.Jacobian(loaded_model, qCurrent)
    M = methods.CalcM(loaded_model, qCurrent)
    h = methods.CalcH(loaded_model, dampingCoeff, qCurrent, qDotCurrent, qDDotCurrent, M)

    desiredTorque = M.dot(qDDotDes) + h

    return desiredTorque



def Task2Joint(qCurrent, qDotCurrent, qDDotCurrent, poseDes, velDes, accelDes):

    jac = methods.Jacobian(loaded_model, qCurrent)
    poseOfObj = methods.GeneralizedPoseOfObj(loaded_model, qCurrent)
    velOfObj = methods.CalcGeneralizedVelOfObject(loaded_model, qCurrent,
                                                  qDotCurrent)
    velOfObj = CalcEulerGeneralizedVel(poseOfObj, velOfObj)


    zDesObjInQuater = EulerToQuaternion(poseDes)
    zCurrentObjInQuater = EulerToQuaternion(poseOfObj)

    poseErrorInQuater = CalcPoseErrorInQuaternion(poseDes,
                                                  poseOfObj,
                                                  zDesObjInQuater,
                                                  zCurrentObjInQuater)

    ## control acceleration of end-effector in task-space: below equ(21)
    accelDes = accelDes + kd_a * (velDes - velOfObj) + kp_a * poseErrorInQuater

    dJdq = methods.CalcdJdq(loaded_model, qCurrent, qDotCurrent, qDDotCurrent)  # JDot_a*qDot_a(1*6)

    qDDotDes = pinv(jac).dot(accelDes - dJdq).flatten()  # equ(21)

    qDes, qDotDes = TrajEstimate(qDDotDes)

    return qDes, qDotDes, qDDotDes



def CalcDesiredTraj(xDes, xDotDes, xDDotDes, t):
    """ trajectory generation of end-effector in task-space:"""

    xDes_now = np.array([xDes[i](t) for i in rlx]).flatten()
    xDotDes_now = np.array([xDotDes[i](t) for i in rlx]).flatten()
    xDDotDes_now = np.array([xDDotDes[i](t) for i in rlx]).flatten()

    return xDes_now, xDotDes_now, xDDotDes_now


def ChooseRef(time):
    if time <= t_end/3:
        out = [desPose_traj_1, desVel_traj_1, desAccel_traj_1, time]

    elif time <= 2*t_end/3:
        out = [desPose_traj_2, desVel_traj_2, desAccel_traj_2, time - t_end/3]

    else:
        out = [desPose_traj_3, desVel_traj_3, desAccel_traj_3, time - 2*t_end/3]

    return out



def JointStatesCallback(data):
    """
    Subscribe to robot's joints' position and velocity and publish torques.
    """
    global time_

    ## Terminate the node after 't_end' seconds of simulation:
    if time_ >= t_end and finiteTimeSimFlag:
        print("\n\nshut down 'main_node'.\n\n")
        rospy.signal_shutdown('node terminated')

    q = data.position  # class tuple, in radians
    qDot = data.velocity  # in rad/s

    q = np.array(q, dtype=float)
    qDot = np.array(qDot, dtype=float)

    qCurrent = np.zeros(singleArmDof)
    qDotCurrent = np.zeros(singleArmDof)
    qDDotCurrent = np.zeros(singleArmDof)

    qCurrent[0] = q[4]  # arm_zero
    qCurrent[1] = q[1]  # arm_one
    qCurrent[2] = q[3]  # arm_two
    qCurrent[3] = q[2]  # arm_three
    qCurrent[4] = q[0]  # arm_four
    qCurrent[5] = q[5]  # hand

    qDotCurrent[0] = qDot[4]  # arm_zero
    qDotCurrent[1] = qDot[1]  # arm_one
    qDotCurrent[2] = qDot[3]  # arm_two
    qDotCurrent[3] = qDot[2]  # arm_three
    qDotCurrent[4] = qDot[0]  # arm_four
    qDotCurrent[5] = qDot[5]  # hand


    xDes_t, xDotDes_t, xDDotDes_t, timePrime = ChooseRef(time_)
    xDesObj, xDotDesObj, xDDotDesObj = CalcDesiredTraj(xDes_t, xDotDes_t,
                                                       xDDotDes_t, timePrime)

    desiredGeneralizedVelOfObj = CalcEulerGeneralizedVel(xDesObj, xDotDesObj)

    desiredGeneralizedAccelOfObj = CalcEulerGeneralizedAccel(xDesObj,
                                                             xDotDesObj,
                                                             xDDotDesObj)

    ## detemine desired states of robot in joint-space: (qDDot_desired, ...)
    jointPose, jointVel, jointAccel = Task2Joint(qCurrent, qDotCurrent,
                                                 qDDotCurrent, xDesObj,
                                                 desiredGeneralizedVelOfObj,
                                                 desiredGeneralizedAccelOfObj)

    # jointTau = methods.InverseDynamic(loaded_model, jointPose, jointVel, jointAccel)
    jointTau = InverseDynamic(qCurrent, qDotCurrent, qDDotCurrent, jointPose,
                              jointVel, jointAccel)

    PubTorqueToGazebo(jointTau)

    time_ += dt


def RemoveCSVFile(path, fileName):
    """Remove the csv file to avoid appending data to the preivous data."""
    if os.path.isfile(path + fileName) is True:
        os.remove(path + fileName)

    else:
        pass  # Do nothing.


if __name__ == '__main__':

    RemoveCSVFile(pathToCSVFile, CSVFileName_plot_data)

    ## Generate desired states for the whole trajectory of object: (LABEL_1)
    ## First trajectory:
    desPose_traj_1, desVel_traj_1, desAccel_traj_1 = \
        TrajPlan(0, t_end/3, initPoseVelAccelOfObj_traj_1, finalPoseVelAccelOfObj_traj_1)

    ## Second trajectory:
    desPose_traj_2, desVel_traj_2, desAccel_traj_2 = \
        TrajPlan(0, t_end/3, initPoseVelAccelOfObj_traj_2, finalPoseVelAccelOfObj_traj_2)

    ## Third trajectory:
    desPose_traj_3, desVel_traj_3, desAccel_traj_3 = \
        TrajPlan(0, t_end/3, initPoseVelAccelOfObj_traj_3, finalPoseVelAccelOfObj_traj_3)

    rlx = range(len(desPose_traj_1))  # range(0, 6) --> 0, 1, 2, 3, 4, 5

    try:
        rospy.Subscriber("/arm/joint_states", JointState, JointStatesCallback)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
