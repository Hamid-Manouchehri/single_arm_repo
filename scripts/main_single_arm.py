#!/usr/bin/env python3
"""
Created on Sat Aug 27 11:20:33 2022

@author: Hamid Manouchehri
@co authors: Mohammad Shahbazi, Nooshin Kohli
"""
from __future__ import division
from numpy.linalg import inv, pinv
from scipy.linalg import qr
import rbdl
from scipy.linalg import sqrtm
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
t_end = 3
g0 = 9.81
finiteTimeSimFlag = True  # TODO: True: simulate to 't_end', Flase: Infinite time

workspaceDof = 6  # TODO
singleArmDof = 6  # TODO

kp_a = 110
kd_a = kp_a/8

startTime = 0
"""
(x(m), y(m), z(m), roll(rad), pitch(rad), yaw(rad)):
Note: any modification to trajectory need modification of 'LABEL_1',
function of 'ChooseRef' and of course the following trajecory definitions:
"""
desiredInitialStateOfObj_traj_1 = np.array([0.25, 0.775 + .15, 0.202, 0., 0., np.pi/2])
desiredFinalStateOfObj_traj_1 = np.array([0., .5, 0.202, 0., 0., np.pi/2])

desiredInitialStateOfObj_traj_2 = desiredFinalStateOfObj_traj_1
desiredFinalStateOfObj_traj_2 = np.array([0.25, 0.775 + .15, 0.202, 0., 0., np.pi/2])

desiredInitialStateOfObj_traj_3 = desiredFinalStateOfObj_traj_2
desiredFinalStateOfObj_traj_3 = desiredInitialStateOfObj_traj_1


initPoseVelAccelOfObj_traj_1 = [desiredInitialStateOfObj_traj_1, np.zeros(workspaceDof), np.zeros(workspaceDof)]
finalPoseVelAccelOfObj_traj_1 = [desiredFinalStateOfObj_traj_1, np.zeros(workspaceDof), np.zeros(workspaceDof)]

initPoseVelAccelOfObj_traj_2 = [desiredInitialStateOfObj_traj_2, np.zeros(workspaceDof), np.zeros(workspaceDof)]
finalPoseVelAccelOfObj_traj_2 = [desiredFinalStateOfObj_traj_2, np.zeros(workspaceDof), np.zeros(workspaceDof)]

initPoseVelAccelOfObj_traj_3 = [desiredInitialStateOfObj_traj_3, np.zeros(workspaceDof), np.zeros(workspaceDof)]
finalPoseVelAccelOfObj_traj_3 = [desiredFinalStateOfObj_traj_3, np.zeros(workspaceDof), np.zeros(workspaceDof)]


# Object params
mass_box = .1  # TODO: Check the parameters of the box in the 'upper_body.xacro' file.
width_box = .4  # TODO
length_box = .25  # TODO
io_zz = 1/12.*mass_box*(width_box**2 + length_box**2)

M_o = np.eye(workspaceDof)*mass_box
M_o[2, 2] = io_zz
h_o = np.array([[0], [mass_box*g0], [0]])

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


def Traj_Estimate(qdd):
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


def traj_plan(t_start, t_end, z_start_o, z_end_o, traj_type='Quantic'):
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


def ChooseRef(time):
    if time <= t_end/3:
        out = [desPose_traj_1, desVel_traj_1, desAccel_traj_1, time]

    elif time <= 2*t_end/3:
        out = [desPose_traj_2, desVel_traj_2, desAccel_traj_2, time - t_end/3]

    else:
        out = [desPose_traj_3, desVel_traj_3, desAccel_traj_3, time - 2*t_end/3]

    return out


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



def Task2Joint_(qCurrent, qDotCurrent, qDDotCurrent, poseDes, velDes, accelDes):

    jac = methods.jc(loaded_model, qCurrent)

    ## task-space:
    poseOfTipOfEndEffector = methods.GeneralizedPoseOfObj(loaded_model, qCurrent)
    velOfTipOfEndEffector = \
        methods.CalcGeneralizedVelOfObject(loaded_model, qCurrent, qDotCurrent)

    ## control acceleration of end-effector in task-space:
    accelDes = accelDes + kd_a * (velDes - velOfTipOfEndEffector) + \
                          kp_a * (poseDes - poseOfTipOfEndEffector)

    dJdq = methods.CalcdJdq(loaded_model, qCurrent, qDotCurrent, qDDotCurrent)  # JDot_a*qDot_a(1*6)

    qDDotDes = pinv(jac).dot(accelDes - dJdq).flatten()  # equ(21)

    qDes, qDotDes = Traj_Estimate(qDDotDes)

    return qDes, qDotDes, qDDotDes


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

    ## joint-space of left arm:
    qCurrent[0] = q[4]  # arm_zero
    qCurrent[1] = q[1]  # arm_one
    qCurrent[2] = q[3]  # arm_two
    qCurrent[3] = q[2]  # arm_three
    qCurrent[4] = q[0]  # arm_four
    qCurrent[5] = q[5]  # hand

    ## velocity of left arm:
    qDotCurrent[0] = qDot[4]  # arm_zero
    qDotCurrent[1] = qDot[1]  # arm_one
    qDotCurrent[2] = qDot[3]  # arm_two
    qDotCurrent[3] = qDot[2]  # arm_three
    qDotCurrent[4] = qDot[0]  # arm_four
    qDotCurrent[5] = qDot[5]  # hand



    x_des_t, xdot_des_t, xddot_des_t, time_prime = ChooseRef(time_)

    ## trajectory generation of end-effector: (task-space)
    xDes_now = np.array([x_des_t[i](time_prime) for i in rlx]).flatten()
    xDotDes_now = np.array([xdot_des_t[i](time_prime) for i in rlx]).flatten()
    xDDotDes_now = np.array([xddot_des_t[i](time_prime) for i in rlx]).flatten()

    ## detemine desired states of robot in joint-space:
    jointPose, jointVel, jointAccel = Task2Joint_(qCurrent, qDotCurrent, qDDotCurrent,
                                                  xDes_now, xDotDes_now, xDDotDes_now)

    jointTau = methods.InverseDynamic(loaded_model, jointPose, jointVel, jointAccel)

    PubTorqueToGazebo(jointTau)

    # methods.WriteToCSV(jointTau, time_)
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
        traj_plan(0, t_end/3, initPoseVelAccelOfObj_traj_1, finalPoseVelAccelOfObj_traj_1)

    ## Second trajectory:
    desPose_traj_2, desVel_traj_2, desAccel_traj_2 = \
        traj_plan(0, t_end/3, initPoseVelAccelOfObj_traj_2, finalPoseVelAccelOfObj_traj_2)

    ## Third trajectory:
    desPose_traj_3, desVel_traj_3, desAccel_traj_3 = \
        traj_plan(0, t_end/3, initPoseVelAccelOfObj_traj_3, finalPoseVelAccelOfObj_traj_3)

    rlx = range(len(desPose_traj_1))  # range(0, 6) --> 0, 1, 2, 3, 4, 5

    try:
        startTime = time.time()
        rospy.Subscriber("/arm/joint_states", JointState, JointStatesCallback)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
