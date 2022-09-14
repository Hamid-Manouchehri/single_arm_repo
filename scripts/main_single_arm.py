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
import rbdl_methods as RBDL  # TODO: change the file name if necessary.
####


time_ = 0
dt = 0.01
t_end = 5
g0 = 9.81
plotLegend = ''
writeHeaderOnceFlag = True
finiteTimeSimFlag = True  # TODO: True: simulate to 't_end', Flase: Infinite time

workspaceDof = 6  # TODO
singleArmDof = 6  # TODO

min_lambda = False  # TODO
min_tangent = False  # TODO

kp_grav = 3

kp_a = 2
kd_a = kp_a/5

kp_tau = 0
kd_tau = 0

startTime = 0
"""
(x(m), y(m), z(m), roll(rad), pitch(rad), yaw(rad)):
Note: any modification to trajectory need modification of 'LABEL_1',
function of 'ChooseRef' and of course the following trajecory definitions:
"""
desiredInitialStateOfObj_traj_1 = np.array([0., 0.9, 0.])
# desiredFinalStateOfObj_traj_1 = np.array([-.5, .5, +np.pi/3])
desiredFinalStateOfObj_traj_1 = np.array([0., .5, 0.])

desiredInitialStateOfObj_traj_2 = desiredFinalStateOfObj_traj_1
# desiredFinalStateOfObj_traj_2 = np.array([.5, .5, -np.pi/3])
desiredFinalStateOfObj_traj_2 = np.array([0., .9, 0.])

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
q = [[0.]*2*singleArmDof]
q = np.hstack((q, q_obj))

q_des = [q[-1][:2*singleArmDof]]

qctrl_des_prev = q_des[-1][:2*singleArmDof]
dqctrl_des_prev = np.zeros(2*singleArmDof)  # TODO: we usually start from rest


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

def QRDecompose(J):

    JT = J.T
    m, n = JT.shape

    if m == 0 or n == 0:
        raise TypeError(
            'Try to calculate QR decomposition, while there is no contact!')

    qr_Q, qr_R = qr(JT)

    qr_R = qr_R[:n, :]

    return qr_Q, qr_R


def CalcPqr(J, Su):
    qr_Q, qr_R = QRDecompose(J)
    return np.dot(Su, qr_Q.T), qr_Q, qr_R


def Task2Joint(J_a, J_b, J_oa, J_ob, J_rel, dJ_adq_a, dJ_bdq_b, r_o_a, r_o_b,
               x_o, dx_o, X_des, Xdot_des, Xddot_des):
    """Calculate equ(21), qDDot_des."""

    dth_o = dx_o[2]  # angular velocity of obj in base frame (rad/s)

    invJ_ob = inv(J_ob)
    J_A = J_rel

    J_B = np.dot(invJ_ob, J_b)  # equ(5) / qDot

    # GDot_oa*qDot_o: (1*3)
    dJ_oadz_o = np.cross([0, 0, dth_o], np.cross(r_o_a, [0, 0, dth_o]))
    # GDot_ob*GInv_ob*J_b*qDot_o: (1*3)
    dJ_obdz_o = np.cross([0, 0, dth_o], np.cross(r_o_b, [0, 0, dth_o]))

    # derivative of equ(7) * qDot:(1*3)
    dJ_A_dq = dJ_adq_a - dJ_oadz_o - np.dot(J_oa, np.dot(invJ_ob,
                                                         dJ_bdq_b - dJ_obdz_o))

    # an element of equ(21):(1*3)
    dJ_B_dq = np.dot(invJ_ob, dJ_bdq_b - dJ_obdz_o)

    Xddot_des_A = np.zeros(workspaceDof)

    # below equ(21): generalized acceleration of object
    Xddot_des_B = Xddot_des + kd_a*(Xdot_des - dx_o) + kp_a*(X_des - x_o)  # a(1*3)

    Xddot_des_A = np.vstack((Xddot_des_A.reshape(workspaceDof, 1), Xddot_des_B.reshape(workspaceDof, 1)))

    J_A = np.vstack((J_A, J_B))
    dJ_A_dq = np.vstack((dJ_A_dq.reshape(workspaceDof, 1), dJ_B_dq.reshape(workspaceDof, 1)))
    # print(J_A)

    J_B = np.array([])
    qddot_des = np.dot(pinv(J_A), Xddot_des_A - dJ_A_dq).flatten()  # equ(21)
    # print(qddot_des)

    q_des, qdot_des = Traj_Estimate(qddot_des)

    Xddot_des_after = Xddot_des_B
    dJ_obdz_o_des = dJ_obdz_o

    return q_des, qdot_des, qddot_des, Xddot_des_after, dJ_obdz_o_des


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

def IntForceParam_mine(J_a, J_b, J_oa, J_ob, S, Mqh, theta, W_lam_i=None):
    """Minimizing constaint force."""
    J = np.vstack((J_a, J_b))  # (6*6)
    k, n = np.shape(J)
    Sc = np.hstack((np.eye(k), np.zeros((k, n - k))))

    Q, RR = qr(J.T)  # (6*6), (6*6)

    R = RR[:k, :k]

    if W_lam_i is None:
        W_lam_i = np.eye(R.shape[0])  # I(6*6)

    if min_tangent:
        WW = np.diag([1., 10, 2])
        Ra = Rotation(theta).T
        Rb = Rotation(theta).T

        auxa = np.dot(Ra.T, np.dot(WW, Ra))
        auxb = np.dot(Rb.T, np.dot(WW, Rb))

        W_lam_i[:3, :3] = auxa
        W_lam_i[3:, 3:] = auxb

    W_lam = W_lam_i  # (6*6)

    # below equ(15)(6*6):
    Wc = np.dot(Q, np.dot(Sc.T, np.dot(inv(R.T),
                                       np.dot(W_lam,
                                       np.dot(inv(R), np.dot(Sc, Q.T))))))

    # equ(15):
    W = np.dot(S, np.dot(Wc, S.T))  # (6*6)
    tau_0 = np.dot(S, np.dot(Wc.T, Mqh))  # (6*6)

    return W, tau_0


def Rotation(t):  # Rotation around Z-axis of joint
    return np.array([np.cos(t), -np.sin(t), 0,
                     np.sin(t), np.cos(t), 0,
                     0, 0, 1]).reshape(3, 3)


def WriteToCSV(data, legendList=None, t=None):
    """
    Write data to the CSV file to have a live plot by reading the file ...
    simulataneously.

    Pass 'data' as a list of  data and 'legendList' as a list of string type,
    legend for each value of the plot.
    Note: 'legendList' and 't' are arbitrary arguments.
    """
    global plotLegend, writeHeaderOnceFlag

    plotLegend = legendList

    if t is None:
        ## to set the time if it is necessary.
        t = time_

    with open(pathToCSVFile + CSVFileName_plot_data, 'a', newline='') as \
            file:
        writer = csv.writer(file)

        if writeHeaderOnceFlag is True and legendList is not None:
            ## Add header to the CSV file.
            writer.writerow(np.hstack(['time', legendList]))
            writeHeaderOnceFlag = False

        writer.writerow(np.hstack([t, data]))  # the first element is time var.


def InverseDynamics(q, dq, ddq, X_des, Xdot_des, Xddot_des,
                    W=None, tau_0=None, kp=None, kd=None):

    JJ_a = RBDL.jc_right(loaded_model, q)
    JJ_b = RBDL.jc_left(loaded_model, q)
    # print(np.round(JJ_a, 2), '\n')

    J_a = np.hstack((JJ_a, np.zeros((workspaceDof, singleArmDof))))  # (6*6)
    J_b = np.hstack((np.zeros((workspaceDof, singleArmDof)), JJ_b))  # (6*6)

    dJ_adq_a = RBDL.CalcdJdq_r(loaded_model, q, dq, ddq)  # JDot_a*qDot_a(1*3)
    dJ_bdq_b = RBDL.CalcdJdq_l(loaded_model, q, dq, ddq)  # JDot_b*qDot_b(1*3)
    # print(np.round(dJ_adq_a, 2), '\n')

    x_a = RBDL.pose_end_r(loaded_model, q)
    x_b = RBDL.pose_end_l(loaded_model, q)
    # print(np.round(x_a, 4), '\n')

    X_o = RBDL.GeneralizedPoseOfObj(loaded_model, q)
    # print(np.round(X_o, 3))

    # generalized postion of object with repect to 'hand a' tip in base frame:
    r_o_a = X_o - x_a
    r_o_b = X_o - x_b

    J_oa = np.array([[1, 0, r_o_a[1]], [0, 1, - r_o_a[0]], [0, 0, 1]])  # G_oa
    J_ob = np.array([[1, 0, r_o_b[1]], [0, 1, - r_o_b[0]], [0, 0, 1]])  # G_ob

    # J_g: equ(7)
    J_rel = J_a - np.dot(J_oa, np.dot(inv(J_ob), J_b))  # (3*12)

    dX_o = RBDL.CalcGeneralizedVelOfObject(loaded_model, q, ddq)

    # Plan Joint-space trajectories
    q_des, qdot_des, qddot_des, xddot_des_after, dJ_obdz_o = \
        Task2Joint(J_a, J_b, J_oa, J_ob, J_rel, dJ_adq_a, dJ_bdq_b, r_o_a,
                   r_o_b, X_o, dX_o, X_des, Xdot_des, Xddot_des)

    M = RBDL.CalcM(loaded_model, q)
    h = RBDL.CalcH(loaded_model, q, ddq)
    # print(np.round(M, 2))

    Mqh = np.dot(M, qddot_des) + h  # left side of equ(8)

    dJ_bdq_b_des = dJ_bdq_b
    dJ_obdz_o_des = dJ_obdz_o

    qddot_des_bar = qddot_des  # (1*6)

    # below_1 equ(8):
    M_bar = M + np.dot(J_b.T, np.dot(inv(J_ob.T),
                                     np.dot(M_o, np.dot(inv(J_ob), J_b))))

    # below_2 equ(8):
    C_barq_des = np.dot(J_b.T, np.dot(inv(J_ob.T), np.dot(M_o,
                                                          np.dot(inv(J_ob), dJ_bdq_b_des - dJ_obdz_o_des))))

    h_bar = h.flatten() + np.dot(J_b.T, np.dot(inv(J_ob.T), h_o)).flatten()

    # left side of equ(8), a (1,6) vector:
    Mqh_bar_des = np.dot(M_bar, qddot_des_bar) + C_barq_des + h_bar

    S = np.eye(2*singleArmDof)

    # k: #independent contact constraints (3), n: #joints (6)
    k, n = J_rel.shape

    # Sc = np.hstack((np.eye(k), np.zeros((k, n - k))))
    Su = np.hstack((np.zeros((n - k, k)), np.eye(n - k)))  # below equ(11)

    # equ(11):
    P, qr_Q, qr_R = CalcPqr(J_rel, Su)  # (3*6), (6*6), (3*3)

    if W is None:
        W = np.eye(S.shape[0])  # (6*6)

    if tau_0 is None:
        tau_0 = np.zeros(S.shape[0])  # (1*6)

    if min_lambda:
        # determnie optimal 'W(6*6)' and 'tau_0(1*6)' for the inv dynamics
        # equ(10):
        W, tau_0 = IntForceParam_mine(J_a, J_b, J_oa, J_ob, S, Mqh, X_o[2])

    w_m_s = np.linalg.matrix_power(sqrtm(W), -1)
    aux = pinv(np.dot(P, np.dot(S.T, w_m_s)))
    winv = np.dot(w_m_s, aux)

    invw = pinv(W)  # (12*12)
    aux3 = np.dot(np.eye(S.shape[0]) - np.dot(winv, np.dot(P, S.T)), invw)

    # equ(10):
    invdyn = np.dot(winv, np.dot(P, Mqh_bar_des)).flatten() + np.dot(aux3, tau_0)

    # J_prime = J_rel[:, [3, 4, 5]]
    # constraint/contact force equ(12):
    # Lambda_a = np.dot(inv(J_prime.T), Mqh_bar_des[[3, 4, 5]]
    #                   - np.dot(S.T, invdyn)[[3, 4, 5]])

    # kp, kd = 0, S = I(6*6):
    desiredTorque = np.dot(S.T, invdyn) + 0*Mqh_bar_des + \
        kp_tau * (q_des - q) + kd_tau * (qdot_des - dq)

    return desiredTorque


def ChooseRef(time):
    if time <= t_end/3:
        out = [desPose_traj_1, desVel_traj_1, desAccel_traj_1, time]

    elif time <= 2*t_end/3:
        out = [desPose_traj_2, desVel_traj_2, desAccel_traj_2, time - t_end/3]

    else:
        out = [desPose_traj_3, desVel_traj_3, desAccel_traj_3, time - 2*t_end/3]

    return out


############### Constants and Desired ################
######################################################
######################################################
#####################################################
############## Running Simulation #################6#

def CalcTau(q, dq, ddq):

    x_des_t, xdot_des_t, xddot_des_t, time_prime = ChooseRef(time_)

    x_des_now = np.array([x_des_t[i](time_prime) for i in rlx]).flatten()
    xdot_des_now = np.array([xdot_des_t[i](time_prime) for i in rlx]).flatten()
    xddot_des_now = \
        np.array([xddot_des_t[i](time_prime) for i in rlx]).flatten()

    print(xdot_des_now)

    tau = InverseDynamics(q, dq, ddq, x_des_now, xdot_des_now, xddot_des_now)

    return tau


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

    q_rbdl = np.zeros(singleArmDof)
    qDot_rbdl = np.zeros(singleArmDof)
    qDDot_rbdl = np.zeros(singleArmDof)

    ## joint-space of left arm:
    q_rbdl[0] = q[4]  # arm_zero
    q_rbdl[1] = q[1]  # arm_one
    q_rbdl[2] = q[3]  # arm_two
    q_rbdl[3] = q[2]  # arm_three
    q_rbdl[4] = q[0]  # arm_four
    q_rbdl[5] = q[5]  # hand

    ## velocity of left arm:
    qDot_rbdl[0] = qDot[4]  # arm_zero
    qDot_rbdl[1] = qDot[1]  # arm_one
    qDot_rbdl[2] = qDot[3]  # arm_two
    qDot_rbdl[3] = qDot[2]  # arm_three
    qDot_rbdl[4] = qDot[0]  # arm_four
    qDot_rbdl[5] = qDot[5]  # hand


    # if (time.time() - startTime) > 4:

    # desiredTauVec = CalcTau(q_rbdl, qDot_rbdl, qDDot_rbdl)
    tau = RBDL.InverseDynamic(loaded_model, q_rbdl, qDot_rbdl, qDDot_rbdl)
    tau[0] = tau[0] + kp_grav * (0. - q_rbdl[0])  # shoulder
    PubTorqueToGazebo(tau)

    time_ += dt

    # else:
    #     tau = RBDL.InverseDynamic(loaded_model, q_rbdl, qDot_rbdl, qDDot_rbdl)
    #     tau[0] = tau[0] + kp_grav * (0. - q_rbdl[0])  # shoulder
    #     PubTorqueToGazebo(tau)


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
