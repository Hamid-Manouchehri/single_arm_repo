#!/usr/bin/env python3
"""
Author: Hamid Manouchehri
Year: 2022-2023

This module plot data saved in 'pathToCSVFile' directory.
'CSVFileName_livePlot_data' file will be created by the 'main_3D.py' or
'main_3D_linear_jac.py'.
"""
import os
from numpy import array
import csv
import rospy
import matplotlib.pyplot as plt
from sys import path
path.insert(1, '/home/rebel/bimanual_ws/src/single_arm_pkg/config/')  # TODO: Insert dir of config folder of the package.
import config

titleOfPlot = "< coriolis and centrifugal forces >"  # TODO
titleOfXaxis = "time (s)"  # TODO
titleOfYaxis = "h (Nm)"  # TODO

CSVFileName_livePlot_data = config.plot_csv_dic['CSVFileName']  # TODO: Change 'plot_csv_dic' to whatever is defined in 'config.py'
pathToCSVFile = config.plot_csv_dic['CSVFileDirectory']  # TODO: Change 'plot_csv_dic' to whatever is defined in 'config.py'


def ReadCSV():
    """Read whole data from the CSV file."""
    csvData = []
    with open(pathToCSVFile + CSVFileName_livePlot_data, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            csvData.append(row)
        return csvData



def main():
    """Compute main routine."""
    livePlotVal_x = []
    livePlotVal_y = []

    if os.path.isfile(pathToCSVFile + CSVFileName_livePlot_data) is True:
        csvFileData = ReadCSV()  # string.

        plotLegend = csvFileData[0][1:]

        numOfRows, numOfElements = array(csvFileData).shape
        # (numOfRows * numOfElements):
        Data = array([[0.]*numOfElements]*(numOfRows-1))

        for i in range(numOfRows-1):
            for j in range(numOfElements):
                Data[i][j] = (float(csvFileData[i+1][j]))  # cast str to float
            livePlotVal_x.append(Data[i][0])
            livePlotVal_y.append(Data[i][1:])

        plt.figure(titleOfPlot)  # title of the figure
        plt.plot(livePlotVal_x, livePlotVal_y)
        plt.xlabel(titleOfXaxis)
        plt.ylabel(titleOfYaxis)
        if csvFileData[0][0] == 'time':  # Check if header (as string) is added
            plt.legend(plotLegend)
        plt.grid()
        plt.tight_layout()
        plt.show()

    else:
        print('There is no CSV source file in the defined directory!')



if __name__ == '__main__':

    try:
        main()

    except rospy.ROSInterruptException:
        pass
