from SubFunctions import *  # Import all functions from SubFunctions
from PySimpleGUI import popup_yes_no  # Import popup_yes_no from PySimpleGUI

# Display a popup to ask the user if they want complete execution
VVV = popup_yes_no("Do You want Complete Execution?")

if VVV == "Yes":
    # Read the dataset with complete execution
    data = ReadDataset(exec=True).read_data()

    # Perform TP (Training Percentage) Analysis
    tP = TPAnalysis(data)
    tP.ComparativeAnalysis()  # Comparative analysis of Training Percentage results
    tP.PerformanceAnalysis()  # Performance analysis of Training Percentage results
    tP.RocAnalysis()  # Performance analysis of Training Percentage results

    # Perform KF (K-Fold) Analysis
    kF = KFAnalysis(data)
    kF.ComparativeAnalysis()  # Comparative analysis using K-Fold validation
    kF.PerformanceAnalysis()  # Performance analysis using K-Fold validation

    # Plot the results of the analyses
    pL = PlotResults()
    pL.TPAnalysisResult()  # Plot analysis results
    pL.KFAnalysisResult()  # Plot K-Fold analysis results

else:
    # Plot the results without performing complete execution
    pL = PlotResults()
    pL.TPAnalysisResult()  # Plot analysis results
    pL.KFAnalysisResult()  # Plot K-Fold analysis results
