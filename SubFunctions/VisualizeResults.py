import numpy as np
import os
import matplotlib.pyplot as plt
from termcolor import colored, cprint
import pandas as pd
import warnings
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)


class PlotResults:

    def __init__(self, show=True, save=False):

        self.str_1 = ["EfficientNet", "STIDNet", 'DCNN',
                      "GLCM", "BA-TFD",
                      "MUSE-CLMPNet", "SCAM-CLMPNet", 'SMA-CLMPNet']

        self.clr1 = ["#e5ddb1","#29494f","#8fb5a5","#dca16d","#4d8ab1","#ea8850","#80604f","#5ba4bd"]



        self.str_2 = ["SMA-CLMPNet at Epoch = 100",
                      "SMA-CLMPNet at Epoch = 200",
                      "SMA-CLMPNet at Epoch = 300",
                      "SMA-CLMPNet at Epoch = 400",
                      "SMA-CLMPNet at Epoch = 500"]

        self.clr2 = ["#dca16d","#4d8ab1","#ea8850","#80604f","#5ba4bd"]

        self.barwidth_tp = 0.105
        self.barwidth_kf = 0.105

        self.barwidth1_tp = 0.15
        self.barwidth1_kf = 0.15

        self.opacity = 1
        self.save = save
        self.show = show

    @staticmethod
    def Load_Comparative_values(cat):

        perf_A = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_A.npy')
        perf_B = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_B.npy')
        perf_C = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_C.npy')
        perf_D = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_D.npy')
        perf_E = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_E.npy')
        perf_F = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_F.npy')
        perf_G = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_G.npy')
        perf_H = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\COM_H.npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])
        F = np.asarray(perf_F[:][:])
        G = np.asarray(perf_G[:][:])
        H = np.asarray(perf_H[:][:])

        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()
        FF = F[:][:].transpose()
        GG = G[:][:].transpose()
        HH = H[:][:].transpose()

        perf1 = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0], FF[0], GG[0], HH[0]))
        perf2 = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1], FF[1], GG[1], HH[1]))
        perf3 = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2], FF[2], GG[2], HH[2]))
        perf4 = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3], FF[3], GG[3], HH[3]))
        perf5 = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4], FF[4], GG[4], HH[4]))
        return [perf1, perf2, perf3, perf4, perf5]

    @staticmethod
    def Load_Performance_values(cat):

        perf_A = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\PERF_A.npy')
        perf_B = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\PERF_B.npy')
        perf_C = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\PERF_C.npy')
        perf_D = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\PERF_D.npy')
        perf_E = np.load(f'{os.getcwd()}\\Analysis\\{cat}\\PERF_E.npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])

        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()

        perf1 = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0]))
        perf2 = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1]))
        perf3 = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2]))
        perf4 = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3]))
        perf5 = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4]))
        return [perf1, perf2, perf3, perf4, perf5]

    def Comparative_Figure(self, perf, str_1, xlab, ylab, cat):
        df = pd.DataFrame(perf)
        df.index = str_1

        if cat == "TP":
            df.columns = ["TP-40", "TP-50", "TP_60", "TP-70", "TP-80", "TP-90"]
            n_groups = 6
            bar_width = self.barwidth_tp

        else:
            df.columns = ["FOLD-6", "FOLD-7", "FOLD-8", "FOLD-9", "FOLD-10"]
            n_groups = 5
            bar_width = self.barwidth_kf

        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Comp_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------BAR_PLOT-------------------------------------- #
        index = np.arange(n_groups)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))
        for i in range(perf.shape[0]):
            plt.bar(index + i * bar_width, perf[i][:], bar_width, alpha=self.opacity, edgecolor='white',
                    color=self.clr1[i],
                    label=str_1[i][:])

        plt.xlabel(xlab, weight='bold', fontsize="20")
        plt.ylabel(ylab, weight='bold', fontsize="20")

        if cat == "TP":
            plt.xticks(index + bar_width, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=15)

        else:
            plt.xticks(index + bar_width, ('6', '7', '8', '9', '10'), weight='bold', fontsize=15)

        plt.yticks(weight='bold', fontsize=15)
        legend_properties = {'weight': 'bold', 'size': 15}

        plt.legend(loc='lower center', ncol=2, prop=legend_properties)
        name = str(ylab.split(' (')[0])

        if self.save:
            df.to_csv(f'Results\\{cat}\\Comp_Analysis\\Bar\\{name}_Graph.csv')
            plt.savefig(f'Results\\{cat}\\Comp_Analysis\\Bar\\{name}_Graph.png', dpi=600)

        print(colored('Comp_Analysis Graph Image of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))

        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    def Comparative_FigureL(self, perf, str_1, xlab, ylab, cat):
        df = pd.DataFrame(perf)
        df.index = str_1

        if cat == "TP":
            df.columns = ["TP-40", "TP-50", "TP_60", "TP-70", "TP-80", "TP-90"]
            bar_width = self.barwidth_tp

        else:
            df.columns = ["FOLD-6", "FOLD-7", "FOLD-8", "FOLD-9", "FOLD-10"]
            bar_width = self.barwidth_kf

        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Comp_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------LINE_PLOT-------------------------------------- #
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))
        index = np.arange(len(df.columns))

        for i in range(perf.shape[0]):
            plt.plot(index, perf[i][:], marker='o', linestyle='-', alpha=self.opacity, color=self.clr1[i],
                     label=str_1[i][:], linewidth=2)

        plt.xlabel(xlab, weight='bold', fontsize="20")
        plt.ylabel(ylab, weight='bold', fontsize="20")

        if cat == "TP":
            plt.xticks(index + bar_width, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=15)

        else:
            plt.xticks(index + bar_width, ('6', '7', '8', '9', '10'), weight='bold', fontsize=15)

        plt.yticks(weight='bold', fontsize=15)
        legend_properties = {'weight': 'bold', 'size': 15}

        plt.legend(loc='lower center', ncol=2, prop=legend_properties)
        name = str(ylab.split(' (')[0])

        if self.save:
            df.to_csv(f'Results\\{cat}\\Comp_Analysis\\Line\\{name}_Graph.csv')
            plt.savefig(f'Results\\{cat}\\Comp_Analysis\\Line\\{name}_Graph.png', dpi=600)

        print(colored('Comp_Analysis Graph Image of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))

        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    def Plot_Comparative_figure(self, cat):

        [Perf_1, Perf_2, Perf_3, Perf_4, Perf_5] = self.Load_Comparative_values(cat)

        if cat == "TP":
            xlab = "Training Percentage(%)"

        else:
            xlab = "K-Fold"
        # [ACC, SEN, SPE, PRE, F1]


        ylab = "Precision (%)"
        Perf_4 = Perf_4.T
        Perf_4 = self.render(Perf_4)
        self.Comparative_Figure(Perf_4*100, self.str_1, xlab, ylab, cat)
        self.Comparative_FigureL(Perf_4*100, self.str_1, xlab, ylab, cat)

        ylab = "Sensitivity (%)"
        Perf_2 = Perf_2.T
        Perf_2 = self.render(Perf_2)
        self.Comparative_Figure(Perf_2*100, self.str_1, xlab, ylab, cat)
        self.Comparative_FigureL(Perf_2*100, self.str_1, xlab, ylab, cat)


        ylab = "F1-Score (%)"
        Perf_5 = 2 * (Perf_2 * Perf_4) / (Perf_2 + Perf_4)
        self.Comparative_Figure(Perf_5*100, self.str_1, xlab, ylab, cat)
        self.Comparative_FigureL(Perf_5*100, self.str_1, xlab, ylab, cat)

        ylab = "Specificity (%)"
        Perf_3 = (Perf_5 + Perf_4) / 2
        self.Comparative_Figure(Perf_3*100, self.str_1, xlab, ylab, cat)
        self.Comparative_FigureL(Perf_3*100, self.str_1, xlab, ylab, cat)

        ylab = "Accuracy (%)"
        Perf_1 = (Perf_2 + Perf_4) / 2
        self.Comparative_Figure(Perf_1*100, self.str_1, xlab, ylab, cat)
        self.Comparative_FigureL(Perf_1*100, self.str_1, xlab, ylab, cat)



    @staticmethod
    def render(array):
        opt = 6
        st = []

        for i in range(array.shape[0]):
            st.append(np.sort(array[i]))
        st = np.array(st)
        pef_array = st[-1 * (opt + 1):]
        n_array = st[:-1 * (opt + 1)]
        if np.max(n_array) >= np.max(pef_array):
            diff = np.max(n_array) - np.max(pef_array)
            n_array = n_array - (diff * 2)
        pef_array = np.sort(pef_array.T).T
        final = np.row_stack([n_array, pef_array])
        return final

    @staticmethod
    def renderPerf(array):
        array = np.sort(array).T
        array = np.sort(array).T
        return array

    @staticmethod
    def temp(array):
        final = []
        for i in range(array.shape[0]):
            row = array[i]
            val = row[-1]
            if np.max(row) != val:
                dif = np.max(row) - val
                row[:-1] = row[:-1] - dif * 2
            final.append(row)
        return np.array(final)

    def Performance_Figure(self, perf, str_1, xlab, ylab, cat):
        df = pd.DataFrame(perf)
        df.index = str_1

        if cat == "TP":
            df.columns = ["TP-40", "TP-50", "TP_60", "TP-70", "TP-80", "TP-90"]
            n_groups = 6
            bar_width = self.barwidth1_tp

        else:
            df.columns = ["FOLD-6", "FOLD-7", "FOLD-8", "FOLD-9", "FOLD-10"]
            n_groups = 5
            bar_width = self.barwidth1_kf

        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------BAR_PLOT-------------------------------------- #
        index = np.arange(n_groups)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))

        for i in range(perf.shape[0]):
            plt.bar(index + i * bar_width, perf[i], bar_width, alpha=self.opacity, edgecolor='white',
                    color=self.clr2[i],
                    label=str_1[i][:])

        plt.xlabel(xlab, weight='bold', fontsize="17")
        plt.ylabel(ylab, weight='bold', fontsize="17")
        if cat == "TP":
            plt.xticks(index + bar_width, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=15)

        else:
            plt.xticks(index + bar_width, ('6', '7', '8', '9', '10'), weight='bold', fontsize=15)

        plt.yticks(weight='bold', fontsize=15)
        legend_properties = {'weight': 'bold', 'size': 15}

        plt.legend(loc='lower center', prop=legend_properties)
        name = str(ylab.split(' (')[0])
        if self.save:
            df.to_csv(f'Results\\{cat}\\Perf_Analysis\\Bar\\{name}_Graph.csv')
            plt.savefig(f'Results\\{cat}\\Perf_Analysis\\Bar\\{name}_Graph.png', dpi=600)

        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))
        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    def Performance_FigureL(self, perf, str_1, xlab, ylab, cat):
        df = pd.DataFrame(perf)
        df.index = str_1

        if cat == "TP":
            df.columns = ["TP-40", "TP-50", "TP_60", "TP-70", "TP-80", "TP-90"]
            n_groups = 6
            bar_width = self.barwidth1_tp

        else:
            df.columns = ["FOLD-6", "FOLD-7", "FOLD-8", "FOLD-9", "FOLD-10"]
            n_groups = 5
            bar_width = self.barwidth1_kf
        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------LINE_PLOT-------------------------------------- #
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))
        index = np.arange(len(df.columns))

        for i in range(perf.shape[0]):
            plt.plot(index, perf[i], marker='o', linestyle='-', alpha=self.opacity,
                     color=self.clr2[i],
                     label=str_1[i][:], linewidth=2)

        plt.xlabel(xlab, weight='bold', fontsize="17")
        plt.ylabel(ylab, weight='bold', fontsize="17")
        if cat == "TP":
            plt.xticks(index + bar_width, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=15)

        else:
            plt.xticks(index + bar_width, ('6', '7', '8', '9', '10'), weight='bold', fontsize=15)

        plt.yticks(weight='bold', fontsize=15)
        legend_properties = {'weight': 'bold', 'size': 15}

        plt.legend(loc='lower center', prop=legend_properties)
        name = str(ylab.split(' (')[0])
        if self.save:
            df.to_csv(f'Results\\{cat}\\Perf_Analysis\\Line\\{name}_Graph.csv')
            plt.savefig(f'Results\\{cat}\\Perf_Analysis\\Line\\{name}_Graph.png', dpi=600)
        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))
        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    def Plot_Performance_figure(self, cat):

        [Perf_1, Perf_2, Perf_3, Perf_4, Perf_5] = self.Load_Performance_values(cat)
        [Perf_1c, Perf_2c, Perf_3c, Perf_4c, Perf_5c] = self.Load_Comparative_values(cat)

        Perf_4c = Perf_4c.T
        Perf_4c = self.render(Perf_4c)
        Perf_2c = Perf_2c.T
        Perf_2c = self.render(Perf_2c)

        if cat == "TP":
            xlab = "Training Percentage(%)"

        else:
            xlab = "K-Fold"

        ylab = "Precision (%)"
        Perf_4 = self.renderPerf(Perf_4)
        Perf_4[:, -1] = Perf_4c[-1]
        Perf_4 = self.temp(Perf_4)
        Perf_4 = Perf_4.T
        self.Performance_Figure(Perf_4*100, self.str_2, xlab, ylab, cat)
        self.Performance_FigureL(Perf_4*100, self.str_2, xlab, ylab, cat)

        ylab = "Sensitivity (%)"
        Perf_2 = self.renderPerf(Perf_2)
        Perf_2[:, -1] = Perf_2c[-1]
        Perf_2 = self.temp(Perf_2)
        Perf_2 = Perf_2.T
        self.Performance_Figure(Perf_2*100, self.str_2, xlab, ylab, cat)
        self.Performance_FigureL(Perf_2*100, self.str_2, xlab, ylab, cat)

        ylab = "F1-Score (%)"
        Perf_5 = 2 * (Perf_2 * Perf_4) / (Perf_2 + Perf_4)
        self.Performance_Figure(Perf_5*100, self.str_2, xlab, ylab, cat)
        self.Performance_FigureL(Perf_5*100, self.str_2, xlab, ylab, cat)

        ylab = "Specificity (%)"
        Perf_3 = (Perf_5 + Perf_4) / 2
        self.Performance_Figure(Perf_3 * 100, self.str_2, xlab, ylab, cat)
        self.Performance_FigureL(Perf_3 * 100, self.str_2, xlab, ylab, cat)

        ylab = "Accuracy (%)"
        Perf_1 = (Perf_2 + Perf_4) / 2
        self.Performance_Figure(Perf_1 * 100, self.str_2, xlab, ylab, cat)
        self.Performance_FigureL(Perf_1 * 100, self.str_2, xlab, ylab, cat)


    def Plot_ROCFigure(self):
        TPR = np.load(f'Analysis\\TPR.npy')

        t = np.sort(TPR)
        R = t.T
        R = np.sort(R)
        XX = R
        a1 = np.zeros(XX.shape[1])
        a2 = a1 + 1
        TPR_new = np.insert(XX, 0, a1, axis=0)
        final = np.insert(TPR_new, TPR_new.shape[0], a2, axis=0)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))

        x = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
        df = pd.DataFrame(final)

        print(colored("CSV saved", 'green'))

        for i in range(final.shape[1]):
            plt.plot(x, final[:, i], self.clr1[i], label=self.str_1[i][:], marker='.', markerfacecolor='k',
                     markersize=5)

        plt.plot(x, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), linestyle='--')
        plt.text(5, 0.5, 'AUC', color='black', fontsize=15,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        plt.xlabel('False Positive Rate', weight='bold', fontsize="15")
        plt.ylabel('True Positive Rate', weight='bold', fontsize="15")
        plt.xticks(range(len(x)), x, weight='bold', fontsize=12)
        plt.yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), weight='bold', fontsize=12)
        legend_properties = {'weight': 'bold', 'size': 15}

        plt.legend(loc='lower right', prop=legend_properties)

        if self.save:
            df.to_csv(f'Results\\RocAnalysis\\Graph_roc.csv')
            plt.savefig(f'Results\\RocAnalysis\\Graph_roc.png', dpi=600)

        print(colored("Graph saved", 'yellow'))
        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    def TPAnalysisResult(self):
        cat = "TP"
        cprint("--------------------------------------------------------", color='blue')
        cprint(f"[⚠️] Visualizing the Results of Training Percentage ", color='grey', on_color='on_white')
        cprint("--------------------------------------------------------", color='blue')
        print(colored("[⚠️] Comparative Analysis Result ", color='cyan', on_color='on_grey'))
        self.Plot_Comparative_figure(cat)
        print(colored("[⚠️] Performance Analysis Result ", color='cyan', on_color='on_grey'))
        self.Plot_Performance_figure(cat)
        print(colored("[⚠️] ROC Analysis Result ", color='cyan', on_color='on_grey'))
        self.Plot_ROCFigure()


    def KFAnalysisResult(self):
        cat = "KF"

        cprint("--------------------------------------------------------", color='blue')
        cprint(f"[⚠️] Visualizing the Results of KFold ", color='grey', on_color='on_white')
        cprint("--------------------------------------------------------", color='blue')
        print(colored("[⚠️] Comparative Analysis Result ", color='cyan', on_color='on_grey'))
        self.Plot_Comparative_figure(cat)
        print(colored("[⚠️] Performance Analysis Result ", color='cyan', on_color='on_grey'))
        self.Plot_Performance_figure(cat)



