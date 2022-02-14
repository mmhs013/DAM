"""
Dynamic Adaptation Model (DAM)

Author: Md. Manjurul Husain Shourov
last edited: 12/08/2021
Website: https://github.com/mmhs013/DAM
"""

import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shutil import copyfile
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('DAM_GUI.ui', self)
        self.setWindowTitle('Dynamic Adaptation Model')
        self.show()

        self.RootDir = os.getcwd()
        self.DataDir = self.RootDir+"/DAM_Database/"
        self.StudyArea = pd.read_csv(self.DataDir + '/Study Area.csv')
        self.adp_need = pd.read_excel(self.DataDir + "/Hazards/Salinity/Constraints.xlsx",sheet_name="Adaptation_need",index_col="THACODE")
        self.actual_adp = pd.read_excel(self.DataDir + "/Hazards/Salinity/Actual_Adaptation.xlsx",sheet_name="Actual_adp",index_col="THACODE")
        self.int_input = pd.read_excel(self.DataDir + "/Hazards/Salinity/Initial_input.xlsx", sheet_name="Initial_input", index_col="THACODE")
        self.BN_wt = pd.read_excel(self.DataDir + "/Hazards/Salinity/Initial_input.xlsx", sheet_name="BN_Weight", index_col="Adaptation_Indicator")
        self.Autonomas_Weight = pd.read_excel(self.DataDir + "For_Risk.xlsx",sheet_name="Autonomas_Weight",index_col="Planned")
        
        self.adp_need = self.adp_need.round(2)
        self.actual_adp = self.actual_adp.round(2)

        self.ImView.setScaledContents(True)
        self.ImView.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.Table1.horizontalHeader().setStyleSheet("QHeaderView {font-weight: bold;}")
        self.Table1.horizontalHeader().setStretchLastSection(True)

        self.Table2.horizontalHeader().setStyleSheet("QHeaderView {font-weight: bold;}")
        self.Table2.horizontalHeader().setStretchLastSection(True)
        self.Table2.hide()
        self.Table2PBtn.hide()

        self.ZoneCombo.addItems(self.StudyArea.Zone.unique())
        self.ZoneName = "West"
        self.ZoneCombo.activated.connect(self.ZoneComboAction)

        self.HazardCombo.addItems([x[1] for x in os.walk(self.DataDir+"/Hazards")][0])
        self.HazardName = "Salinity"
        self.HazardCombo.activated.connect(self.HazardComboAction)

        self.LevelName = "Upazilla"
        self.UpazillaName = "Batiaghata"
        self.VillageName = "Amirpur"

        self.VillageCombo.hide()
        self.VillageLabel.hide()
        self.ZoneData = self.StudyArea[self.StudyArea.Zone == self.ZoneName]
        self.UpazillaCombo.addItems(self.ZoneData.Upazilla.unique())
        self.LevelCombo.activated.connect(self.LevelComboAction)

        self.UpazillaCombo.activated.connect(self.UpazillaComboAction)
        self.VillageCombo.activated.connect(self.VillageComboAction)

        self.MapPBtn.clicked.connect(lambda: self.save("",'SaveMap'))
        self.Table1PBtn.clicked.connect(lambda: self.save(self.old_tbl1Data,'SaveTable'))
        self.Table2PBtn.clicked.connect(lambda: self.save(self.tbl2Data,'SaveTable'))

        self.Table1.cellChanged.connect(lambda: self.cellchanged(self.Table1, 1))
        self.Table2.cellChanged.connect(lambda: self.cellchanged(self.Table2, 2))

        self.CalculatePButton.setText("Calculation")
        self.CalculatePButton.clicked.connect(self.Action)


    def ZoneComboAction(self):
        self.ZoneName = self.sender().currentText()
        self.CalculatePButton.setText("Calculation")


    def HazardComboAction(self):
        self.HazardName = self.sender().currentText()
        self.CalculatePButton.setText("Calculation")

    
    def LevelComboAction(self):
        self.LevelName = self.sender().currentText()
        self.ZoneData = self.StudyArea[self.StudyArea.Zone == self.ZoneName]
        self.CalculatePButton.setText("Calculation")

        if self.LevelName == 'Upazilla':
            self.VillageCombo.hide()
            self.VillageLabel.hide()
            self.Table2.hide()
            self.Table2PBtn.hide()

            self.UpazillaCombo.clear()
            self.UpazillaCombo.addItems(self.ZoneData.Upazilla.unique())
            self.UpazillaName = self.ZoneData.Upazilla.unique()[0]

        elif self.LevelName == 'Village':
            self.VillageCombo.show()
            self.VillageLabel.show()
            self.Table2.show()
            self.Table2PBtn.show()

            self.VillageCombo.clear()
            self.UpazillaData = self.ZoneData[self.ZoneData.Upazilla == self.UpazillaName]
            self.VillageCombo.addItems(self.UpazillaData.Village.unique())
            self.VillageName = self.UpazillaData.Village.unique()[0]


    def UpazillaComboAction(self):
        self.UpazillaName = self.sender().currentText()

        self.VillageCombo.clear()
        # self.ZoneData = self.StudyArea[self.StudyArea.Zone == self.ZoneName]
        self.UpazillaData = self.ZoneData[self.ZoneData.Upazilla == self.UpazillaName]
        self.VillageCombo.addItems(self.UpazillaData.Village.unique())
        self.VillageName = self.UpazillaData.Village.unique()[0]

        self.CalculatePButton.setText("Calculation")

    def VillageComboAction(self):
        self.VillageName = self.sender().currentText()

        self.CalculatePButton.setText("Calculation")


    def TableFillUp(self, Table, data, TableNo):
        if TableNo == 1:
            
            if self.LevelName == 'Upazilla':
                data.iloc[4:,1:] = data.iloc[4:,1:].astype(float).round(0).astype(int)
                fix_row = 4
            elif self.LevelName == 'Village':
                data.iloc[5:,1:] = data.iloc[5:,1:].astype(float).round(0).astype(int)
                fix_row = 4

        if TableNo == 2:
            data.iloc[1:,1] = data.iloc[1:,1].astype(float).round(0).astype(int)
            fix_row = 0  
        Table.setRowCount(0)
        Table.setColumnCount(data.shape[1])
        Table.setHorizontalHeaderLabels(data.iloc[0])

        for r in range(data.shape[0]-1):
            Table.insertRow(r)
            for c in range(data.shape[1]):
                item = QtWidgets.QTableWidgetItem(str(data.iloc[r+1, c]))

                if (c != data.shape[1]-1) or (r < fix_row):
                    item.setFlags( QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsEnabled )

                Table.setItem(r, c, item)
    
    def cellchanged(self, Table, TableNo):
        self.TableNo = TableNo
        self.colLoc = Table.currentColumn()
        self.rowLoc = Table.currentRow()
        self.CalculatePButton.setText("Re-Calculation")


    def JoinData(self, df):
        StudyArea_Up = self.StudyArea.groupby(self.StudyArea.THACODE).first()
        StudyArea_Up.drop(["Union", "Mouza", "Village","Weight"], axis=1, inplace=True)
        self.StudyArea_Up_Zone = StudyArea_Up[StudyArea_Up['Zone'] == "West"]

        return self.StudyArea_Up_Zone.join(df)


    def risk_calulation(self, actual_adp):
        wt_adp = (actual_adp * self.BN_wt.Weight)
        Risk = (((1 - wt_adp.sum(axis=1)) + self.int_input.Sensitivity) * self.int_input.Hazard * self.int_input.Exposure).to_frame()
        Risk_norm = (Risk - Risk.min()) / (Risk.max() - Risk.min()) *100
        Risk_norm.columns = ["Risk"]

        return Risk_norm

    
    def drawMap(self, riskData, title):
        gdf = gpd.read_file(self.DataDir +"/Shapefiles/DAM_Upazilla_WGS_1984.shp")
        gdf.THACODE = gdf.THACODE.astype(int)
        gdfz = gdf[gdf.Zone == self.ZoneName]
        gdfz = gdfz.loc[:,['THACODE','geometry']]
        self.joinData = gdfz.set_index('THACODE').join(riskData)

        ax = self.joinData.plot(alpha=0.8,column='Risk', edgecolor='black', cmap='RdYlGn_r', figsize=(7,10),
        legend=True, legend_kwds = {'label': 'Risk Level'} , vmin=0,vmax=100)

        margin_x = 0.005
        margin_y = 0.005
        xlim = ([self.joinData.total_bounds[0] - margin_x,  self.joinData.total_bounds[2]+ margin_x])
        ylim = ([self.joinData.total_bounds[1] - margin_y,  self.joinData.total_bounds[3]+ margin_y])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title,fontsize=14)

        for x, y, label in zip(self.joinData.centroid.x, self.joinData.centroid.y, self.joinData["Upazilla"]):
            ax.annotate(label, xy=(x, y),horizontalalignment='center',fontsize=9)

        plt.xlabel('Longitude',fontsize=10)
        plt.ylabel('Latitude',fontsize=10)
        plt.yticks(rotation=90)

        plt.savefig(self.DataDir + '/Temp/im.jpg',bbox_inches='tight',dpi=300)
        self.FilePath = self.DataDir + '/Temp/im.jpg'

        self.PixMap = QtGui.QPixmap(self.FilePath)
        self.ImView.setPixmap(self.PixMap)
        self.ImView.adjustSize()

    
    def save(self,tableData, save_opt):
        if save_opt == 'SaveTable':
            newfilePath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Warning", "", 
                         "CSV(*.csv);;All Files(*.*) ")
            
            if newfilePath != "": 
                tableData.to_csv(newfilePath, header=False, index=False)

        else:
            # selecting file path 
            newfilePath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "", 
                            "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
            
            # if file path is blank return back 
            if newfilePath == "": 
                return
            
            # saving canvas at desired path 
            copyfile(self.FilePath, newfilePath)


    def village_adp_calc(self, up_data, a):
        nn = len(up_data.Weight)
        init = np.zeros(nn)
        lower_b = np.zeros(nn)
        upper_b = np.ones(nn) * 100
        bound = tuple(zip(lower_b, upper_b))

        for nm in range(len(self.adp_name)):
            
            def vill_obj(x):
                s = up_data[self.adp_name[nm]].iloc[0]
                for i in range(nn):
                    s = s - x[i] * up_data.Weight[i]
                return s**2

            cons_eq = []
            for idx in range(nn):
                cons_eq.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx] - a[idx]})

            solution = minimize(vill_obj, init, method='SLSQP', bounds=bound, constraints=cons_eq)

            up_data.loc[:, self.adp_name[nm]] = solution.x
            
        return up_data



    def optimization(self, i, ind, adp_idn, adp_need):

        init = np.zeros(20)
        lower_b = np.zeros(20)
        lower_b[4:19] = np.ones(15) * 0.001

        upper_b = np.ones(20)
        upper_b[4:19] = self.adp_need.loc[i,:].to_numpy()
        upper_b[[10,14,17,19]] = np.ones(4)
        bound = tuple(zip(lower_b, upper_b))

        adp_upz = adp_need.loc[i,:].copy()

        a = [self.int_input.loc[i,:].Exposure * self.int_input.loc[i,:].Hazard,
            self.int_input.loc[i,:].Exposure,
            self.int_input.loc[i,:].Sensitivity,
            0.13 * self.int_input.loc[i,:].Exposure + 0.87 * self.int_input.loc[i,:].Sensitivity]
        a.extend(list(adp_upz))
        a.append(self.int_input.loc[i,:].Hazard)

        Objective = lambda x: x[0]*x[19]*((0.0548*x[1]+0.358*x[2]+0.587*x[3])-(0.077*x[4]+0.075*x[5]+0.05*x[6]+0.0588*x[7]+0.0569*x[8]+0.0283*x[9]
                            +0.0467*x[10]+0.0449*x[11]+0.0366*x[12]+0.0266*x[13]+0.0352*x[14]+0.06*x[15]+0.0523*x[16]+0.0487*x[17]+0.302*x[18]))

        ineq1 = lambda x: a[2] - (0.0548 * x[1] + 0.358 * x[2] + 0.587 * x[3])
        ineq2 = lambda x: 0.56 * x[5] + 0.44 * x[7]
        ineq3 = lambda x: 0.51 * x[4] + 0.49 * x[5]
        ineq4 = lambda x: 0.17 * x[4] + 0.16 * x[5] - 0.665 * x[18]
        ineq5 = lambda x: 0.138 * x[5] + 0.092 * x[6]  + 0.108 * x[7] + 0.1047 * x[8] + 0.556 * x[18]
        ineq6 = lambda x: 0.147 * x[8] + 0.073 * x[9] + 0.78 * x[18]
        ineq7 = lambda x: 0.725 * x[5] + 0.274 * x[9]
        ineq8 = lambda x: -0.1176 * x[11] *0.092 * x[14] + 0.79 * x[18]
        ineq9 = lambda x: 0.117 * x[4] + 0.113 * x[5] + 0.089 * x[7] + 0.086 * x[8] + 0.0556 * x[12] + 0.0794 * x[16]+ 0.458 * x[18]
        ineq10 = lambda x: 0.099 * x[4] + 0.096 * x[5] + 0.0644 * x[6] + 0.075 * x[7]  + 0.073 * x[8] + 0.058 * x[11] + 0.0341 * x[13] + 0.0451 * x[14] + 0.067 * x[16] + 0.387 * x[18]
        ineq11 = lambda x: 0.123 * x[4] + 0.119 * x[5] + 0.0933 * x[7] + 0.09 * x[8] + 0.0955 * x[15] + 0.479 * x[18]
        ineq12 = lambda x: 0.59 * x[5] + 0.41 * x[16]
        ineq13 = lambda x: 0.127 * x[4] + 0.123 * x[5] + 0.097 * x[7] + 0.093 * x[9] + 0.06 * x[13] + 0.5 * x[18]

        eq1 = lambda x: (x[0] * x[19]) - a[0]
        eq2 = lambda x: x[0] - a[1]
        eq3 = lambda x: 0.13 * x[0] + 0.379 * x[1] + 0.328 * x[2] + 0.504 * x[3] - a[3]
        eq4 = lambda x: x[10] - a[10]
        eq5 = lambda x: x[14] - a[14]
        eq6 = lambda x: x[17] - a[17]
        eq7 = lambda x: x[19] - a[19]


        cons_eq = [
            {'type': 'ineq', 'fun': ineq1},
            {'type': 'ineq', 'fun': ineq2},
            {'type': 'ineq', 'fun': ineq3},
            {'type': 'ineq', 'fun': ineq4},
            {'type': 'ineq', 'fun': ineq5},
            {'type': 'ineq', 'fun': ineq6},
            {'type': 'ineq', 'fun': ineq7},
            {'type': 'ineq', 'fun': ineq8},
            {'type': 'ineq', 'fun': ineq9},
            {'type': 'ineq', 'fun': ineq10},
            {'type': 'ineq', 'fun': ineq11},
            {'type': 'ineq', 'fun': ineq12},
            {'type': 'ineq', 'fun': ineq13},
            NonlinearConstraint(eq1, 0, 0),
            NonlinearConstraint(eq2, 0, 0),
            NonlinearConstraint(eq3, 0, 0),
            NonlinearConstraint(eq4, 0, 0),
            NonlinearConstraint(eq5, 0, 0),
            NonlinearConstraint(eq6, 0, 0),
            NonlinearConstraint(eq7, 0, 0),
        ]

        for idx in set(adp_idn):
            cons_eq.append({'type': 'eq', 'fun': lambda x, idx=idx: x[idx] - a[idx]})

        solution = minimize(Objective, init, method='trust-constr', bounds=bound, constraints=cons_eq)
        
        return np.round(solution.x[4:19], 2)


    def Action(self):
        sender = self.sender()

        # path = self.DataDir + '/Hazards/' + self.HazardName

        
        if sender.text() == 'Calculation':
            # PlanData = pd.read_excel(path + '/Adaptation Deficit_upazila.xlsx',sheet_name='Sheet1')


            self.Th_id = self.StudyArea[self.StudyArea.Upazilla == self.UpazillaName].THACODE.iloc[0]

            if self.LevelName == 'Upazilla':
                self.adp_idn = []
                self.deficiet = (self.adp_need - self.actual_adp) * 100
                self.deficiet[self.deficiet < 0] = 0
                self.new_adp_need = self.adp_need.copy()
                self.new_actual_adp = self.actual_adp.copy()
                self.adp_name = self.adp_need.columns

                PlanData = self.JoinData(self.deficiet)
                PlanData.insert(0, "Adaptation", "Deficiency")
                PlanData.insert(4, "Risk", 0)
                self.tbl1Data = PlanData[PlanData.Upazilla == self.UpazillaName].T.reset_index()
                # self.old_tbl1Data = self.tbl1Data.copy()
 
                

                riskData = self.JoinData(self.risk_calulation(self.new_actual_adp))

                self.tbl1Data.loc[self.tbl1Data.iloc[:,0] == 'Risk', self.tbl1Data.columns[-1]] = riskData.loc[self.Th_id,'Risk']

                self.old_tbl1Data = self.tbl1Data.copy()
                self.TableFillUp(self.Table1 ,self.tbl1Data, 1)
                self.drawMap(riskData, "Present Day Risk Map " + self.ZoneName + " Zone")
             

            elif self.LevelName == 'Village':
                self.deficiet_vill = self.StudyArea.join(self.deficiet, on="THACODE")
                count = self.deficiet_vill.groupby('THACODE').count()['Upazilla']
                count.name = 'Count'
                self.deficiet_vill = self.deficiet_vill.join(count,on="THACODE")
                
                self.village_data = self.deficiet_vill[self.deficiet_vill.THACODE == self.Th_id]
                # self.village_data = self.village_adp_calc(self.village_data, np.zeros(len(self.village_data)))

                self.tbl1Data = self.village_data[self.village_data.Village == self.VillageName]
                self.tbl1Data.insert(0, "Adaptation", "Deficiency")
                self.old_tbl1Data_vill = self.tbl1Data.drop(['Union', 'THACODE','Mouza','Weight','Count'], axis=1).T.reset_index()
                
                # print(self.old_tbl1Data_vill)
                self.TableFillUp(self.Table1 ,self.old_tbl1Data_vill, 1)


                riskData = self.JoinData(self.risk_calulation(self.actual_adp))
                self.drawMap(riskData, "Present Day Risk Map " + self.ZoneName + " Zone")
                

        elif sender.text() == 'Re-Calculation':
            if self.LevelName == 'Upazilla':
                self.ind = self.Table1.item(self.rowLoc, 0).text()

                self.new_tbl1Data = self.old_tbl1Data.iloc[:,[0,-1]].copy()
                new_value = self.new_tbl1Data.iloc[self.rowLoc+1, 1] - int(self.Table1.item(self.rowLoc, self.colLoc).text())
                Temp_adp = self.new_adp_need.loc[self.Th_id,:].copy()

                self.new_adp_need.loc[self.Th_id, self.ind] = self.new_adp_need.loc[self.Th_id, self.ind] - new_value/100
                self.adp_idn.append(self.rowLoc)

                optimized_res = self.optimization(self.Th_id, self.ind, self.adp_idn, self.new_adp_need)

                
                self.new_adp_need.loc[self.Th_id,:] = optimized_res
                
                self.new_actual_adp.loc[self.Th_id,:] = self.new_actual_adp.loc[self.Th_id,:] + Temp_adp - optimized_res
                
                self.deficiet = (self.new_adp_need - self.actual_adp) * 100
                self.deficiet[self.deficiet < 0] = 0

                self.new_tbl1Data = self.JoinData(self.deficiet)
                self.new_tbl1Data.insert(0, "Adaptation", "Changed Deficiency")
                self.new_tbl1Data.insert(4, "Risk", 0)
                self.new_tbl1Data = self.new_tbl1Data[self.new_tbl1Data.Upazilla == self.UpazillaName].T.reset_index()
                self.new_tbl1Data.columns = [*self.new_tbl1Data.columns[:-1], self.colLoc]

                riskData = self.JoinData(self.risk_calulation(self.new_actual_adp))
                self.new_tbl1Data.loc[self.new_tbl1Data.iloc[:,0] == 'Risk', self.new_tbl1Data.columns[-1]] = riskData.loc[self.Th_id,'Risk']
                self.old_tbl1Data = self.old_tbl1Data.set_index("index").join(self.new_tbl1Data.set_index("index")).reset_index()


                self.TableFillUp(self.Table1 ,self.old_tbl1Data, 1)
                self.drawMap(riskData, "Present Day Risk Map " + self.ZoneName + " Zone")


            elif self.LevelName == 'Village':
                if self.TableNo == 1:
                    self.ind = self.Table1.item(self.rowLoc, 0).text()

                    self.village_data.loc[(self.village_data.Village == self.VillageName), self.ind] = int(self.Table1.item(self.rowLoc, self.colLoc).text())
                    old_vill_ind_data = self.village_data.loc[:, self.ind]
                    self.village_data.iloc[:,8:] = self.village_data.iloc[:,8:].multiply(self.village_data["Weight"], axis="index")
                    new_value = self.deficiet.loc[self.deficiet.index == self.Th_id, self.ind] - self.village_data.groupby(self.village_data.THACODE).sum()[self.ind]
                                      
                    # same as upazilla re-calculation
                    Temp_adp = self.new_adp_need.loc[self.Th_id,:].copy()
                    self.new_adp_need.loc[self.Th_id, self.ind] = self.new_adp_need.loc[self.Th_id, self.ind] - new_value.values[0]/100

                    # optimized_res = self.optimization(self.Th_id, self.ind, [], self.new_adp_need)
                    
                    self.new_adp_need.loc[self.Th_id,:] = optimized_res
                    self.new_actual_adp = self.actual_adp.copy()
                    self.new_actual_adp.loc[self.Th_id,:] = self.actual_adp.loc[self.Th_id,:] + Temp_adp - optimized_res
                    
                    self.deficiet = (self.new_adp_need - self.actual_adp) * 100
                    self.deficiet[self.deficiet < 0] = 0

                    # same village calculation
                    self.deficiet_vill = self.StudyArea.join(self.deficiet, on="THACODE")
                    count = self.deficiet_vill.groupby('THACODE').count()['Upazilla']
                    count.name = 'Count'
                    self.deficiet_vill = self.deficiet_vill.join(count,on="THACODE")
                    
                    self.village_data = self.deficiet_vill[self.deficiet_vill.THACODE == self.Th_id]
                    # self.village_data = self.village_adp_calc(self.village_data, np.zeros(len(self.village_data)))

                    self.village_data.loc[:, self.ind] = old_vill_ind_data

                    self.tbl1Data = self.village_data[self.village_data.Village == self.VillageName]
                    self.tbl1Data.insert(0, "Adaptation", "Deficiency")
                    self.new_tbl1Data = self.tbl1Data.drop(['Union', 'THACODE','Mouza','Weight','Count'], axis=1).T.reset_index()
                    self.old_tbl1Data_vill = self.old_tbl1Data_vill.set_index("index").join(self.new_tbl1Data.set_index("index"), lsuffix='_left', rsuffix='_right').reset_index()

                    self.TableFillUp(self.Table1 ,self.old_tbl1Data_vill, 1)
                    
                    



                # if self.TableNo == 1:

                #     if self.rowLoc == 12:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MI_deficit')
                #     elif self.rowLoc == 14:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MI_deficit')
                #     elif self.rowLoc == 7:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MI_deficit')
                #     elif self.rowLoc == 6:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MI_deficit')


                # elif self.TableNo == 2:
                #     if self.rowLoc in [32, 33]:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MI_deficit')
                #     elif self.rowLoc in [37, 38]:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MI_deficit')
                #     elif self.rowLoc in [19, 20, 21]:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MI_deficit')
                #     elif self.rowLoc in [12, 13, 14, 15, 16, 17, 18]:
                #         PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MA_deficit')
                #         AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MI_deficit')


                # PlanData.drop(['Union ','Mouza','THACODE'], axis=1, inplace=True)
                # UzPData = PlanData[PlanData.Upazilla == self.UpazillaName]
                # self.tbl1Data = UzPData[UzPData.Village == self.VillageName].T.reset_index()
                # # self.tbl1Data
                # self.TableFillUp(self.Table1 ,self.tbl1Data, 1)

                # AutonData.drop(['Union ','Mouza','THACODE'], axis=1, inplace=True)
                # UzAData = AutonData[AutonData.Upazilla == self.UpazillaName]
                # self.tbl2Data = UzAData[UzAData.Village  == self.VillageName].T.reset_index()
                # self.TableFillUp(self.Table2 ,self.tbl2Data, 2)




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())