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
            data.iloc[4:,1:] = data.iloc[4:,1:].astype(float).round(0).astype(int)
            if self.LevelName == 'Upazilla':
                fix_row = 4
            elif self.LevelName == 'Village':
                fix_row = 3

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


    def optimization(self, path, i, ind, adp_need):
        constraint_01 = pd.read_excel(path + "/Constraints.xlsx",sheet_name="Constraint_01",index_col="Indicators",skiprows=1)
        constraint_02 = pd.read_excel(path + "/Constraints.xlsx",sheet_name="Constraint_02",index_col="Indicators",skiprows=1)
        Lower_Bound = pd.read_excel(path + "/Constraints.xlsx",sheet_name="Lower_bound",index_col="Indicators",skiprows=1)
        Upper_Bound = pd.read_excel(path + "/Constraints.xlsx",sheet_name="Upper_bound",index_col="Indicators",skiprows=1)
        Upper_Bound = Upper_Bound.astype(float)

        adp_upz = adp_need.loc[i,:].copy()

        cons1 = constraint_01.loc[ind,:]
        cons2 = constraint_02.loc[ind,:]

        init = np.zeros(20)
        lower_b = Lower_Bound.loc[ind,:]
        upper_b = Upper_Bound.loc[ind,:]
        lower_b[lower_b == -1] = adp_upz.loc[(lower_b == -1).to_numpy()[4:19]].to_numpy()
        upper_b[upper_b == -1] = adp_upz.loc[(upper_b == -1).to_numpy()[4:19]].to_numpy()
        bound = tuple(zip(lower_b, upper_b))

        coeff = cons1.iloc[4:19].copy()
        coeff[adp_upz.index == ind] = 0
        coeff2 = cons2.iloc[4:19].copy()
        coeff2[adp_upz.index == ind] = 0

        a = [self.int_input.loc[i,:].Exposure * self.int_input.loc[i,:].Hazard,
            self.int_input.loc[i,:].Exposure,
            self.int_input.loc[i,:].Sensitivity,
            self.int_input.loc[i,:].Exposure + self.int_input.loc[i,:].Sensitivity]
        a.extend(list(adp_upz))
        a.append(self.int_input.loc[i,:].Hazard)

        Objective = lambda x: x[0]*x[19]*((0.0548*x[1]+0.358*x[2]+0.587*x[3])-(0.077*x[4]+0.075*x[5]+0.05*x[6]+0.0588*x[7]+0.0569*x[8]+0.0283*x[9]
                            +0.0467*x[10]+0.0449*x[11]+0.0366*x[12]+0.0266*x[13]+0.0352*x[14]+0.06*x[15]+0.0523*x[16]+0.0487*x[17]+0.302*x[18]))

        ineq1 = lambda x: (cons1.x0*x[0] + cons1.x1*x[1] + cons1.x2*x[2] + cons1.x3*x[3] + cons1.x4*x[4] + cons1.x5*x[5] + cons1.x6*x[6] +
                cons1.x7*x[7] + cons1.x8*x[8] + cons1.x9*x[9] + cons1.x10*x[10] + cons1.x11*x[11] + cons1.x12*x[12] + cons1.x13*x[13] +
                cons1.x14*x[14] + cons1.x15*x[15] + cons1.x16*x[16] + cons1.x17*x[17] + cons1.x18*x[18] + cons1.x19*x[19])

        # ineq2 = lambda x: (cons2.x0*x[0] + cons2.x1*x[1] + cons2.x2*x[2] + cons2.x3*x[3] + cons2.x4*x[4] + cons2.x5*x[5] + cons2.x6*x[6] +
        #         cons2.x7*x[7] + cons2.x8*x[8] + cons2.x9*x[9] + cons2.x10*x[10] + cons2.x11*x[11] + cons2.x12*x[12] + cons2.x13*x[13] +
        #         cons2.x14*x[14] + cons2.x15*x[15] + cons2.x16*x[16] + cons2.x17*x[17] + cons2.x18*x[18] + cons2.x19*x[19])

        ineq2 = lambda x: 0.0548 * x[1] + 0.358 * x[2] + 0.587 * x[3] - a[2]

        eq1 = NonlinearConstraint(lambda x: x[0] * x[19] - a[0], 0, 0)
        eq2 = lambda x: x[0] - a[1]
        
        eq3 = lambda x: x[0] + 0.0548 * x[1] + 0.358 * x[2] + 0.587 * x[3] - a[3]
        eq4 = lambda x: x[19] - a[19]

        cons_eq = [{'type': 'ineq', 'fun': ineq1},
                {'type': 'ineq', 'fun': ineq2},
                eq1,
                {'type': 'eq', 'fun': eq2},
                {'type': 'eq', 'fun': eq3},
                {'type': 'eq', 'fun': eq4},
                # {'type': 'eq', 'fun': eq5}
                ]

        for idx in range(4,19):
            if (coeff[idx-4] == 0):
                cons_eq.append({'type': 'eq', 'fun': lambda x, idx=idx: x[idx] - a[idx]})

        solution = minimize(Objective, init, method='trust-constr', bounds=bound, constraints=cons_eq)
        
        return solution.x[4:19]


    def Action(self):
        sender = self.sender()

        path = self.DataDir + '/Hazards/' + self.HazardName

        
        if sender.text() == 'Calculation':
            # PlanData = pd.read_excel(path + '/Adaptation Deficit_upazila.xlsx',sheet_name='Sheet1')
            self.deficiet = (self.adp_need - self.actual_adp) * 100
            self.deficiet[self.deficiet < 0] = 0
            self.new_adp = self.adp_need.copy()

            if self.LevelName == 'Upazilla':
                PlanData = self.JoinData(self.deficiet)
                PlanData.insert(0, "Adaptation", "Deficiency")
                PlanData.insert(4, "Risk", 0)
                self.tbl1Data = PlanData[PlanData.Upazilla == self.UpazillaName].T.reset_index()
                # self.old_tbl1Data = self.tbl1Data.copy()
 
                self.Th_id = self.StudyArea_Up_Zone[self.StudyArea_Up_Zone.Upazilla == self.UpazillaName].index[0]

                riskData = self.JoinData(self.risk_calulation(self.actual_adp))

                self.tbl1Data.loc[self.tbl1Data.iloc[:,0] == 'Risk', self.tbl1Data.columns[-1]] = riskData.loc[self.Th_id,'Risk']

                self.old_tbl1Data = self.tbl1Data.copy()
                self.TableFillUp(self.Table1 ,self.tbl1Data, 1)
                self.drawMap(riskData, "Present Day Risk Map " + self.ZoneName + " Zone")
             

            elif self.LevelName == 'Village':
                self.deficiet_vill = self.StudyArea.join(self.deficiet, on="THACODE")
                count = self.deficiet_vill.groupby('THACODE').count()['Upazilla']
                count.name = 'Count'
                self.deficiet_vill = self.deficiet_vill.join(count,on="THACODE")

                self.deficiet_vill.iloc[:,8:] = self.deficiet_vill.iloc[:,8:].multiply(self.deficiet_vill["Weight"], axis="index").multiply(self.deficiet_vill["Count"], axis="index")

                self.deficiet_temp = self.deficiet_vill.loc[(self.deficiet_vill.Upazilla == self.UpazillaName) & (self.deficiet_vill.Village == self.VillageName)].T
                self.old_tbl1Data_vill = self.deficiet_temp.drop(['Union', 'THACODE','Mouza','Weight','Count'], axis=0).reset_index()
                self.TableFillUp(self.Table1 ,self.old_tbl1Data_vill, 1)
                # print(self.deficiet_vill)
                


                auto_deficiet = self.deficiet_temp.join(self.Autonomas_Weight)
                auto_deficiet.iloc[:,-1] = auto_deficiet.iloc[:,0] * auto_deficiet.iloc[:,-1]
                auto_deficiet.insert(0, "Adaptation", "Deficiency")
                auto_deficiet.drop(['District', 'Mouza', 'THACODE', 'Union','Village','Upazilla','Weight','Zone','Count'], axis=0, inplace=True)
                auto_deficiet = auto_deficiet.reset_index()
                self.old_tbl2Data_vill = auto_deficiet.iloc[:,-2:]
                self.old_tbl2Data_vill.loc[-1] = ["Adaptation", "Deficiency"]
                self.old_tbl2Data_vill.index = self.old_tbl2Data_vill.index + 1
                self.old_tbl2Data_vill = self.old_tbl2Data_vill.sort_index()
                self.old_tbl2Data_vill.T.insert(0, "Adaptation", "Deficiency")

                self.TableFillUp(self.Table2 ,self.old_tbl2Data_vill, 2)

                riskData = self.JoinData(self.risk_calulation(self.actual_adp))
                self.drawMap(riskData, "Present Day Risk Map " + self.ZoneName + " Zone")
                

        elif sender.text() == 'Re-Calculation':
            if self.LevelName == 'Upazilla':
                self.ind = self.Table1.item(self.rowLoc, 0).text()

                self.new_tbl1Data = self.old_tbl1Data.iloc[:,[0,-1]].copy()
                new_value = self.new_tbl1Data.iloc[self.rowLoc+1, 1] - int(self.Table1.item(self.rowLoc, self.colLoc).text())
                Temp_adp = self.new_adp.loc[self.Th_id,:].copy()
                self.new_adp.loc[self.Th_id, self.ind] = self.new_adp.loc[self.Th_id, self.ind] - new_value/100

                optimized_res = self.optimization(path, self.Th_id, self.ind, self.new_adp)
                
                self.new_adp.loc[self.Th_id,:] = optimized_res
                self.new_actual_adp = self.actual_adp.copy()
                self.new_actual_adp.loc[self.Th_id,:] = self.actual_adp.loc[self.Th_id,:] + Temp_adp - optimized_res
                
                self.deficiet = (self.new_adp - self.actual_adp) * 100
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
                    self.deficiet_vill.loc[(self.deficiet_vill.Upazilla == self.UpazillaName) & (self.deficiet_vill.Village == self.VillageName),self.ind] = int(self.Table1.item(self.rowLoc, self.colLoc).text())
                    # deficiet_vill.groupby(deficiet_vill.THACODE).sum()

                    # self.new_tbl1Data = self.old_tbl1Data.iloc[:,[0,-1]].copy()
                    # new_value = self.new_tbl1Data.iloc[self.rowLoc+1, 1] - int(self.Table1.item(self.rowLoc, self.colLoc).text())
                    # Temp_adp = self.new_adp.loc[self.Th_id,:].copy()


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