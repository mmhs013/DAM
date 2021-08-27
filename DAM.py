"""
Dynamic Adaptation Model (DAM)

Author: Md. Manjurul Husain Shourov
last edited: 12/08/2021
Website: https://github.com/mmhs013/DAM
"""

import sys
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shutil import copyfile
from PyQt5 import uic, QtWidgets, QtGui

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('DAM_GUI.ui', self)
        self.setWindowTitle('Dynamic Adaptation Model')
        self.show()

        self.RootDir = os.getcwd()
        self.DataDir = self.RootDir+"/DAM_Database/"
        # self.StudyArea = gpd.read_file(self.DataDir + '\Shapefiles\DAM_Upazilla_WGS_1984.shp')
        self.StudyArea = pd.read_csv(self.DataDir + '/Study Area.csv')

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
        self.Table1PBtn.clicked.connect(lambda: self.save(self.tbl1Data,'SaveTable'))
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


    def TableFillUp(self, Table, data):
        Table.setRowCount(0)
        Table.setColumnCount(data.shape[1])
        Table.setHorizontalHeaderLabels(data.iloc[0])

        for r in range(data.shape[0]-1):
            Table.insertRow(r)
            for c in range(data.shape[1]):
                Table.setItem(r, c, QtWidgets.QTableWidgetItem(str(data.iloc[r+1, c])))
    
    def cellchanged(self, Table, TableNo):
        self.TableNo = TableNo
        self.colLoc = Table.currentColumn()
        self.rowLoc = Table.currentRow()
        self.CalculatePButton.setText("Re-Calculation")

    #     Table.cellActivated.connect(lambda: self.cell(Table))

    # def cell(self, Table):
    #     text = Table.currentItem().text()
    #     print(text)

    def risk_calulation(self, upzName, path, diff):
        df = pd.read_excel(path + '/Risk_calculation_1.xlsx')
        actual_adp = pd.read_excel(self.DataDir + "/For_Risk.xlsx", sheet_name="Risk")
        wt = pd.read_excel(self.DataDir + "/For_Risk.xlsx",sheet_name="Weight")
        wt.set_index("indicators",inplace=True)
        wt = wt.iloc[:,0]
        upz_adp = actual_adp[actual_adp.Upazilla == upzName]
        adp = upz_adp.iloc[0,6:]
        adp = adp + diff
        cp = 100 - (wt * adp).sum()
        vul = cp + upz_adp.Sensitivity.values
        rsk = vul * upz_adp.Exposure.values * upz_adp.Hazard.values
        scl_rsk = (rsk - 42160.08267) / (286449.194 - 42160.08267) * 100

        df.loc[df.THANAME == upzName, 'Risk'] = scl_rsk

        return df
    
    def drawMap(self, riskData, title):
        gdf = gpd.read_file(self.DataDir +"/Shapefiles/DAM_Upazilla_WGS_1984.shp")
        gdf.THACODE = gdf.THACODE.astype(int)
        gdfz = gdf[gdf.Zone == self.ZoneName]
        gdfz = gdfz.loc[:,['THACODE','geometry']]
        self.joinData = gdfz.set_index('THACODE').join(riskData.set_index('THACODE'))

        ax = self.joinData.plot(alpha=0.8,column='Risk', edgecolor='black', cmap='RdYlGn_r', figsize=(7,10),
        legend=True, legend_kwds = {'label': 'Risk Level'} , vmin=0,vmax=100)

        margin_x = 0.005
        margin_y = 0.005
        xlim = ([self.joinData.total_bounds[0] - margin_x,  self.joinData.total_bounds[2]+ margin_x])
        ylim = ([self.joinData.total_bounds[1] - margin_y,  self.joinData.total_bounds[3]+ margin_y])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title,fontsize=14)

        print(self.joinData)
        for x, y, label in zip(self.joinData.centroid.x, self.joinData.centroid.y, self.joinData["THANAME"]):
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


    def Action(self):
        sender = self.sender()

        path = self.DataDir + '/Hazards/' + self.HazardName

        if sender.text() == 'Calculation':
            if self.LevelName == 'Upazilla':
                PlanData = pd.read_excel(path + '/Adaptation Deficit_upazila.xlsx',sheet_name='Sheet1')
                self.tbl1Data = PlanData[PlanData.Upazilla == self.UpazillaName].T.reset_index()
                self.old_tbl1Data = self.tbl1Data.copy()

                self.TableFillUp(self.Table1 ,self.tbl1Data)

                riskData = pd.read_excel(path + '/Risk_calculation_1.xlsx')
                self.drawMap(riskData, "Present Day Risk Map" + self.ZoneName + "Zone")
                

            elif self.LevelName == 'Village':
                PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village.xlsx',sheet_name='MA_deficit')
                PlanData.drop(['Union ','Mouza','THACODE'], axis=1, inplace=True)
                UzPData = PlanData[PlanData.Upazilla == self.UpazillaName]
                self.tbl1Data = UzPData[UzPData.Village == self.VillageName].T.reset_index()
                self.TableFillUp(self.Table1 ,self.tbl1Data)

                AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village.xlsx',sheet_name='MI_deficit')
                AutonData.drop(['Union ','Mouza','THACODE'], axis=1, inplace=True)
                UzAData = AutonData[AutonData.Upazilla == self.UpazillaName]
                self.tbl2Data = UzAData[UzAData.Village  == self.VillageName].T.reset_index()
                self.TableFillUp(self.Table2 ,self.tbl2Data)

        elif sender.text() == 'Re-Calculation':
            if self.LevelName == 'Upazilla':
                if self.rowLoc == 13:
                    PlanData = pd.read_excel(path + '/Adaptation Deficit_upazila_after_filling_Communication_infrastructure.xlsx',sheet_name='Sheet1')

                elif self.rowLoc == 15:
                    PlanData = pd.read_excel(path + '/Adaptation Deficit_upazila_after_filling_Growth_center.xlsx',sheet_name='Sheet1')

                elif self.rowLoc == 8:
                    PlanData = pd.read_excel(path + '/Adaptation Deficit_upazila_after_filling_Irrigation System.xlsx',sheet_name='Sheet1')

                elif self.rowLoc == 7:
                    PlanData = pd.read_excel(path + '/Adaptation Deficit_upazila_after_filling_Safe Drinking water.xlsx',sheet_name='Sheet1')

                self.tbl1Data = PlanData[PlanData.Upazilla == self.UpazillaName].T.reset_index()
                self.TableFillUp(self.Table1 ,self.tbl1Data)

                diff = self.old_tbl1Data.set_index("index").iloc[5:,0] - self.tbl1Data.set_index("index").iloc[5:,0]
                riskData = self.risk_calulation(self.UpazillaName, path, diff)

                self.drawMap(riskData, "Revised Present Day Risk Map" + self.ZoneName + "Zone")

            elif self.LevelName == 'Village':
                if self.TableNo == 1:

                    if self.rowLoc == 12:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MI_deficit')
                    elif self.rowLoc == 14:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MI_deficit')
                    elif self.rowLoc == 7:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MI_deficit')
                    elif self.rowLoc == 6:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MI_deficit')


                elif self.TableNo == 2:
                    if self.rowLoc in [32, 33]:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Communicvation_infrastructure.xlsx',sheet_name='Village_MI_deficit')
                    elif self.rowLoc in [37, 38]:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_growth_center.xlsx',sheet_name='Village_MI_deficit')
                    elif self.rowLoc in [19, 20, 21]:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Irrigation_system.xlsx',sheet_name='Village_MI_deficit')
                    elif self.rowLoc in [12, 13, 14, 15, 16, 17, 18]:
                        PlanData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MA_deficit')
                        AutonData = pd.read_excel(path + '/MA&MI_Deficit_for village_after filling_Safe_drinking_water.xlsx',sheet_name='Village_MI_deficit')


                PlanData.drop(['Union ','Mouza','THACODE'], axis=1, inplace=True)
                UzPData = PlanData[PlanData.Upazilla == self.UpazillaName]
                self.tbl1Data = UzPData[UzPData.Village == self.VillageName].T.reset_index()
                self.TableFillUp(self.Table1 ,self.tbl1Data)

                AutonData.drop(['Union ','Mouza','THACODE'], axis=1, inplace=True)
                UzAData = AutonData[AutonData.Upazilla == self.UpazillaName]
                self.tbl2Data = UzAData[UzAData.Village  == self.VillageName].T.reset_index()
                self.TableFillUp(self.Table2 ,self.tbl2Data)




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())