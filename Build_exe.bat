pyinstaller DAM_gui.spec
echo D | xcopy .\DAM_Database .\dist\DAM_Database /E
copy .\DAM_GUI.ui .\dist\DAM_GUI.ui