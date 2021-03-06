# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

hidden_imports = [
    'fiona',
	'fiona._shim',
	'fiona.schema',
    'gdal',
    'geos',
    'shapely',
    'shapely.geometry',
    'pyproj',
    'rtree',
    'geopandas.datasets',
    'pytest',
    'pandas._libs.tslibs.timedeltas',
	'mapclassify',
	'scipy'
	'scipy.special.cython_special',
	'sklearn',
	'sklearn.neighbors.typedefs',
	'sklearn.neighbors.quad_tree',
	'sklearn.tree._utils',
	'sklearn.utils._cython_blas',
	'xlrd',
	'openpyxl'
]

a = Analysis(['DAM.py'],
             pathex=['.'],
             binaries=collect_dynamic_libs("rtree"),
             datas=collect_data_files('geopandas', subdir='datasets'),
             hiddenimports=hidden_imports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='DAM',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
