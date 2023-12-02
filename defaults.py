# configuration parameters
cparams = {
    'N'     : {'val' : 625,      'type' : int,    'desc' : '     xyz-dim'},
    'N0'    : {'val' : 625,      'type' : int,    'desc' : ' fid xyz-dim'},
    'nx0'   : {'val' : 200,      'type' : int,    'desc' : '     xyz-dim'},
    'x2yz'  : {'val' : 1,        'type' : int,    'desc' : ' ydim / xdim'},
    'nm'    : {'val' : 100,      'type' : int,    'desc' : ' # of scales'},
    'sqrtN' : {'val' : 5,        'type' : int,    'desc' : ' N^2 samples'},
    'box'   : {'val' : 205.,     'type' : float,  'desc' : ' boxsize/Mpc'},
    'logM1' : {'val' : 10.,      'type' : float,  'desc' : '       logM1'},
    'logM2' : {'val' : 15.,      'type' : float,  'desc' : '       logM2'},
    'zoom'  : {'val' : 0.666,    'type' : float,  'desc' : '  train zoom'},
    'kmax0' : {'val' :  6,       'type' : float,  'desc' : '  train kmax'},
    'fltr'  : {'val' : "matter", 'type' : str,    'desc' : '      filter'},
    'ctype' : {'val' : "int8",   'type' : str,    'desc' : ' cfield type'},
    'mtype' : {'val' : "int8",   'type' : str,    'desc' : '   mask type'},
    'sprms' : {'val' : "d0",     'type' : str,    'desc' : 'sample plist'},
    'mask'  : {'val' : True,     'type' : 'bool', 'desc' : '  do masking'},
    'excl'  : {'val' : False,    'type' : 'bool', 'desc' : 'do exclusion'},
    'soft'  : {'val' : False,    'type' : 'bool', 'desc' : 'thresholding'},
    'test'  : {'val' : False,    'type' : 'bool', 'desc' : '        test'},
    'sampl' : {'val' : True,     'type' : 'bool', 'desc' : '      sample'},
    'ctdwn' : {'val' : True,     'type' : 'bool', 'desc' : 'cfield order'},
    'ftdwn' : {'val' : False,    'type' : 'bool', 'desc' : '  flow order'},
    'flowl' : {'val' : True,     'type' : 'bool', 'desc' : '  inflow LPT'},
    'ploss' : {'val' : True,     'type' : 'bool', 'desc' : '  pspec loss'},
    'reprt' : {'val' : True,     'type' : 'bool', 'desc' : 'report config'}
}

# profile parameters
pparams = {
    'inner' : {'val' : -1.9, 'type' : float, 'desc' : '  inner plaw'},
    'outer' : {'val' : -3.0, 'type' : float, 'desc' : '  outer plaw'},
    'beta'  : {'val' : -2.0, 'type' : float, 'desc' : '    ext plaw'},
    'd0'    : {'val' :   93, 'type' : float, 'desc' : '    deltavir'},
    'gamma' : {'val' : 1.06, 'type' : float, 'desc' : '     M0 / Mh'}
}

# field parameters
fparams = {
    'delc0'   : {'val' :  1.62, 'type' : float, 'desc' : '       deltac'},
    'delca'   : {'val' :  0.67, 'type' : float, 'desc' : 'd[deltac]/dsg'},
    'smooth1' : {'val' :  0.71, 'type' : float, 'desc' : '      smooth1'},
    'smooth2' : {'val' :  1.76, 'type' : float, 'desc' : '      smooth2'} 
}

# all parameters
allparams = {
    'cparams' : cparams,
    'pparams' : pparams,
    'fparams' : fparams
}

# default sampling  bounds
samplbnds = {}

# profile parameters
samplbnds['inner']    = {'lower':  -2.1, 'upper': -1.6, 'logscale' : False}
samplbnds['outer']    = {'lower':  -2.7, 'upper': -3.3, 'logscale' : False}
samplbnds['beta']     = {'lower':  -2.9, 'upper':-0.05, 'logscale' : False}
samplbnds['d0']       = {'lower':    50, 'upper':  500, 'logscale' :  True}
samplbnds['gamma']    = {'lower':  1.01, 'upper':  1.3, 'logscale' :  True}

# field parameters
samplbnds['delc0']    = {'lower': 0.99*1.62, 'upper': 1.01*1.62, 'logscale' : False}
samplbnds['delca']    = {'lower': 0.99*0.67, 'upper': 1.01*0.67, 'logscale' : False}
samplbnds['smooth1'] = {'lower': 0.5, 'upper': 2.0, 'logscale' :  False}
samplbnds['smooth2'] = {'lower': 0.5, 'upper': 2.0, 'logscale' :  False}
