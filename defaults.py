# configuration parameters
cparams = {
    'N'     : {'val' : 625,      'type' : int,    'desc' : '     xyz-dim'},
    'N0'    : {'val' : 625,      'type' : int,    'desc' : ' fid xyz-dim'},
    'nx0'   : {'val' : 200,      'type' : int,    'desc' : '     xyz-dim'},
    'x2yz'  : {'val' : 1,        'type' : int,    'desc' : ' ydim / xdim'},
    'nm'    : {'val' : 20,       'type' : int,    'desc' : ' # of scales'},
    'sqrtN' : {'val' : 5,        'type' : int,    'desc' : ' N^2 samples'},
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
    'ploss' : {'val' : True,     'type' : 'bool', 'desc' : '  pspec loss'}
}

# profile parameters
pparams = {
    'inner' : {'val' : -1.5, 'type' : float, 'desc' : '  inner plaw'},
    'outer' : {'val' : -3.0, 'type' : float, 'desc' : '  outer plaw'},
    'beta'  : {'val' : -2.0, 'type' : float, 'desc' : '    ext plaw'},
    'd0'    : {'val' :  200, 'type' : float, 'desc' : '    deltavir'},
    'gamma' : {'val' :  1.1, 'type' : float, 'desc' : '     M0 / Mh'}
}

# field parameters
fparams = {
    'alpha'    : {'val' :  0.0, 'type' : float, 'desc' : ' deltac tilt'},
    'logM0'    : {'val' : 13.0, 'type' : float, 'desc' : '  tilt pivot'},
    'lptsigma' : {'val' :  1.0, 'type' : float, 'desc' : '   Rs / RLag'} 
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
samplbnds['inner']    = {'lower':  -2.0, 'upper': -0.5, 'logscale' : False}
samplbnds['outer']    = {'lower':  -2.5, 'upper': -3.5, 'logscale' : False}
samplbnds['beta']     = {'lower':  -2.9, 'upper':-0.05, 'logscale' : False}
samplbnds['d0']       = {'lower':    50, 'upper':  500, 'logscale' :  True}
samplbnds['gamma']    = {'lower':  1.01, 'upper':  1.3, 'logscale' :  True}

# field parameters
samplbnds['alpha']    = {'lower':  -0.1, 'upper':  0.1, 'logscale' : False}
samplbnds['logM0']    = {'lower':    11, 'upper':   15, 'logscale' : False}
samplbnds['lptsigma'] = {'lower':   0.3, 'upper':  2.0, 'logscale' : False}
