# configuration parameters
cparams = {
    'N'     : {'val' : 625,      'type' : int,    'desc' : '      xyz-dim'},
    'N0'    : {'val' : 625,      'type' : int,    'desc' : '  fid xyz-dim'},
    'nx0'   : {'val' : 200,      'type' : int,    'desc' : '      xyz-dim'},
    'x2yz'  : {'val' : 1,        'type' : int,    'desc' : '  ydim / xdim'},
    'nsc'   : {'val' : 20,       'type' : int,    'desc' : '  # of scales'},
    'sqrtN' : {'val' : 5,        'type' : int,    'desc' : '  N^2 samples'},
    'box'   : {'val' : 205.,     'type' : float,  'desc' : '  boxsize/Mpc'},
    'logM1' : {'val' : 10.,      'type' : float,  'desc' : '        logM1'},
    'logM2' : {'val' : 15.,      'type' : float,  'desc' : '        logM2'},
    'zoom'  : {'val' : 0.666,    'type' : float,  'desc' : '   train zoom'},
    'kmax0' : {'val' :  6,       'type' : float,  'desc' : '   train kmax'},
    'fltr'  : {'val' : "matter", 'type' : str,    'desc' : '       filter'},
    'ctype' : {'val' : "int8",   'type' : str,    'desc' : '  cfield type'},
    'mtype' : {'val' : "int8",   'type' : str,    'desc' : '    mask type'},
    'sprms' : {'val' : "d0",     'type' : str,    'desc' : ' sample plist'},
    'scsp'  : {'val' : "logF",   'type' : str,    'desc' : 'scale spacing'},
    'mask'  : {'val' : True,     'type' : 'bool', 'desc' : '   do masking'},
    'excl'  : {'val' : False,    'type' : 'bool', 'desc' : ' do exclusion'},
    'soft'  : {'val' : False,    'type' : 'bool', 'desc' : ' thresholding'},
    'test'  : {'val' : False,    'type' : 'bool', 'desc' : '         test'},
    'sampl' : {'val' : False,    'type' : 'bool', 'desc' : '       sample'},
    'ctdwn' : {'val' : True,     'type' : 'bool', 'desc' : ' cfield order'},
    'ftdwn' : {'val' : False,    'type' : 'bool', 'desc' : '   flow order'},
    'flowl' : {'val' : True,     'type' : 'bool', 'desc' : '   inflow LPT'},
    'ploss' : {'val' : True,     'type' : 'bool', 'desc' : '   pspec loss'},
    'reprt' : {'val' : True,     'type' : 'bool', 'desc' : ' report config'}
}

# default sampling  bounds
samplbnds = {}

# profile parameters
pparams = {
    'pi' : {'val' : -1.5, 'type' : float, 'desc' : '  inner plaw'},
    'po' : {'val' : -3.0, 'type' : float, 'desc' : '  outer plaw'},
    'pe' : {'val' : -2.0, 'type' : float, 'desc' : '    ext plaw'},
    'd0' : {'val' :  1.6, 'type' : float, 'desc' : 'deltavir/100'},
    'fM' : {'val' : 1.06, 'type' : float, 'desc' : '     M0 / Mh'}
}
samplbnds['pi'] = {'lower': -1.9, 'upper': -1.1, 'logscale' : False}
samplbnds['po'] = {'lower': -2.7, 'upper': -3.3, 'logscale' : False}
samplbnds['pe'] = {'lower': -1.0, 'upper': -2.0, 'logscale' : False}
samplbnds['d0'] = {'lower':  0.5, 'upper':    5, 'logscale' :  True}
samplbnds['fM'] = {'lower': 1.01, 'upper': 1.11, 'logscale' :  True}

# cfield and flow parameters
fparams = {
    # dc0 and dca parameterize a "moving barrier"
    #   deltas = convolution[delta, H(<Rs)r^n}
    #   deltac = d0 + dca * <deltas^2>^(1/2)
    # i.e. model can deviate from fits by Musso & Sheth (2021):
    #   [dc0,dca,n] = [1.56, 0.63, 2], [1.78,0.81, 4]
    # cparams.fltr = "matter" --> n=2; cparams.fltr = "energy" --> n=4
    'dc0' : {'val' :  1.62, 'type' : float, 'desc' : '       deltac'}, 
    'dca' : {'val' :  0.67, 'type' : float, 'desc' : 'd[deltac]/dsg'},
    'sm1' : {'val' :  0.71, 'type' : float, 'desc' : '      smooth1'},
    'sm2' : {'val' :  1.76, 'type' : float, 'desc' : '      smooth2'},
    'sma' : {'val' :     2, 'type' : float, 'desc' : '      smootha'} 
}
samplbnds['dc0'] = {'lower': 0.9*1.62, 'upper': 1.1*1.62, 'logscale' : False}
samplbnds['dca'] = {'lower': 0.9*0.67, 'upper': 1.1*0.67, 'logscale' : False}
samplbnds['sm1'] = {'lower': 0.71-0.1, 'upper': 0.71+0.1, 'logscale' : False}
samplbnds['sm2'] = {'lower': 1.76-0.1, 'upper': 1.76+0.1, 'logscale' : False}
samplbnds['sma'] = {'lower':      0.1, 'upper':       10, 'logscale' :  True}

# all parameters
allparams = {
    'cparams' : cparams,
    'pparams' : pparams,
    'fparams' : fparams
}