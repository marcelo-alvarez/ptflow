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
    'ctype' : {'val' : "float32",'type' : str,    'desc' : '  cfield type'},
    'mtype' : {'val' : "float32",'type' : str,    'desc' : '    mask type'},
    'sprms' : {'val' : "d0",     'type' : str,    'desc' : '   sopt plist'},
    'gprms' : {'val' : "dt1",    'type' : str,    'desc' : '   gopt plist'},
    'space' : {'val' : "logF",   'type' : str,    'desc' : 'scale spacing'},
    'ltype' : {'val' : "pos",    'type' : str,    'desc' : '    loss type'},
    'rname' : {'val' : "test",   'type' : str,    'desc' : '     run name'},
    'mask'  : {'val' : True,     'type' : 'bool', 'desc' : '   do masking'},
    'excl'  : {'val' : False,    'type' : 'bool', 'desc' : ' do exclusion'},
    'soft'  : {'val' : True,     'type' : 'bool', 'desc' : ' thresholding'},
    'test'  : {'val' : False,    'type' : 'bool', 'desc' : '         test'},
    'sampl' : {'val' : True,     'type' : 'bool', 'desc' : 'sample params'},
    'sopt'  : {'val' : False,    'type' : 'bool', 'desc' : '   sample opt'},
    'gopt'  : {'val' : False,    'type' : 'bool', 'desc' : '     grad opt'},
    'ctdwn' : {'val' : True,     'type' : 'bool', 'desc' : ' cfield order'},
    'ftdwn' : {'val' : False,    'type' : 'bool', 'desc' : '   flow order'},
    'flowl' : {'val' : True,     'type' : 'bool', 'desc' : '   inflow LPT'},
    'reprt' : {'val' : True,     'type' : 'bool', 'desc' : 'report config'},
    'vbose' : {'val' : True,     'type' : 'bool', 'desc' : '      verbose'}
}

# default sampling  bounds
pbounds = {}

# profile parameters
pparams = {
    'pi' : {'val' : -1.9, 'type' : float, 'desc' : '  inner plaw'},
    'po' : {'val' : -2.7, 'type' : float, 'desc' : '  outer plaw'},
    'pe' : {'val' : -1.0, 'type' : float, 'desc' : '    ext plaw'},
    'd0' : {'val' :  1.6, 'type' : float, 'desc' : 'deltavir/100'},
    'fM' : {'val' : 1.06, 'type' : float, 'desc' : '     M0 / Mh'}
}
pbounds['pi'] = {'lower': -1.3, 'upper': -1.1, 'logscale' : False}
pbounds['po'] = {'lower': -3.3, 'upper': -2.7, 'logscale' : False}
pbounds['pe'] = {'lower': -2.0, 'upper': -0.5, 'logscale' : False}
pbounds['d0'] = {'lower':  0.5, 'upper':    5, 'logscale' :  True}
pbounds['fM'] = {'lower': 1.01, 'upper':  2.0, 'logscale' :  True}

# cfield and flow parameters
fparams = {
    # dc0 and dca parameterize a "moving barrier"
    #   deltas = convolution[delta, H(<Rs)r^n}
    #   deltac = d0 + dca * <deltas^2>^(1/2)
    # i.e. model can deviate from fits by Musso & Sheth (2021):
    #   [dc0,dca,n] = [1.56, 0.63, 2], [1.78,0.81, 4]
    # cparams.fltr = "matter" --> n=2; cparams.fltr = "energy" --> n=4

    'dc0' : {'val' :  1.62, 'type' : float, 'desc' : '       deltac'}, 
    'dcp' : {'val' :  0.67, 'type' : float, 'desc' : 'd[deltac]/dsg'},

    'dt1' : {'val' :  1.95, 'type' : float, 'desc' : ' d threshold1'},
    'dt2' : {'val' :  4.41, 'type' : float, 'desc' : ' d threshold1'},
    'dtp' : {'val' :  1.00, 'type' : float, 'desc' : ' d thresholdp'},

    'ms1' : {'val' :  1.00, 'type' : float, 'desc' : ' mask smooth1'},
    'ms2' : {'val' :  1.00, 'type' : float, 'desc' : ' mask smooth2'},
    'msp' : {'val' :  1.00, 'type' : float, 'desc' : ' mask smoothp'},

    'ls1' : {'val' :  0.71, 'type' : float, 'desc' : '  lpt smooth1'},
    'ls2' : {'val' :  1.76, 'type' : float, 'desc' : '  lpt smooth2'},
    'lsp' : {'val' :  2.00, 'type' : float, 'desc' : '  lpt smoothp'},

    'fw1' : {'val' :  1.00, 'type' : float, 'desc' : ' flow weight1'},
    'fw2' : {'val' :  1.00, 'type' : float, 'desc' : ' flow weight2'},
    'fwp' : {'val' :  1.00, 'type' : float, 'desc' : ' flow weightp'},


}

pbounds['dc0'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}
pbounds['dcp'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}

pbounds['ls1'] = {'lower': 0.8, 'upper': 1.2, 'logscale' : True}
pbounds['ls2'] = {'lower': 0.3, 'upper': 1.0, 'logscale' : True}
pbounds['lsp'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}

pbounds['ms1'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}
pbounds['ms2'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}
pbounds['msp'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}

pbounds['fw1'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}
pbounds['fw2'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}
pbounds['fwp'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}

pbounds['dt1'] = {'lower': 0.9, 'upper': 1.1, 'logscale' : True}
pbounds['dt2'] = {'lower': 0.9, 'upper': 1.1, 'logscale' : True}
pbounds['dtp'] = {'lower': 0.5, 'upper': 2.0, 'logscale' : True}

# all parameters
allparams = {
    'cparams' : cparams,
    'pparams' : pparams,
    'fparams' : fparams
}