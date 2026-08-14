import numpy as np 
import matplotlib.pyplot as plt 

import scipy.linalg as lin 

from pathlib import Path

import sys
sys.path.append( "/Users/markbook/Cobra/" ) 

import jackknife as jk 
import OOP_TEST as QFT 

from iminuit import Minuit

import matplotlib.colors as mcolors 

field = QFT.NLSM( TIME=256, LEN=128, COUPLING=1.54 ) 

one_2pt = QFT.one_2pt( field ) 

work_folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/2pt/2P/"

#name = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/1_Correlations/wahab/2pt/1P/1M/re_one_corr_p{0}_TAVG64.bn"
name = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/1P/re_one_corr_p{0}_TAVG256.bn"

paths = [ name.format( n ) for n in range(3) ]  

one_2pt.read_corrs( paths ) 

GUESS = [ [ 0.073296, 1182.5 ],       #CHI2 = 1.2 * ( 14 - 2 )
          [ 0.08827,   983.9 ],       #CHI2 = 0.5 * (  9 - 2 )
          [ 0.1223,   707.5 ],       # 
          [ 0.1645,   530.5 ],       # 
          [ 0.2092,   412.9 ],
          [ 0.2559,   341.9 ],
          [ 0.3037,   289.8 ]
            ][:3]  #CHI2 = 1.0 * ( 11 - 2 )

fit_ranges = [ [ 13, 26 ], 
               [ 16, 24 ], 
               [ 14, 22 ], 
               [ 15, 25 ], 
               [ 15, 25 ],
               [ 15, 25 ],
               [ 15, 24 ] ][:3]



one_2pt.fit_spectrum( fit_ranges, GUESS=GUESS )

for Es in one_2pt.spectrum: 
    print( np.mean(Es/one_2pt.spectrum[0]) ) 
#quit()

print( "MASS: ", np.mean(field.MASS) )

s_range = np.square( np.arange( 2.0001, 6.001+0.03125, 0.03125 ) * np.mean(field.MASS) ) 
k_range = np.sqrt( 0.25 * s_range - np.square( np.mean(field.MASS) ) )  
delta_range = field.PhaseShift_I1( k_range, m=np.mean( field.MASS ) ) 
kcot_delta_range  = k_range / np.tan( delta_range ) 
kcot_delta_to_m_range  = (k_range/np.mean(field.MASS)) / np.tan( delta_range ) 

TNOT = 10 

ti_list = [ [  9, 11,  9, 11 ],
            [  7,  9,  9,  9 ],
            [  6,  9,  8,  8 ] ] 

tf_list = [ [ 37, 38, 35, 35 ],
            [ 30, 37, 32, 33 ],
            [ 20, 26, 20, 26 ] ]


#the_colors  = [ 'r', 'g', 'b' ]
the_markers = [ 'o', 's', '^' ] 

the_markers_per_boost = [ 'o', '^', 's', 'p' ]

the_labels  = [ r'$E_0$', r'$E_1$', r'$E_2$' ]


plt.rcParams.update({'font.size': 20})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rc('text', usetex=True)

centimeters = 1.0 / 2.54

fig, axs = plt.subplots( 1, 1, figsize=( 5*centimeters*2, 4.8*centimeters*2 ) )

right_side = axs.spines["right"]
right_side.set_visible(False)
top_side = axs.spines["top"]
top_side.set_visible(False)

axs.plot( np.square(k_range.real/np.mean(field.MASS)), 
          kcot_delta_to_m_range.real, c='k' )

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict) 

c = mcolors.ColorConverter().to_rgb 

cmap = make_colormap( [ c('steelblue'), c('orchid'),    0.25, 
                        c('orchid'),    c('goldenrod'), 0.65, 
                        c('goldenrod'), c('firebrick'), 0.90, 
                        c('firebrick') ]   ) 

norm = mcolors.Normalize( vmin=0, vmax=3 )#8 )

samples_jk = [] ### <<<------
kcot_delta_jk_ensemble = [] ### <<<------


for BOOST in [ 0, 1, 2, 3 ]: 

    print()

    spectrum = [] 

    for m in field.MASS: 

        s_range = np.arange( 2.001, 6.01, 0.01 )**2 * m**2 
        
        k_range = np.sqrt( 0.25*s_range - m**2 ) 
        
        delta   = field.PhaseShift_I1( k_range, m=m )
        
        check = field.ComputeSpectrum_alt( s_range, BOOST, delta, diff=0.1, L=field.LEN, m=m )
    
        #print( check )
    
        spectrum.append( [ check[i][0] for i in range( len( check ) ) ] )
    
    memo = [ check[i][1] for i in range( len( check ) ) ]

    print( "N =", BOOST, "SPECTRUM: " )
    
    theory = [] 
    
    counter = 0
    for energies in np.array( spectrum ).T:
        total = np.sqrt( np.square(energies) + np.square( 2.0*np.pi*BOOST / field.LEN ) )
        print( np.mean(total), "+/-", jk.jk_std(total), memo[counter] )
        theory.append( np.mean(total) )
        counter += 1
    print()

    print()

    if BOOST%2 == 0: 
        start = 1 
    else: 
        start = 0

    folder_BOOST = work_folder + "./Spectrum/N"+str(BOOST)+"/TNOT_is_"+str(TNOT) 

    for STATE in range( 3 ): 

        #if BOOST==3 and STATE==0 : continue

        ti = ti_list[ STATE ][ BOOST ] 

        folder = folder_BOOST + "/ti_is_"+str(ti)
        path   = folder + "/E_{}".format( STATE ) 

        aux_E = [] 

        for line in open( path, 'r' ): 

            aux_E.append( np.fromstring( line, dtype=float, sep=" " )[0] ) 

        aux_E = np.array( aux_E ) 

        #samples_jk.append( aux_E ) ### <<<------

        s_jk = np.square( aux_E ) - np.square( field.UNIT * BOOST ) 

        #k_jk = np.sqrt( 0.25 * s_jk - np.square( field.MASS ) )  
        k_jk = np.sqrt( 0.25 * s_jk - np.square( np.mean(field.MASS) ) )  

        samples_jk.append( k_jk ) ### <<<------

        k_mns = np.mean( k_jk ) 
        k_err = jk.jk_std( k_jk ) 

        beta = field.UNIT * BOOST / aux_E

        gamma = 1.0 / np.sqrt( 1.0 - beta*beta ) 

        aux_jk = ( 2.0 *(STATE + start) + BOOST%2 ) * np.pi - gamma * field.LEN * k_jk 
        
        #aux_jk = ( 2.0 * STATE + BOOST%2 ) * np.pi - gamma * field.LEN * k_jk  

        #delta_jk = ( 2.0*np.pi + aux_jk ) % ( 2.0 * np.pi ) * 0.5 

        delta_jk = aux_jk * 0.5

        kcot_delta_jk = k_jk / np.tan( delta_jk ) 

        kcot_delta_jk_ensemble.append( kcot_delta_jk ) ### <<<------

        kcot_delta_mns = np.mean( kcot_delta_jk ).real 
        kcot_delta_err = jk.jk_std( kcot_delta_jk ).real

        kcot_delta_mns_to_m = np.mean( kcot_delta_jk/field.MASS ).real 
        kcot_delta_err_to_m = jk.jk_std( kcot_delta_jk/field.MASS ).real 

        the_label = None 

        if not BOOST: 
            the_label = None#the_labels[STATE] 

        #axs.errorbar( np.square(k_mns/np.mean(field.MASS)), 
        #              kcot_delta_mns, xerr=k_err, yerr=kcot_delta_err, 
        #              fmt=the_markers[STATE], c= cmap( norm( BOOST ) ) , zorder=24, 
        #              label = the_label, ms=8, capsize=6, mew=1.8, elinewidth=2.5, 
        #              mfc='w' ) 
        
        axs.errorbar( np.square(k_mns/np.mean(field.MASS)), 
                      kcot_delta_mns_to_m, xerr=k_err, yerr=kcot_delta_err_to_m, 
                      fmt=the_markers_per_boost[BOOST], c= "firebrick" , zorder=24, 
                      label = the_label, ms=8, capsize=6, mew=1.8, elinewidth=2.5, 
                      mfc='w' ) 

""" 
The code below fits kcot_delta as an effective range expansion
"""      

## Change variables: 'a', 'r' to 'A', 'R' 
## This is because the effective range expansion 'a' and 'r' 
## would actually be: 'a' = -1/'A' & 'r' = 'R'/2

def eff_range( pars, k_cm ): 

    a = pars[0]
    r = pars[1]

    return a + r * np.square( k_cm ) 

nof_pars = 2

samples_jk = np.array( samples_jk ) 
kcot_delta_jk_ensemble = np.array( kcot_delta_jk_ensemble )

cov_matrix = jk.jk_cov_matrix( kcot_delta_jk_ensemble ) 

InvCov     = np.matrix( np.linalg.inv( cov_matrix ) ) 

dof = len( samples_jk ) - nof_pars 

a_guess = 0.55987
r_guess = 4.10629

a_jk, r_jk = [], [] 

Chi2_dof = []

counter = -1 
for k_jk_copy in samples_jk.T: 

    counter += 1 

    data = kcot_delta_jk_ensemble.T[counter]

    def Chi2( a, r ): 

        fit   = eff_range( [a,r], k_jk_copy ) 
        diffs = np.matrix( data - fit ).T 

        return ( diffs.T @ InvCov @ diffs )[0,0] 
    
    Chi2_fit = Minuit( Chi2, a=a_guess, r=r_guess ) 

    Chi2_fit.errordef = Minuit.LEAST_SQUARES 

    Chi2_fit.strategy = 2 

    Chi2_fit.errors["a"] = 0.001 
    Chi2_fit.errors["r"] = 1.0 

    Chi2_fit.migrad() 

    a_jk.append( Chi2_fit.values[0] )
    r_jk.append( Chi2_fit.values[1] ) 

    Chi2_dof.append( Chi2_fit.fval / dof )  

'''
Path( "./EffRange" ).mkdir( parents=True, exist_ok=True ) 

file = open( "./EffRange/EffRange_A.jk", "w" ) 
for x in a_jk:
    file.write( "{0}\n".format( x ) )
file.close() 

file = open( "./EffRange/EffRange_R.jk", "w" ) 
for x in r_jk:
    file.write( "{0}\n".format( x ) )
file.close() 
'''

a_jk = np.array( a_jk )
r_jk = np.array( r_jk )

print( a_jk[:3] )

print( )
print( "a: " )
print( np.mean( a_jk ), "+/-", jk.jk_std( a_jk ) )
print( )
print( "r: " )
print( np.mean( r_jk ), "+/-", jk.jk_std( r_jk ) )
print( )
print( "Chi2 per dof: ", np.mean( Chi2_dof ) ) 
print( )

print( )
print( "am: " )
print( np.mean( (-1.0 / a_jk)*field.MASS ), "+/-", jk.jk_std( (-1.0/a_jk)*field.MASS ) )
print( )
print( "rm: " )
print( np.mean( 2.0*r_jk*field.MASS ), "+/-", jk.jk_std( 2.0*r_jk*field.MASS ) )
print( )
print( "Chi2 per dof: ", np.mean( Chi2_dof ) ) 
print( )

k_cot_fit_jk_aux = [] 

s_range_for_fit = np.square( np.arange( 2.0001, 6.001+0.05, 0.05 ) * np.mean(field.MASS) ) 
k_range_for_fit = np.sqrt( 0.25 * s_range_for_fit - np.square( np.mean(field.MASS) ) )  

np.shape( k_range_for_fit )

print( "LOOP START" )

######################
### FOR ERRORBANDS ### 

for pars in np.array( [ a_jk, r_jk ] ).T: 

    k_cot_fit = eff_range( pars, k_range_for_fit ) 

    k_cot_fit_jk_aux.append( k_cot_fit ) 


print( "LOOP END" )
print( np.shape( k_cot_fit_jk_aux ) )
k_cot_fit_jk = np.array( k_cot_fit_jk_aux ).T 
print( "LOOP END" )


print( "HERE 0" )

print( np.shape( k_cot_fit_jk ) )

k_cot_fit_mns = np.array( [ np.mean(   x/field.MASS ) for x in k_cot_fit_jk ] )
k_cot_fit_err = np.array( [ jk.jk_std( x/field.MASS ) for x in k_cot_fit_jk ] )

k_cot_fit_upper = k_cot_fit_mns + k_cot_fit_err 
k_cot_fit_lower = k_cot_fit_mns - k_cot_fit_err 

print( "HERE " ) 

axs.fill_between( np.square(k_range_for_fit/np.mean( field.MASS )), 
                  k_cot_fit_upper, k_cot_fit_lower, 
                  interpolate=True, color='g', alpha=0.5, label=r"${\rm eff. range}$" )

### FOR ERRORBANDS ### 
######################

print( "HERE " )


""" 
The code above fits kcot_delta as an effective range expansion
""" 
        
#axs.set_ylabel( r"$(q^\star/m)\cot\delta$" )
axs.set_ylabel( r"$(q^\star/m)\mathcal{K}^{-1}/\rho$" )
axs.set_xlabel( r"$q^{\star 2}/m^2$" )

axs.set_ylim( 3.139, 8.139 )
axs.set_xlim( -0.001, 4.249 )

#axs.legend()

#sm = plt.cm.ScalarMappable( cmap=cmap, norm=norm ) 
#sm.set_array([])

#cbar = fig.colorbar( sm, ax=axs, label=r'$\mathbf{d}$' ) 

#cbar.set_ticks( [0,1,2,3] )
#cbar.set_ticklabels( [0,3] )

#fig.suptitle( r'$T\times L=256\times128 \qquad \mathbf{P}=\frac{2\pi}{L}\times\mathbf{d}$' )

#import matplotlib.lines as mlines
#marker1 = mlines.Line2D([], [], color='k', marker='o', mfc='w', mew=1.8, linestyle='None', markersize=8)
#marker2 = mlines.Line2D([], [], color='k', marker='s', mfc='w', mew=1.8, linestyle='None', markersize=8)
#marker3 = mlines.Line2D([], [], color='k', marker='^', mfc='w', mew=1.8, linestyle='None', markersize=8)
#marker4 = mlines.Line2D([], [], color='g', marker='None', linestyle='-', linewidth=10.0, alpha=0.8, markersize=8)
#marker5 = mlines.Line2D([], [], color='k', marker='None', linestyle='-', linewidth=1.0, alpha=1.0, markersize=8)

#axs.legend([marker1, marker2, marker3, marker4, marker5], [r'$(E_0,\mathbf{P})$', r'$(E_1,\mathbf{P})$', r'$(E_2,\mathbf{P})$', r'eff. range', r'Analytic'])

as_type='pdf'

name = work_folder+'/kcot_delta_Lattice_A.'+as_type

plt.savefig( name, dpi=384,
        format=as_type,
        transparent=True, bbox_inches='tight', pad_inches=0.1,
        metadata=None )

plt.show()