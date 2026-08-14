import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 

from pathlib import Path

import sys
sys.path.append( "/Users/markbook/Cobra/" ) 

import jackknife   as jk 
import OOP_TEST_v4 as QFT 

import matplotlib.colors as mcolors 

SIZE  = 4 
BOOST = 1 
SKIP  = 1 
TMAX  = 42 

field = QFT.NLSM( TIME=256, LEN=128, COUPLING=1.54 ) 

one_2pt = QFT.one_2pt( field ) 
name    = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/1P/re_one_corr_p{0}_TAVG256.bn" 
paths   = [ name.format( n ) for n in range(1) ]  

one_2pt.read_corrs( paths ) 

GUESS      = [ [ 0.07329, 1183 ] ] 
fit_ranges = [ [ 13, 25 ] ] 

one_2pt.fit_spectrum( fit_ranges, GUESS=GUESS, CHI2dof=True ) 

print( "MASS: ", np.mean( field.MASS ) ) 

MASS = np.mean( field.MASS ) 
M2   = MASS * MASS 

## Compute theoretical FV spectrum 
s_range = np.square( np.arange( 2.0001, 6.002, 0.002 ) ) * M2 
spectrum_theory = [] 
for level in field.get_spectrum( s_range, BOOST, field.phase_shift_I1 ) :
    if level: 
        spectrum_theory.append( level[0] ) 
##print( np.array( spectrum_theory ) / MASS ) 

## Location of the correlation functions 
folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/2P/N"+str(BOOST)+"/"
re_name = "re_two_corr_i{0}_j{1}_TAVG256_new.bn"
im_name = "im_two_corr_i{0}_j{1}_TAVG256_new.bn" 
re_paths = [ [ folder+re_name.format( i, j ) for i in range( SKIP, SKIP+SIZE ) ] for j in range( SKIP, SKIP+SIZE ) ]
im_paths = False 
if BOOST: 
    im_paths = [ [ folder+im_name.format( i, j ) for i in range( SKIP, SKIP+SIZE ) ] for j in range( SKIP, SKIP+SIZE ) ] 


#### Options for analysis of two-particle states 
##BOOST = 0
#STATE, GUESS = 0, 0.17344  ## STATE=0 
#the_label = r"$\chi^2=1.2\times(16-1)\quad\sqrt{s_n}/m=2.370(3)$"
#STATE, GUESS = 1, 0.238  ## STATE=1   
#the_label = r"$\chi^2=1.4\times(10-1)\quad\sqrt{s_n}/m=3.276(4)$"
#STATE, GUESS = 2, 0.320  ## STATE=2    

##BOOST = 1
STATE, GUESS = 0, 0.1607  ## STATE=0 
the_label = r"$\chi^2=1.9\times(15-1)\quad\sqrt{s_n}/m=2.089(2)$"
#STATE, GUESS = 1, 0.2060  ## STATE=1   
#the_label = r"$\chi^2=1.1\times(16-1)\quad\sqrt{s_n}/m=2.740(3)$"
#STATE, GUESS = 2, 0.2793 ## STATE=2  
 
##BOOST = 2
#STATE, GUESS = 0, 0.1939  ## STATE=0 
#the_label = r"$\chi^2=0.7\times(11-1)\quad\sqrt{s_n}/m=2.284(3)$"
#STATE, GUESS = 1, 0.2469  ## STATE=1   
#the_label = r"$\chi^2=0.8\times(11-1)\quad\sqrt{s_n}/m=3.105(5)$"
#STATE, GUESS = 2, 0.2266 ## STATE=2   

##BOOST = 3
#STATE, GUESS = 0, 0.2105  ## STATE=0 
#the_label = r"$\chi^2=1.0\times(15-1)\quad\sqrt{s_n}/m=2.053(4)$"
#STATE, GUESS = 1, 0.2351  ## STATE=1   
#the_label = r"$\chi^2=0.6\times(16-1)\quad\sqrt{s_n}/m=2.505(5)$"
#STATE, GUESS = 2, 0.2914  ## STATE=2  


"""
tf_vals || P = 0 || P = 1 || P = 2 || P = 3  

     E0 || 37    || 38    || 35    || 35 
     
     E1 || 30    || 37    || 32    || 33
     
     E2 || 20    || 26    || 20    || 26
"""

tf_vals = [ [ 32, 38, 35, 35 ], 
            [ 30, 37, 32, 33 ], 
            [ 20, 26, 20, 26 ] ] 


Chi2s, Es, dEs = [], [], []
ti_range       = np.arange( 7, 16 ) 

work_folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/2pt/2P"

Path(work_folder+"/Spectrum").mkdir(parents=True, exist_ok=True) 
Path(work_folder+"/Spectrum/N{0}".format(BOOST)).mkdir(parents=True, exist_ok=True) 

#TNOT_MIN = 5 
#TNOT_MAX = 12

TNOT = 10 

ti_list = [ [  9, 11,  9, 11 ],
            [  7,  9,  9,  9 ],
            [  6,  9,  8,  8 ] ] 

ti_list = [ [  9, 11,  9, 11 ],
            [  7,  9,  9,  9 ],
            [  6,  9,  8,  8 ] ] 

#tf_list = [ [ 37, 38, 35, 35 ],
#            [ 30, 37, 32, 33 ],
#            [ 20, 26, 20, 26 ] ]

tf_list = [ [ 25, 25, 20, 25 ],
            [ 25, 25, 20, 25 ],
            [ 25, 25, 20, 25 ] ]

tf_list = [ [ 25, 25, 20, 25 ],
            [ 16, 25, 20, 25 ],
            [ 25, 25, 20, 25 ] ]

GUESS   = [ [ 0.173, 0.238 ], 
            [ 0.161, 0.206 ], 
            [ 0.194, 0.247 ],
            [ 0.161, 0.210 ] ][ BOOST ]

fit_ranges = [ ( ti_list[n][BOOST], tf_list[n][BOOST] ) for n in range( 2 ) ] 

two_2pt = QFT.two_2pt( field, TNOT=TNOT, SIZE=SIZE, BOOST=BOOST, SKIP=SKIP, TMAX=TMAX ) 

two_2pt.read_corrs( re_paths, im_paths=im_paths ) 
two_2pt.solve_GEVP()
two_2pt.fit_spectrum( fit_ranges, GUESS=GUESS ) 

print( np.shape( two_2pt.Chi2dof ) )
print( np.mean( two_2pt.Chi2dof[STATE] ), fit_ranges[STATE] )
Ecm = np.sqrt( two_2pt.spectrum[STATE]**2 - (field.UNIT*BOOST)**2 )
print( np.mean(   Ecm / one_2pt.NLSM.MASS_jk ) )
print( jk.jk_std( Ecm / one_2pt.NLSM.MASS_jk ) )

En_jk = two_2pt.spectrum[ STATE ] 

check = []

for jk_copy in range( 1000 ): 

    C0 = two_2pt.corr_matrix[ jk_copy ][ TNOT ] 

    aux = [] 

    for t in range( TMAX ): 

        Ct = two_2pt.corr_matrix[ jk_copy ][ t ] 

        vec_n = two_2pt.interps[ jk_copy ][ t ][ :, STATE ] 
        vec_m = two_2pt.interps[ jk_copy ][ t ][ :, STATE ] 

        numerator   = ( vec_m.H @ Ct @ vec_n )[0,0] 
        denominator = ( vec_m.H @ C0 @ vec_n )[0,0] 
        factor      = np.exp( ( t - TNOT ) * En_jk[ jk_copy ] ) 

        aux.append( factor * numerator / denominator ) 

    check.append( aux )

check = np.array( check ).T

print()
print( len( check ) )
print()

t_range = np.arange( TMAX )

check_avgs = [ np.mean(   ensemble ).real for ensemble in check ]
check_errs = [ jk.jk_std( ensemble ) for ensemble in check ] 


plt.rcParams.update({'font.size': 22})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r"""\usepackage{bm}"""]

plt.rc('text', usetex=True)

fig, axs = plt.subplots( 1, 1, figsize=( 6.4*(1.5+0.05), 2.4*1.5 ) ) 

right_side = axs.spines["right"]
right_side.set_visible(False)
top_side = axs.spines["top"]
top_side.set_visible(False)

fit_range = np.arange( ti_list[STATE][BOOST], tf_list[STATE][BOOST]+1 )

for nu in range( len( t_range ) ): 

    t = t_range[nu] 

    if t in fit_range : 
        plt.errorbar( t_range[nu], check_avgs[nu], xerr=0, yerr=check_errs[nu], 
                      fmt="o", c="firebrick", zorder=24, ms=8, capsize=6, 
                      mew=1.8, elinewidth=2.5, mfc="w" )
    else: 
        plt.errorbar( t_range[nu], check_avgs[nu], xerr=0, yerr=check_errs[nu], 
                      fmt="o", c="silver", zorder=24, ms=8, capsize=6, 
                      mew=1.8, elinewidth=2.5, mfc="w" )

axs.set_ylim( 0.9850, 1.0350 ) 
axs.set_xlim( -0.25, 30.25 )

axs.hlines( 1.0, -0.3, 30.3, color="k", linewidth=1.5 )

axs.set_xlabel( r"$t$" ) 
#plt.ylabel( r"$e^{(t-t_0)E_n}\times\frac{\mathbf{v}^{(n)\dagger}C^{\rm 2pt}_{{\Sigma}}(t;\mathbf{P})\mathbf{v}^{(n)}}{\mathbf{v}^{(n)\dagger}C^{\rm 2pt}_{{\Sigma}(t_0;\mathbf{P})\mathbf{v}^{(n)}}$" )
#ylabel = ( r"$e^{(t-t_0)E_n}$"
#           + r"$\frac{ \left[ C^{\rm 2pt}_{\mathbf{\Sigma}}( t; \mathbf{P} ) \right]_{nn} }{ \left[ C^{\rm 2pt}_{\mathbf{\Sigma}}( t_0; \mathbf{P} ) \right]_{nn} }$"
#           )
ylabel = r"$e^{(t-t_0)E_n}\lambda^{(n)}$"

ylabelcolor = "k" 
if STATE==1:
    ylabelcolor = "w"

axs.set_ylabel( ylabel, color=ylabelcolor ) 

axs.text( 5, 1.0225, the_label, zorder=48 )

if BOOST==0: 
    plt.title( r"$n=$"+"{0}".format(STATE) ) 
if True: 
    dlabels = [ r"$\mathbf{0}$", r"$\mathbf{1}$", r"$\mathbf{2}$", r"$\mathbf{3}$" ]
    axs2 = axs.twinx() 
    if STATE==1: 
        label_color="k"
    else:
        label_color="w"
    axs2.set_ylabel( r"$\frac{L}{2\pi}\mathbf{P}=$"+dlabels[BOOST], color=label_color )
    right_side = axs2.spines["right"]
    right_side.set_visible(False)
    top_side = axs2.spines["top"]
    top_side.set_visible(False)
    right_side = axs2.spines["left"]
    right_side.set_visible(False)
    top_side = axs2.spines["bottom"]
    top_side.set_visible(False)
    axs2.set_yticks([])
    #axs2.set_xticks([])
    pass

as_type = "pdf"
name = work_folder+"/TwoParticle_PCorr_E{0}_P{1}.".format( STATE, BOOST )+as_type

plt.savefig( name, dpi=480, format=as_type, transparent=True,
                 bbox_inches="tight", metadata=None )

plt.show()