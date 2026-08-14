import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path 

import sys 
sys.path.append( "/Users/markbook/Cobra/" ) 

import jackknife as jk 
import OOP_TEST_v4 as QFT 

field = QFT.NLSM( TIME=256, LEN=128, COUPLING=1.54 ) 

one_2pt = QFT.one_2pt( field ) 


name = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/1P/re_one_corr_p{0}_TAVG256.bn"

paths = [ name.format( n ) for n in range(1) ]  

one_2pt.read_corrs( paths ) 

GUESS = [ [ 0.07329, 1183 ],
          [ 0.08825,  303 ],
          [ 0.12565,  218 ],
          [ 0.16455,  162 ],
          [ 0.20965,  125 ] ][:1]

fit_ranges = [ [ 13, 25 ], 
               [ 16, 25 ], 
               [ 14, 25 ], 
               [ 11, 19 ], 
               [ 12, 20 ] ][:1]

one_2pt.fit_spectrum( fit_ranges, GUESS=GUESS )

print( "MASS: ", np.mean(field.MASS) )



TNOT = 10 

ti_list = [ [  9, 11,  9, 11 ],
            [  7,  9,  9,  9 ],
            [  6,  9,  8,  8 ] ] 

tf_list = [ [ 37, 38, 35, 35 ],
            [ 30, 37, 32, 33 ],
            [ 20, 26, 20, 26 ] ]

work_folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/2pt/2P" 

E_avgs_list = [] 
E_errs_list = [] 

for BOOST in [ 0, 1, 2, 3 ]: 

    aux_avgs = [] 
    aux_errs = [] 

    print()

    spectrum = [] 

    print( "N =", BOOST, "SPECTRUM: " )

    folder_BOOST = "/Spectrum/N"+str(BOOST)+"/TNOT_is_"+str(TNOT) 

    aux_Ecm = [] 
    aux_dEcm = [] 

    for STATE in range( 3 ): 

        #if BOOST==3 and STATE==0 : continue

        ti = ti_list[ STATE ][ BOOST ] 

        folder = work_folder + folder_BOOST + "/ti_is_"+str(ti)
        path   = folder + "/E_{}".format( STATE ) 

        aux_E = [] 

        for line in open( path, 'r' ): 

            aux_E.append( np.fromstring( line, dtype=float, sep=" " )[0] ) 

        aux_E = np.array( aux_E )

        E_cm = np.sqrt( aux_E*aux_E - np.square( field.UNIT * BOOST ) ) / field.MASS 

        print( np.mean( E_cm ), jk.jk_std( E_cm ) )

        aux_Ecm.append( np.mean( E_cm ) ) 
        aux_dEcm.append( jk.jk_std( E_cm ) ) 

    E_avgs_list.append( aux_Ecm ) 
    E_errs_list.append( aux_dEcm )

L_range = np.arange( 127.0, 129.0, 0.1 )

m2 = np.mean( field.MASS )**2 
        
s_range = np.arange( 2.001, 6.699, 0.01 )**2 * m2 

spectrum_from_theory = [] 

for BOOST in range( 4 ): 

    aux = [ [] for n in range( 5 ) ] 

    for ell in L_range:

        aux_field = QFT.NLSM( TIME=256, LEN=ell, COUPLING=1.54 ) 
        aux_field.MASS = field.MASS
        
        delta = aux_field.phase_shift_I1( s_range ) 
        check = aux_field.get_spectrum( s_range, BOOST, field.phase_shift_I1, n_max=4 ) 

        for x in check: 
            
            if x: 
                if x[1]>5: break
                aux[ x[1] ].append( x[0]/np.sqrt(m2) ) 

    aux_2 = [] 
    
    for nonzero in aux: 

        if nonzero: aux_2.append( nonzero ) 

    spectrum_from_theory.append( aux_2 )


free_spectrum_from_theory = [] 

for BOOST in range( 4 ): 

    aux = [ [] for n in range( 5 ) ] 

    for ell in L_range:

        aux_field = QFT.NLSM( TIME=256, LEN=ell, COUPLING=1.54 ) 
        aux_field.MASS = field.MASS
    
        
    
        
        #delta   = field.phase_shift_I1( k_range, m=np.sqrt(m2) ) 
        check = aux_field.get_spectrum( s_range, BOOST, field.phase_shift_I1, n_max=4, FREE=True ) 

        for x in check: 
            if x: 
                if x[1]>5: break
                aux[ x[1] ].append( x[0]/np.sqrt(m2) ) 

    aux_2 = [] 
    
    for nonzero in aux: 

        if nonzero: aux_2.append( nonzero ) 

    free_spectrum_from_theory.append( aux_2 )


    #if BOOST>1: quit() 


####
####

plt.rcParams.update({'font.size': 22})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold' 

plt.rc('text', usetex=True)

centimeters = 1.0 / 2.54

fig, axs = plt.subplots( 1, 4, figsize=( 3.5*centimeters*2, 4.8*centimeters*2 ) ) 

the_titles = [ r"$\mathbf{d}=\mathbf{0}$", 
               r"$\mathbf{d}=\mathbf{1}$", 
               r"$\mathbf{d}=\mathbf{2}$", 
               r"$\mathbf{d}=\mathbf{3}$" ]

MASS = np.mean( field.MASS )

def naive_ERE( pars, s ): 

    k_cm = np.sqrt( 0.25 * s - MASS*MASS ) 

    c0 = pars[0] # THIS IS -1/a 
    c1 = pars[1] # THIS IS  r/2 

    return c0 + c1 * np.square( k_cm )


def K_inv_ERE( pars, s ): 

    rho = field.phase_space( s ) 
    kcm = np.sqrt( 0.25 * s - MASS*MASS )

    return rho * naive_ERE( pars, s ) / kcm 



mL=9.381
dmL = 0.007

mL_range = L_range * np.mean(field.MASS)

ERE_data = [ line for line in open( work_folder+"/EffRange/ERE_pars.jk", "r" ) ]
ERE_pars = [ np.fromstring( line, dtype=float, sep=", " ) for line in ERE_data[1:] ]  
ERE_pars = np.array( ERE_pars ) 

the_legends=[ None, None, None, None ]  

the_markers_per_boost = [ 'o', '^', 's', 'p' ]

for boost in range( 4 ): 

    ########
    ########
    ##
    ##

    ERE_spectrum_at_boost_above = [ [], [], [] ] 
    ERE_spectrum_at_boost_below = [ [], [], [] ] 
    

    for mL_val in mL_range: 

        aux_field = QFT.NLSM( LEN=mL_val/np.mean( field.MASS ), MASS=np.mean( field.MASS ) )

        ERE_spectrum_at_mL = [ [], [], [] ] 
        #
        for pars in ERE_pars: 

            def delta_ERE( s ):  
                A,R = pars[0], pars[1]
                K_inv = K_inv_ERE( [A,R], s ) 
                K = 1.0 / K_inv 
                rho = field.phase_space( s ) 
                return np.arctan( rho * K ) 

            ERE_spectrum_at_pars = aux_field.get_spectrum( s_range, boost, delta_ERE, n_max=4, steps=100 )

            counter = 0 
            for val in ERE_spectrum_at_pars:
                if val: 
                    ERE_spectrum_at_mL[counter].append( val[0] )
                    counter += 1
                if counter>2: break 

        #print( np.shape(ERE_spectrum_at_mL ) )
        #print() 

        ERE_E0_at_mL_jk = 0

        counter = 0
        for ERE_En_jk in ERE_spectrum_at_mL: 

            ERE_En_avg = np.mean( np.array( ERE_En_jk )/np.mean(field.MASS) ) 
            ERE_En_err = jk.jk_std( np.array( ERE_En_jk )/np.mean(field.MASS) )

            ERE_spectrum_at_boost_above[counter].append( ERE_En_avg+ERE_En_err )
            ERE_spectrum_at_boost_below[counter].append( ERE_En_avg-ERE_En_err )

            counter += 1

    #print( np.shape( ERE_spectrum_at_boost_above ) )
    #quit()

    #for nu in range(3):  
    #    axs[boost].fill_between( mL_range, ERE_spectrum_at_boost_above[nu], 
    #                             ERE_spectrum_at_boost_below[nu],
    #                             interpolate=True, color='g', alpha=0.8, 
    #                             zorder=100
    #                              )

    ##
    ##
    ########
    ########

    if boost==3: 
        the_legends = [ "Theory", "Free", "Lattice", "Eff. Range" ]

    for E_th in spectrum_from_theory[boost][:4]: 
        axs[boost].plot( mL_range, E_th, c="k", ls="-", lw=2, zorder=12 )

    for E_th in free_spectrum_from_theory[boost][:4]:  
        print( len(mL_range), len(E_th) )
        axs[boost].plot( mL_range, E_th, c="k", ls=":", lw=2, zorder=12 )

    E_avgs = E_avgs_list[boost] 
    E_errs = E_errs_list[boost] 

    mL_arr  = np.ones( len( E_avgs ) ) * mL 
    dmL_arr = np.ones( len( E_avgs ) ) * dmL

    #axs[boost].errorbar( mL_arr, E_avgs, xerr=None, yerr=E_errs, #xerr=dmL_arr, yerr=E_errs, 
    #                     fmt="o", c="firebrick", ms=8, capsize=10, mew=2.0, 
    #                     elinewidth=2.5, mfc="w", zorder=24 ) 
    
    axs[boost].errorbar( mL_arr, E_avgs, xerr=None, yerr=E_errs, #xerr=dmL_arr, yerr=E_errs, 
                         fmt=the_markers_per_boost[boost], c="firebrick", ms=8, capsize=10, mew=2.0, 
                         elinewidth=2.5, mfc="w", zorder=24 ) 

    #for nu in range( len( E_avgs ) ): 
    #    axs[boost].errorbar( mL,  )

    right_side = axs[boost].spines["right"]
    right_side.set_visible(False)
    top_side = axs[boost].spines["top"]
    top_side.set_visible(False)

    axs[boost].set_ylim( 1.9, 3.9 )
    axs[boost].set_yticks( [2,3] )
    #axs[boost].set_xticks( [9.38] )

    if boost: axs[boost].set_yticklabels( [ "", "" ] )

    if not boost: axs[boost].set_ylabel( r"$E^\star/m$" )

    
    #if not nu: 
    #    axs[nu].tick_params( axis="y", colors="k" )
    #else:
    #    axs[nu].tick_params( axis="y", colors="w" )

    axs[boost].set_xlim( 9.37, 9.39 )  

    axs[boost].set_xticks( [ 9.38 ] )

    axs[boost].set_xticklabels( [[ r"$\mathbf{0}$", 
                                   r"$\mathbf{1}$", 
                                   r"$\mathbf{2}$", 
                                   r"$\mathbf{3}$" ][boost]] )

    #if not boost: 
    #    axs[boost].hlines( y=2.0, xmin=9.25, xmax=9.5, colors="k", ls=":", lw=2 )

    #axs[boost].set_xticks( [ 9.38 ] )

    #axs[boost].set_title( the_titles[boost] )

    if boost==3: pass

axs[3].set_xlabel( r"$\frac{L}{2\pi}\mathbf{P}^{~}$" )

as_type = "pdf"
name = work_folder+"/Spectrum."+as_type 

plt.savefig( name, dpi=720, format=as_type, transparent=True, 
             bbox_inches="tight", metadata=None )

plt.show()




