import numpy as np 
import matplotlib.pyplot as plt 
from iminuit import Minuit 

import sys 
sys.path.append( "/Users/markbook/Cobra" ) 

import jackknife as jk 
import OOP_TEST_v4 as model 

field = model.NLSM( TIME=256, LEN=128, COUPLING=1.54 ) 

NO_XLABEL = False 
NO_YLABEL = False

TMAX       = 12
NOF_STATES =  7  
STATE_OUT  = -1
STATE_IN   =  3 
EXCHANGE   = np.abs( STATE_OUT ) - np.abs( STATE_IN ) 
LABEL      = ( STATE_OUT, STATE_IN )

SEPARATE   = 0 
PROBE      = 0
WITH_FIT   = 1

work_folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/3pt/1P/" 

### ONE-PARTICLE SPECTRUM ### 

data_folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/1P/" 
paths       = [ data_folder+'re_one_corr_p'+str( n )+'_TAVG256.bn' for n in range( NOF_STATES ) ] 

one_particle_2pt = model.one_2pt( field ) 
one_particle_2pt.read_corrs( paths ) 

GUESS = [ [ 0.073296, 1182.5 ],
          [ 0.08827,   983.9 ],
          [ 0.1223,    707.5 ],
          [ 0.1645,    530.5 ],
          [ 0.2092,    412.9 ],
          [ 0.2559,    341.9 ] ]

fit_ranges = [ [ 13, 34 ], 
               [ 16, 33 ], 
               [ 13, 32 ], 
               [ 11, 31 ], 
               [ 13, 26 ], 
               [  9, 24 ] ]

one_particle_2pt.fit_spectrum( fit_ranges, GUESS=GUESS ) 

print( np.mean( field.MASS ) )

### ONE-PARTICLE MATRIX ELEMENTS ### 

data_folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/3pt/1P/TMAX{0}/".format( TMAX ) 

re_name = "re_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( STATE_OUT, STATE_IN, TMAX ) 
im_name = "im_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( STATE_OUT, STATE_IN, TMAX ) 

re_path = data_folder+re_name 
im_path = data_folder+im_name 

one_particle_3pt = model.one_3pt( field, one_particle_2pt, TMAX ) 
print( "HERE 57" )

one_particle_3pt.read_corrs( re_path, im_path, LABEL ) 

if EXCHANGE: 
    re_name = "re_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( STATE_IN, STATE_OUT, TMAX ) 
    im_name = "im_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( STATE_IN, STATE_OUT, TMAX ) 
    re_path = data_folder+re_name 
    im_path = data_folder+im_name 
    one_particle_3pt.read_corrs( re_path, im_path, (LABEL[1],LABEL[0]) ) 

    print( one_particle_3pt.corr_labels ) 
    #quit()

print( np.shape( one_particle_3pt.re_3pt_corrs ) )
print( np.shape( one_particle_3pt.im_3pt_corrs ) ) 
print( one_particle_3pt.corr_labels ) 

ME_jk = one_particle_3pt.compute_ratio( 0, STATE_OUT, STATE_IN, EXCHANGE=EXCHANGE ) 

if PROBE: 

    print( np.arange( 0+2, TMAX-1 ) )

    A = []

    for ti in range( -TMAX+2, 0-1 ): 

        index_ti = ti + TMAX-1
        

        A_aux, Chi2dof_aux = [], [] 

        for tf in range( 0+2, TMAX-1 ): 
            index_tf = tf + TMAX-1

            sample_jk = ME_jk[ index_ti:index_tf+1 ] 
            cov       = jk.jk_cov_matrix( sample_jk ) 
            InvCov    = np.matrix( np.linalg.inv( cov ) ) 

            A_jk, Chi2dof_jk = [], [] 

            dof = tf+1-ti-1

            for data in sample_jk.T : 

                def Chi2( A ): 
                    diffs = np.matrix( data - A ).T 
                    return ( diffs.T @ InvCov @ diffs )[0,0] 
                
                Chi2_fit = Minuit( Chi2, A=0.975 ) 

                Chi2_fit.errordef = Minuit.LEAST_SQUARES 
                Chi2_fit.strategy = 2 
                Chi2_fit.errors   = 0.0001 

                Chi2_fit.migrad() 

                A_jk.append( Chi2_fit.values[0] ) 
                Chi2dof_jk.append( Chi2_fit.fval / dof )

            A_aux.append( np.mean( A_jk ) ) 
            Chi2dof_aux.append( np.mean( Chi2dof_jk ) ) 
        
        print( Chi2dof_aux, "ti=", ti )
        A.append( A_aux )
    print()
    print("F(Q2)")
    print( np.arange( 0+1, TMAX-1 ) )
    count = -11
    for FF in A: 

        count += 1
        print( FF, "ti=", count )

    print()

    quit() 

List_of_LABELS    = [ ( 3, 4), ( 3, 5), 
                      ( 2, 3), ( 2, 4), ( 2, 5), 
                      ( 1, 2), ( 1, 3), ( 1, 4), ( 1, 5),
                      ( 0, 0), ( 0, 1), ( 0, 2), ( 0, 3), ( 0, 4), ( 0, 5),
                      (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5), 
                      (-2, 2), (-2, 3), (-2, 4), 
                      (-3, 3) ]

List_of_FITRANGES = [ (-4, 3), (-3, 4), 
                      (-7, 7), (-7, 8), (-5, 5), 
                      (-9, 8), (-3, 4), (-4, 8), (-3, 6), 
                      (-5, 6), (-5, 4), (-7, 6), (-5, 6), (-3, 7), (-7, 5),
                      (-4, 6), (-7, 7), (-6, 4), (-2, 4), (-7, 6), 
                      (-5, 8), (-4, 5), (-5, 5), 
                      (-6, 6) ] 

List_of_FFGUESS   = [ 0.9914, 0.9802, 
                      0.9982, 1.0000, 0.9986, 
                      1.0007, 0.9914, 0.9845, 0.9783, 
                      0.9969, 0.9953, 0.9814, 0.9774, 0.9612, 0.9553, 
                      0.9797, 0.9637, 0.9525, 0.9338, 0.9313, 
                      0.9332, 0.9221, 0.9125, 
                      0.9061 ]

if SEPARATE: 

    ME_one_way_jk = one_particle_3pt.compute_ratio( 0, STATE_OUT, STATE_IN, EXCHANGE=False ) 

    ME_the_other_way_jk = one_particle_3pt.compute_ratio( 1, STATE_IN, STATE_OUT, EXCHANGE=False )

print( np.shape( ME_jk ) ) 

### VISUAL ### 

plt.rcParams.update({'font.size': 20})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rc('text', usetex=True)

fig, axs = plt.subplots( 1, 1, figsize=( 6.4*(1.5+0.05), 2.4*1.5 ) )

right_side = axs.spines["right"]
right_side.set_visible(False)
top_side = axs.spines["top"]
top_side.set_visible(False)

ME_avgs = np.array( [ np.mean(x) for x in ME_jk ] )
ME_errs = np.array( [ jk.jk_std(x) for x in ME_jk ] ) 

t_range = np.arange( -TMAX+1, TMAX, 1 ) 

if SEPARATE: 

    ME_one_avgs = np.array( [ np.mean(x) for x in ME_one_way_jk ] )
    ME_two_avgs = np.array( [ np.mean(x) for x in ME_the_other_way_jk ] )

    ME_one_errs = np.array( [ jk.jk_std(x) for x in ME_one_way_jk ] )
    ME_two_errs = np.array( [ jk.jk_std(x) for x in ME_the_other_way_jk ] )

    axs.errorbar( 
        t_range, ME_one_avgs, xerr=None, yerr=ME_one_errs, fmt="o", 
        c="steelblue", zorder=24, ms=8, capsize=6, mew=1.8, 
        elinewidth=2.5, mfc="w"
     )
    
    axs.errorbar( 
        t_range, ME_two_avgs, xerr=None, yerr=ME_two_errs, fmt="o", 
        c="goldenrod", zorder=24, ms=8, capsize=6, mew=1.8, 
        elinewidth=2.5, mfc="w"
     )
    
if WITH_FIT: 

    pointer = List_of_LABELS.index( (STATE_OUT,STATE_IN) )

    the_label = List_of_LABELS[ pointer ]  
    the_range = List_of_FITRANGES[ pointer ] 
    the_guess = List_of_FFGUESS[ pointer ] 

    ti, tf = the_range

    print(  )
    one_particle_3pt.get_form_factor( the_label, EXCHANGE=EXCHANGE )
    print(  )

    one_particle_3pt.fit_form_factor( the_label, ti, tf, GUESS=the_guess ) 
    print( np.shape( one_particle_3pt.formfactor_fit ) )
    print( np.shape( one_particle_3pt.Chi2 ) )
    print( "Chi2dof=", np.mean( one_particle_3pt.Chi2[0] ) )
    print( "(ti,tf)=", List_of_FITRANGES[pointer] ) 
    print( "f(Q^2)=", np.mean( one_particle_3pt.formfactor_fit[0] ), jk.jk_std( one_particle_3pt.formfactor_fit[0] ) )  

    ## BREIT-FRAME 
    #chi2label=r"$\quad\chi^2=1.0\times( 12 - 1 )$" ## For (OUT,IN)=(0,0)
    #chi2label=r"$\quad\chi^2=1.0\times( 11 - 1 )$" ## For (OUT,IN)=(-1,1)
    #chi2label=r"$\quad\chi^2=1.0\times( 14 - 1 )$" ## For (OUT,IN)=(-2,2)
    #chi2label=r"$\quad\chi^2=1.0\times( 13 - 1 )$" ## For (OUT,IN)=(-3,3) 

    ## NOT BREIT-FRAME 
    #chi2label=r"$\quad\chi^2=0.5\times( 10 - 1 )$" ## For (OUT,IN)=(0,1)
    #chi2label=r"$\quad\chi^2=1.0\times( 14 - 1 )$" ## For (OUT,IN)=(0,2)
    #chi2label=r"$\quad\chi^2=1.3\times( 15 - 1 )$" ## For (OUT,IN)=(-1,2)
    chi2label=r"$\quad\chi^2=0.6\times( 11 - 1 )$" ## For (OUT,IN)=(-1,3)


    ff_val_jk  = one_particle_3pt.formfactor_fit[0] 
    ff_val_avg = np.mean(   ff_val_jk ) 
    ff_val_err = jk.jk_std( ff_val_jk ) 

    dt = 0.1

    t_band = np.arange( ti-dt, tf+2*dt, dt ) 

    for t in t_band: 

        ff_val_upper = np.ones( len( t_band ) ) * ( ff_val_avg + ff_val_err )
        ff_val_lower = np.ones( len( t_band ) ) * ( ff_val_avg - ff_val_err )

    axs.fill_between( t_band, ff_val_upper, ff_val_lower, 
                      interpolate=True, color="g", alpha=0.5 )

for nu in range( len( t_range ) ): 

    t = t_range[nu] 

    if t >= ti and t <= tf: 
        axs.errorbar( 
            t, ME_avgs[nu], xerr=None, yerr=ME_errs[nu], fmt="o", 
            c="firebrick", zorder=24, ms=8, capsize=6, mew=1.8, 
            elinewidth=2.5, mfc="w"
            )
    else:  
        axs.errorbar( 
            t, ME_avgs[nu], xerr=None, yerr=ME_errs[nu], fmt="o", 
            c="silver", zorder=24, ms=8, capsize=6, mew=1.8, 
            elinewidth=2.5, mfc="w"
            )

axs.set_xlabel( r"$t'$" ) 
axs.set_ylabel( r"$f(Q^2)$" ) 

if NO_XLABEL: axs.set_xlabel( r"$t'$", c="w" ) 
if NO_YLABEL: axs.set_ylabel( r"$f(Q^2)$", c="w" ) 

d_LABELS = [ r"$-\mathbf{5}$", r"$-\mathbf{4}$", r"$-\mathbf{3}$", r"$-\mathbf{2}$", r"$-\mathbf{1}$", r"$\mathbf{0}$", r"$\mathbf{1}$", r"$\mathbf{2}$", r"$\mathbf{3}$", r"$\mathbf{4}$", r"$\mathbf{5}$" ]

axs.set_title( r"$\frac{L}{2\pi}(\mathbf{p}_f,\mathbf{p}_i)=($"+d_LABELS[5+STATE_OUT]+r"$,$"+d_LABELS[5+STATE_IN]+r"$)$"+chi2label )

as_type = "pdf" 

name = work_folder + "1P_ME_OUT"+str(STATE_OUT)+"_IN"+str(STATE_IN)+"."+as_type 

plt.savefig( name, dpi=720, format=as_type, transparent=True, 
             bbox_inches="tight", metadata=None )

plt.show()






