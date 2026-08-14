import numpy as np 
import matplotlib.pyplot as plt 
from iminuit import Minuit 

import sys 
sys.path.append( "/Users/markbook/Cobra" ) 

import jackknife as jk 

import OOP_TEST_v4 as model 

field = model.NLSM() 

## C2pt PART 

one_particle_2pt = model.one_2pt( field )  

NOF_STATES = 7

folder = '/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/1P/' 
paths  = [ folder+'re_one_corr_p'+str( n )+'_TAVG256.bn' for n in range( NOF_STATES ) ] 

one_particle_2pt.read_corrs( paths ) 

''' BOLD TUNED '''
fit_ranges = [ ( 13, 34 ),  ## May go further 
               ( 16, 34 ), 
               ( 13, 32 ), 
               ( 11, 31 ), 
               ( 15, 31 ), 
               ( 12, 26 ),
               ( 10, 20 )
             ][:NOF_STATES]

guess_E = [ 0.073295, 0.08835, 0.1223, 0.1645, 0.2092, 0.2556, 0.3037 ] 
guess_A = [   1182.5,   985.6,  707.5,  530.5,  530.5,  413.7,  340.5 ] 

GUESS = np.array( [ guess_E, guess_A ] ).T 

one_particle_2pt.fit_spectrum( fit_ranges, GUESS, CHI2dof=True ) 

T_SEP = 12 

labels = [ ( 0, 0), ( 1,-1), ( 2,-2), ( 3,-3),
           ( 1, 0), ( 2, 0), ( 3, 0), ( 4, 0), ( 5, 0), 
           (-2, 1), ( 2, 1), ( 3, 1), ( 4, 1), ( 5, 1), 
           ( 3, 2), ( 4, 2), ( 5, 2), 
           (-2, 3), (-1, 3), ( 4, 3), ( 5, 3), 
           (-1, 4), (-2, 4),
           (-1, 5) ] 

folder  = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/3pt/1P/TMAX{0}/".format(T_SEP) 

one_particle_3pt = model.one_3pt( field, one_particle_2pt, T_SEP ) 

for OUT, IN in labels : 

    re_name = "re_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( OUT, IN, T_SEP ) 
    im_name = "im_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( OUT, IN, T_SEP ) 
    one_particle_3pt.read_corrs( folder+re_name, folder+im_name, ( OUT, IN ) ) 

for OUT, IN in labels : 
    re_name = "re_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( IN, OUT, T_SEP ) 
    im_name = "im_1P1_corr_N_OUT_{0}_N_IN_{1}_TMAX{2}_TAVG256_new.bn".format( IN, OUT, T_SEP ) 
    one_particle_3pt.read_corrs( folder+re_name, folder+im_name, ( IN, OUT ) ) 

for OUT, IN in labels: 
    one_particle_3pt.get_form_factor( (OUT,IN), EXCHANGE=True ) 

fit_ranges =  [ (-4,3), (-4,6), (-5,6), (-6,3), (-5,4), (-7,6), 
                (-5,5), (-3,5), (-4,4), (-7,7), (-8,6), (-4,4), 
                (-4,5), (-3,6), (-7,5), (-4,5), (-5,5), (-4,5), 
                (-6,4), (-4,3), (-4,4), (-2,5), (-5,5), (-5,5) ] 

GUESS = [ 0.995, 0.980, 0.932, 0.911, 0.995, 0.981, 
          0.978, 0.960, 0.959, 0.964, 1.000, 0.991, 
          0.986, 0.983, 1.000, 0.994, 1.002, 0.922, 
          0.953, 0.990, 0.978, 0.933, 0.912, 0.936  ]

for nu, label in enumerate( labels ): 

    print( nu )

    ti, tf = fit_ranges[nu][0], fit_ranges[nu][1] 

    one_particle_3pt.fit_form_factor( label, ti, tf, GUESS=GUESS[nu] )




plt.rcParams.update({'font.size': 22})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rc('text', usetex=True)

fig, axs = plt.subplots( 1, 1, figsize=( 6.4*(1.5+0.05), 3.2*1.5 ) )

right_side = axs.spines["right"]
right_side.set_visible(False)
top_side = axs.spines["top"]
top_side.set_visible(False)

ff_mns_list = [] 
ff_err_list = [] 
Q2_by_m2_mns_list = [] 
Q2_by_m2_err_list = [] 

M2 = field.MASS * field.MASS

for nu in range( len( one_particle_3pt.Q2 ) ): 

    

    Q2_by_m2_mns = np.mean( one_particle_3pt.Q2[nu].real/M2 ) 

    Q2_by_m2_err = None 
    if Q2_by_m2_mns: Q2_by_m2_err = jk.jk_std( 0.5*np.sqrt(np.abs(one_particle_3pt.Q2[nu].real)/M2) ) 

    ff_mns = np.mean( one_particle_3pt.formfactor_fit[nu].real ) 
    ff_err = jk.jk_std( one_particle_3pt.formfactor_fit[nu].real ) 

    axs.errorbar( 0.5*np.sqrt(Q2_by_m2_mns), ff_mns, xerr=Q2_by_m2_err, yerr=ff_err, 
                  fmt="o", c="firebrick", ms=8, capsize=6, mew=1.8, 
                  mec="firebrick", elinewidth=2.5, mfc="w", zorder=6 ) 
    
    print( nu, Q2_by_m2_mns )
    
    if Q2_by_m2_mns < 2.6:
    
        Q2_by_m2_mns_list.append( Q2_by_m2_mns ) 
        if Q2_by_m2_mns:
            Q2_by_m2_err_list.append( Q2_by_m2_err ) 
        else:
            Q2_by_m2_err_list.append( 0 )
        ff_mns_list.append( ff_mns ) 
        ff_err_list.append( ff_err )
    
additional = [ ( 0, 4), ( 1, 4), ( 2, 4), ( 3, 4), (-1, 4), 
               (-2, 4), ( 0, 5), ( 1, 5), ( 2, 5), ( 3, 5), 
               ( 4, 5), (-1, 5) ] 

for OUT, IN in additional : 

    E_out = one_particle_2pt.spectrum[ np.abs( OUT ) ] 
    E_in  = one_particle_2pt.spectrum[ np.abs( IN  ) ]  

    Q2 = np.mean( np.square( (OUT-IN) * field.UNIT ) - np.square( E_out-E_in ) )

    #axs.vlines( Q2/M2, 0.9, 1.0, colors="k", linestyles="--" )
    
axs.set_ylabel( r"$f(Q^2)$" ) 
axs.set_xlabel( r"$\frac{1}{2}\sqrt{Q^2/m^2}$" )

#plt.show()


print( np.shape( one_particle_3pt.formfactor_fit ) )
print( np.shape( one_particle_3pt.Q2 ) )

samples_jk = np.array( one_particle_3pt.formfactor_fit )

Q2_jk      = np.array( one_particle_3pt.Q2 ) 

cov = jk.jk_cov_matrix( samples_jk ) 

InvCov = np.matrix( np.linalg.inv( cov ) ) 

def hypth( Q2, mu2, A ): 

    return 1.0 / ( Q2 / mu2 + 1.0 ) + A * Q2 
    
    #return ( 1.0 + A * Q2 ) / ( Q2 / mu2 + 1.0 ) 

mu2_jk = [] 
A_jk   = [] 
Chi2_jk = [] 

for nu, data in enumerate( samples_jk.T ): 

    Q2 = Q2_jk.T[nu] 

    dof = len( data ) - 2.0

    def Chi2( mu2, A ): 
        diff = np.matrix( data - hypth( Q2, mu2, A ) ).T 
        return ( diff.T @ InvCov @ diff )[0,0] 
    
    Chi2_fit = Minuit( Chi2, mu2=0.236, A=2.10 ) 

    Chi2_fit.errordef = Minuit.LEAST_SQUARES 
    Chi2_fit.strategy = 2 
    Chi2_fit.errors   = 0.001

    Chi2_fit.migrad() 

    mu2_jk.append( Chi2_fit.values[0] )
    A_jk.append( Chi2_fit.values[1] ) 
    Chi2_jk.append( Chi2_fit.fval/dof )

mu2_jk = np.array( mu2_jk ) 
A_jk   = np.array( A_jk ) 

print( "mu/m    : ", np.mean( np.sqrt(mu2_jk) / field.MASS ), jk.jk_std( np.sqrt(mu2_jk) / field.MASS ) )
print( "sqrt(A)m: ", np.mean( field.MASS * np.sqrt(A_jk) ), jk.jk_std( np.sqrt(A_jk) * field.MASS ) )
print( "Chi2/dof: ", np.mean( Chi2_jk ), jk.jk_std( np.array(Chi2_jk) ) )


####
one_particle_3pt.fit_form_factor_in_Q2()
####

#pars_jk = one_particle_3pt.formfactor_pars.T 

#print( np.mean( pars_jk[0] ), jk.jk_std( pars_jk[0] ) )
#print( np.mean( pars_jk[1] ), jk.jk_std( pars_jk[1] ) )

folder  = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/3pt/1P/"

one_particle_3pt.save_data( folder=folder )

#quit()

dk = 0.0115*0.125
k_range = np.arange( 0.00000001, 2.*np.pi*3.1/float(field.LEN)+dk, dk ) 
s_range = 4.0 * ( np.square( k_range ) + M2 ) 

analytic_ff_1P = field.form_factor_1P( s_range ) 
Q2_analytic_range = 4.0 * np.square( k_range )

axs.plot( 0.5*np.sqrt(Q2_analytic_range)/np.sqrt(M2), analytic_ff_1P, 
         '-', lw="2", c="k", label="Analytic", zorder=-36 )

ff_fit_jk = [] 

Q2_range = np.arange( 0.000001, 0.09025, 0.00025 )

for nu in range( 1000 ): 

    ff_fit_jk.append( [ hypth( Q2, mu2_jk[nu], A_jk[nu] ) for Q2 in Q2_range ] )


ff_fit_jk = np.array( ff_fit_jk ).T 

ff_fit_upper, ff_fit_lower = [], [] 

for x in ff_fit_jk:
    avg = np.mean( x )
    err = jk.jk_std( x ) 

    ff_fit_upper.append( avg + err )
    ff_fit_lower.append( avg - err ) 

axs.fill_between( 0.5*np.sqrt(Q2_range)/np.sqrt(M2), ff_fit_upper, ff_fit_lower, color="g", 
                  alpha=0.5, label=r"$\frac{1}{Q^2/\mu^2+1}+AQ^2$", zorder=-24 ) 

axs.set_xlim( -0.001, 2.05 ) 
axs.set_ylim( 0.8725, 1.015 )

#axs.text( 0.0, 0.94,  r"$\mu^2=0.236(16)$" )
#axs.text( 0.17, 0.925, r"$A=2.09(21)$" )
#axs.text( 0.06, 0.910,  r"$\frac{\chi^2}{\rm dof}=1.2$" ) 

#plt.legend( loc="lower left" )

#### FOR ZOOMING
#zm = axs.inset_axes( [ 8.0, 9.5, 2.2, 0.035 ] ) 
#zm = axs.inset_axes( [ 0.60, 0.65, 0.35, 0.30 ] ) 
#
#print( Q2_range[99:100]/M2, Q2_range[49:50]/M2 )
#
#zm.fill_between( Q2_range[:50]/M2, ff_fit_upper[:50], 
#                 ff_fit_lower[:50], color="g", alpha=0.65  )
#
#zm.errorbar( Q2_by_m2_mns_list, ff_mns_list, 
#             xerr=Q2_by_m2_err_list, yerr=ff_err_list, 
#             fmt="o", c="firebrick", ms=8, capsize=6, mew=1.8, 
#             mec="firebrick", elinewidth=2.5, mfc="w", zorder=24 ) 
#
#zm.set_xlim( -0.1, 2.0 ) 
#zm.set_ylim( 0.980, 1.005 )

folder  = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/3pt/1P/"
as_type = "pdf"
name = folder+"FormFactor_1P_vs_k."+as_type

transparent=True 
if as_type=="png" : transparent=False 


plt.savefig( name, dpi=480, format=as_type, transparent=transparent,
             bbox_inches="tight", metadata=None ) 

plt.show()
